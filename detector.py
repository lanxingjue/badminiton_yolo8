"""
动作检测器 - 负责从视频中提取关键点和分类动作
Linus原则：一个文件做一件事，做好它
集成GPU优化和成功的人体选择算法
"""

import cv2
import torch
import torch.cuda.amp as amp
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
import warnings

from core import Keypoints, BadmintonShot
from config import MODEL_CONFIG, RAW_CLASSES, TRAINING_CONFIG

# 忽略YOLO的警告信息
warnings.filterwarnings("ignore", category=UserWarning)

def safe_float(value):
    """
    安全的数值转换，处理所有numpy类型
    从trainer.py集成的函数
    """
    if isinstance(value, (np.ndarray, np.generic)):
        if hasattr(value, 'item'):
            return float(value.item())
        elif len(value.shape) == 0:  # 0维数组（标量）
            return float(value)
        elif value.size == 1:  # 只有一个元素
            return float(value.flat[0])
        else:
            raise ValueError(f"Cannot convert array of size {value.size} to scalar")
    else:
        return float(value)

def select_nearest_person_keypoints(results, frame_height=640, frame_width=640):
    """
    从YOLOv8姿态检测结果中选择最靠近摄像头的人
    从trainer.py集成的成功算法
    """
    if not results or len(results) == 0:
        return None, None
    
    result = results[0]
    
    if not hasattr(result, 'keypoints') or result.keypoints is None:
        return None, None
    
    keypoints_data = result.keypoints
    
    if len(keypoints_data.xy) == 0:
        return None, None
    
    best_idx = None
    max_score = 0.0
    max_possible_area = float(frame_height * frame_width)
    
    for i in range(len(keypoints_data.xy)):
        try:
            coords = keypoints_data.xy[i].cpu().numpy().astype(np.float64)
            confidence = keypoints_data.conf[i].cpu().numpy().astype(np.float64)
            
            # 过滤低置信度关键点
            valid_mask = confidence > 0.3
            valid_points = coords[valid_mask]
            valid_conf = confidence[valid_mask]
            
            if len(valid_points) < 5:  # 至少需要5个高置信度关键点
                continue
            
            # 计算包围盒面积
            min_xy = valid_points.min(axis=0)
            max_xy = valid_points.max(axis=0)
            
            bbox_width = safe_float(max_xy[0] - min_xy[0])
            bbox_height = safe_float(max_xy[1] - min_xy[1])
            bbox_area = bbox_width * bbox_height
            
            # 计算质心位置
            centroid_y = safe_float(valid_points[:, 1].mean())
            position_score = centroid_y / frame_height
            
            # 计算平均置信度
            avg_confidence = safe_float(valid_conf.mean())
            
            # 综合评分：面积50% + 位置30% + 置信度20%
            area_weight = 0.5
            position_weight = 0.3
            confidence_weight = 0.2
            
            # 面积归一化
            if bbox_area > 0:
                normalized_area = min(np.log10(bbox_area + 1) / np.log10(max_possible_area + 1), 1.0)
            else:
                normalized_area = 0.0
            
            normalized_position = min(position_score, 1.0)
            normalized_confidence = min(avg_confidence, 1.0)
            
            composite_score = (
                area_weight * normalized_area +
                position_weight * normalized_position +
                confidence_weight * normalized_confidence
            )
            
            if composite_score > max_score:
                max_score = composite_score
                best_idx = i
                
        except Exception as e:
            continue
    
    if best_idx is None:
        return None, None
    
    # 返回最佳候选人的关键点和置信度
    best_keypoints = keypoints_data.xy[best_idx].cpu().numpy()
    best_confidence = keypoints_data.conf[best_idx].cpu().numpy()
    
    return best_keypoints, best_confidence

class BadmintonDetector:
    """
    羽毛球动作检测器 - GPU优化版本
    
    职责：
    1. 从视频帧中提取人体关键点
    2. 对关键点序列进行动作分类
    3. 生成动作质量分析结果
    """
    
    def __init__(self, pose_model_path: str = 'yolov8n-pose.pt',
                 action_model_path: Optional[str] = None):
        """
        初始化检测器
        
        Args:
            pose_model_path: YOLOv8姿态检测模型路径
            action_model_path: 训练好的动作分类模型路径
        """
        print("初始化羽毛球动作检测器...")
        
        # 🔧 GPU优化：根据硬件选择合适的姿态检测模型
        if torch.cuda.is_available():
            # GPU模式使用更大更精确的模型
            if pose_model_path == 'yolov8n-pose.pt':
                pose_model_path = 'yolov8m-pose.pt'
            print(f"🎯 GPU模式：使用更大的姿态检测模型 {pose_model_path}")
        else:
            print(f"🎯 CPU模式：使用轻量级姿态检测模型 {pose_model_path}")
        
        # 初始化姿态检测模型
        try:
            self.pose_model = YOLO(pose_model_path)
            print(f"✅ 姿态检测模型加载成功: {pose_model_path}")
        except Exception as e:
            print(f"❌ 姿态检测模型加载失败: {e}")
            raise
        
        # 构建动作分类器网络
        self.action_classifier = self._build_action_classifier()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_classifier.to(self.device)
        
        # 🔧 混合精度支持
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            print("⚡ 启用混合精度推理加速")
        
        # 如果提供了训练好的模型，加载权重
        if action_model_path and os.path.exists(action_model_path):
            try:
                self.action_classifier.load_state_dict(torch.load(action_model_path, map_location=self.device))
                self.action_classifier.eval()
                print(f"✅ 动作分类模型加载成功: {action_model_path}")
            except Exception as e:
                print(f"⚠️ 动作分类模型加载失败，使用未训练模型: {e}")
        else:
            print("⚠️ 未提供动作分类模型，使用未训练模型")
        
        # 初始化帧缓冲区
        self.frame_buffer: List[Keypoints] = []
        self.buffer_size = MODEL_CONFIG['sequence_length']
        
        print(f"🔧 使用设备: {self.device}")
        print(f"🔧 序列缓冲区大小: {self.buffer_size}")
    
    def _build_action_classifier(self) -> torch.nn.Module:
        """
        构建动作分类器网络
        Linus原则：简单胜过复杂，可理解胜过聪明
        """
        input_dim = MODEL_CONFIG['keypoints'] * 2 * MODEL_CONFIG['sequence_length']  # 17*2*10=340
        
        return torch.nn.Sequential(
            # 输入层：340维 -> 256维
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),  # 🔧 使用inplace操作节省内存
            torch.nn.Dropout(MODEL_CONFIG['dropout_rate']),
            
            # 隐藏层：256维 -> 128维
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(MODEL_CONFIG['dropout_rate']),
            
            # 输出层：128维 -> 18类
            torch.nn.Linear(128, 18)
        )
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[Keypoints]:
        """
        从单帧图像中提取人体关键点
        集成优化的人体选择算法
        """
        try:
            # YOLOv8姿态检测
            results = self.pose_model(frame, verbose=False)
            
            # 🎯 使用相同的人体选择算法
            best_keypoints, best_confidence = select_nearest_person_keypoints(
                results, frame.shape[0], frame.shape[1]
            )
            
            if best_keypoints is None or best_confidence is None:
                return None
            
            # 检查置信度是否足够高
            avg_confidence = safe_float(best_confidence.mean())
            if avg_confidence < TRAINING_CONFIG['min_confidence_threshold']:
                return None
            
            return Keypoints(points=best_keypoints, confidence=best_confidence)
            
        except Exception as e:
            print(f"关键点提取失败: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Optional[BadmintonShot]:
        """
        处理单帧图像，返回动作分析结果（当缓冲区满时）
        
        Args:
            frame: 输入图像帧
            
        Returns:
            BadmintonShot对象或None
        """
        # 调整帧大小到模型输入尺寸
        resized_frame = cv2.resize(frame, MODEL_CONFIG['input_size'])
        
        # 提取关键点
        keypoints = self.extract_keypoints(resized_frame)
        
        if keypoints is None:
            return None
        
        # 更新序列缓冲区
        self.frame_buffer.append(keypoints)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)  # 移除最旧的帧
        
        # 当缓冲区满时进行动作分类
        if len(self.frame_buffer) == self.buffer_size:
            return self._classify_action()
        
        return None
    
    def _classify_action(self) -> BadmintonShot:
        """
        对缓冲区中的关键点序列进行动作分类
        GPU优化版本
        
        Returns:
            BadmintonShot分析结果
        """
        try:
            # 准备输入数据：将关键点序列展平为一维向量
            sequence_data = []
            for keypoints in self.frame_buffer:
                sequence_data.extend(keypoints.points.flatten())
            
            # 转换为PyTorch张量
            input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device, non_blocking=True)
            
            # 前向推理
            with torch.no_grad():
                if self.use_amp:
                    # 🔧 使用混合精度推理
                    with amp.autocast():
                        outputs = self.action_classifier(input_tensor)
                else:
                    outputs = self.action_classifier(input_tensor)
                
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
            
            # 创建分析结果
            result = BadmintonShot.from_raw_class(
                raw_class=predicted_class,
                keypoints_seq=self.frame_buffer.copy(),
                classification_confidence=confidence
            )
            
            return result
            
        except Exception as e:
            print(f"动作分类失败: {e}")
            # 返回默认结果
            return BadmintonShot.from_raw_class(
                raw_class=0,  # 默认为短发球
                keypoints_seq=self.frame_buffer.copy(),
                classification_confidence=0.0
            )
    
    def process_video(self, video_path: str) -> List[BadmintonShot]:
        """
        处理整个视频文件，返回所有检测到的动作
        GPU优化版本
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            BadmintonShot结果列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return []
        
        results = []
        frame_count = 0
        max_frames = TRAINING_CONFIG['max_frames_per_video'] * 2  # GPU可以处理更多帧
        
        print(f"开始处理视频: {video_path}")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame)
            if result is not None:
                results.append(result)
                print(f"检测到动作: {result.category_name} (置信度: {result.confidence:.2f})")
            
            frame_count += 1
            
            # 显示进度
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count} 帧...")
        
        cap.release()
        print(f"视频处理完成，共检测到 {len(results)} 个动作")
        return results
    
    def reset_buffer(self):
        """重置帧缓冲区"""
        self.frame_buffer.clear()
    
    def get_buffer_status(self) -> dict:
        """获取缓冲区状态信息"""
        return {
            'current_size': len(self.frame_buffer),
            'max_size': self.buffer_size,
            'is_ready': len(self.frame_buffer) == self.buffer_size
        }

# 导入os模块
import os
