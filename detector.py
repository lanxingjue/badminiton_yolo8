"""
动作检测器 - 负责从视频中提取关键点和分类动作
Linus原则：一个文件做一件事，做好它
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
import warnings

from core import Keypoints, BadmintonShot
from config import MODEL_CONFIG, RAW_CLASSES, TRAINING_CONFIG
# from trainer import 
# 忽略YOLO的警告信息
warnings.filterwarnings("ignore", category=UserWarning)

class BadmintonDetector:
    """
    羽毛球动作检测器
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
        
        # 初始化姿态检测模型
        try:
            self.pose_model = YOLO(pose_model_path)
            print(f"✅ 姿态检测模型加载成功: {pose_model_path}")
        except Exception as e:
            print(f"❌ 姿态检测模型加载失败: {e}")
            raise
        
        # 构建动作分类器网络
        self.action_classifier = self._build_action_classifier()
        
        # 如果提供了训练好的模型，加载权重
        if action_model_path and os.path.exists(action_model_path):
            try:
                self.action_classifier.load_state_dict(torch.load(action_model_path, map_location='cpu'))
                self.action_classifier.eval()
                print(f"✅ 动作分类模型加载成功: {action_model_path}")
            except Exception as e:
                print(f"⚠️  动作分类模型加载失败，使用未训练模型: {e}")
        else:
            print("⚠️  未提供动作分类模型，使用未训练模型")
        
        # 初始化帧缓冲区
        self.frame_buffer: List[Keypoints] = []
        self.buffer_size = MODEL_CONFIG['sequence_length']
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_classifier.to(self.device)
        
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
            torch.nn.ReLU(),
            torch.nn.Dropout(MODEL_CONFIG['dropout_rate']),
            
            # 隐藏层：256维 -> 128维
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
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
        
        Returns:
            BadmintonShot分析结果
        """
        try:
            # 准备输入数据：将关键点序列展平为一维向量
            sequence_data = []
            for keypoints in self.frame_buffer:
                sequence_data.extend(keypoints.points.flatten())
            
            # 转换为PyTorch张量
            input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            
            # 前向推理
            with torch.no_grad():
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
        max_frames = TRAINING_CONFIG['max_frames_per_video']
        
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
