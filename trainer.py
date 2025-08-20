"""
模型训练器 - 训练羽毛球动作分类模型
Linus原则：工具要简单、可靠、可预测
集成了成功的人体选择算法 + GPU优化支持 + 稳定性增强
"""

import os
import glob
import cv2
import torch
import torch.cuda.amp as amp  # 🔧 混合精度训练
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random
import time
import json
from datetime import datetime
from ultralytics import YOLO

from detector import BadmintonDetector
from core import Keypoints
from config import MODEL_CONFIG, TRAINING_CONFIG, RAW_CLASSES

def get_optimal_config():
    """
    根据硬件自动优化配置
    针对RTX 4090等高端GPU进行优化
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🚀 检测到GPU: {gpu_name}")
        print(f"💾 显存: {vram_gb:.1f}GB")
        
        # RTX 4090优化配置 - 修复配置冲突
        if "4090" in gpu_name or "4080" in gpu_name:
            batch_size = 128
            num_workers = 0  # 🔧 避免CUDA多进程问题
            prefetch_factor = None  # 🔧 单进程模式必须为None
            print("🎯 使用RTX 4090优化配置（单进程模式）")
        elif "3090" in gpu_name or "3080" in gpu_name:
            batch_size = 96
            num_workers = 0  # 🔧 统一使用单进程
            prefetch_factor = None
            print("🎯 使用RTX 30系优化配置（单进程模式）")
        elif "2080" in gpu_name or "2070" in gpu_name:
            batch_size = 64
            num_workers = 0  # 🔧 统一使用单进程
            prefetch_factor = None
            print("🎯 使用RTX 20系优化配置（单进程模式）")
        else:
            batch_size = 32
            num_workers = 0  # 🔧 统一使用单进程
            prefetch_factor = None
            print("🎯 使用通用GPU配置（单进程模式）")
            
        return {
            'device': device,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,  # 🔧 关键修复
            'pin_memory': True,
            'mixed_precision': False,
            'persistent_workers': False
        }
    else:
        print("⚠️ 未检测到GPU，使用CPU配置")
        return {
            'device': torch.device('cpu'),
            'batch_size': 16,
            'num_workers': 0,
            'prefetch_factor': None,  # 🔧 关键修复
            'pin_memory': False,
            'mixed_precision': False,
            'persistent_workers': False
        }


def preprocess_frame_stable(frame):
    """
    🔧 稳定版图像预处理
    提高图像质量，增强人体检测效果
    """
    try:
        # 提高对比度和亮度
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        # 锐化滤波去除运动模糊
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        return frame
    except Exception as e:
        print(f"    ⚠️ 预处理失败: {e}, 返回原图")
        return frame

def safe_float_robust(value):
    """
    🔧 终极版数值转换函数
    """
    try:
        if value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        if hasattr(value, 'dtype'):
            arr = np.asarray(value)
            
            if arr.size == 0:
                return 0.0
            
            if arr.ndim == 0:
                val = float(arr.item())
                return 0.0 if (np.isnan(val) or np.isinf(val)) else val
            
            # 对于多元素数组，取第一个有效值
            flat = arr.flatten()
            for i in range(len(flat)):
                val = float(flat[i])
                if not (np.isnan(val) or np.isinf(val)):
                    return val
            
            return 0.0
        
        val = float(value)
        return 0.0 if (np.isnan(val) or np.isinf(val)) else val
        
    except Exception:
        return 0.0

def select_nearest_person_keypoints_stable(results, frame_height=640, frame_width=640):
    """
    🔧 Linus式调试版：人体选择算法
    """
    # print(f"    🔍 【人体选择开始】")
    
    try:
        # # 检查1: results有效性
        # print(f"      检查1 - Results:")
        # print(f"        类型: {type(results)}")
        # print(f"        是否为None: {results is None}")
        # print(f"        长度: {len(results) if results else 'N/A'}")
        
        if not results or len(results) == 0:
            # print(f"      ❌ 检查1失败: results无效")
            return None, None

        # 检查2: 第一个结果
        result = results[0]
        # print(f"      检查2 - 第一个结果:")
        # print(f"        类型: {type(result)}")
        # print(f"        是否有keypoints属性: {hasattr(result, 'keypoints')}")
        
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            # print(f"      ❌ 检查2失败: 无keypoints属性或为None")
            return None, None

        # 检查3: keypoints数据
        keypoints_data = result.keypoints
        # print(f"      检查3 - Keypoints数据:")
        # print(f"        类型: {type(keypoints_data)}")
        # print(f"        是否有xy属性: {hasattr(keypoints_data, 'xy')}")
        # print(f"        是否有conf属性: {hasattr(keypoints_data, 'conf')}")
        
        if not hasattr(keypoints_data, 'xy'):
            # print(f"      ❌ 检查3失败: 无xy属性")
            return None, None
            
        # print(f"        xy长度: {len(keypoints_data.xy)}")
        
        if len(keypoints_data.xy) == 0:
            # print(f"      ❌ 检查3失败: xy为空")
            return None, None

        # 开始处理每个人
        # print(f"      🎯 开始处理 {len(keypoints_data.xy)} 个人物")
        
        best_idx = None
        max_score = 0.0
        max_possible_area = float(frame_height * frame_width)
        valid_persons = 0

        for i in range(len(keypoints_data.xy)):
            # print(f"        👤 人物{i+1}:")
            
            try:
                # 数据获取
                coords_tensor = keypoints_data.xy[i]
                confidence_tensor = keypoints_data.conf[i]
                
                # print(f"          数据类型: coords={type(coords_tensor)}, conf={type(confidence_tensor)}")
                # print(f"          数据形状: coords={coords_tensor.shape}, conf={confidence_tensor.shape}")
                
                # 转换为numpy
                coords = coords_tensor.cpu().numpy().astype(np.float32)
                confidence = confidence_tensor.cpu().numpy().astype(np.float32)
                
                # print(f"          numpy形状: coords={coords.shape}, conf={confidence.shape}")
                # print(f"          置信度统计: min={confidence.min():.3f}, max={confidence.max():.3f}, mean={confidence.mean():.3f}")
                
                if coords.size == 0 or confidence.size == 0:
                    # print(f"          ❌ 数据为空，跳过")
                    continue

                # 过滤有效关键点
                valid_mask = confidence > 0.05  # 使用较低阈值
                valid_points = coords[valid_mask]
                valid_conf = confidence[valid_mask]
                
                # print(f"          有效关键点: {len(valid_points)}/17 (阈值>0.05)")
                
                if len(valid_points) < 1:  # 至少需要1个关键点
                    # print(f"          ❌ 有效关键点不足，跳过")
                    continue

                # 计算包围盒
                x_coords = valid_points[:, 0]
                y_coords = valid_points[:, 1]
                
                x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
                y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
                
                bbox_width = max(0.0, x_max - x_min)
                bbox_height = max(0.0, y_max - y_min)
                bbox_area = max(1.0, bbox_width * bbox_height)
                
                # print(f"          包围盒: 宽{bbox_width:.1f}, 高{bbox_height:.1f}, 面积{bbox_area:.1f}")
                
                # 计算质心
                centroid_y = float(np.mean(y_coords))
                position_score = max(0.0, min(1.0, centroid_y / frame_height))
                
                # 计算平均置信度
                avg_confidence = float(np.mean(valid_conf))
                
                # print(f"          质心Y: {centroid_y:.1f}, 位置评分: {position_score:.3f}")
                # print(f"          平均置信度: {avg_confidence:.3f}")
                
                # 简化评分算法
                composite_score = avg_confidence * 0.7 + position_score * 0.3
                
                # print(f"          💯 综合评分: {composite_score:.3f}")
                
                valid_persons += 1
                
                if composite_score > max_score:
                    max_score = composite_score
                    best_idx = i
                    # print(f"          👑 新的最佳候选!")
                    
            except Exception as person_e:
                # print(f"          ❌ 处理人物{i+1}异常: {person_e}")
                import traceback
                traceback.print_exc()
                continue

        # print(f"      📊 处理完成: 有效人物{valid_persons}个, 最佳索引{best_idx}, 最高分{max_score:.3f}")

        if best_idx is None:
            # print(f"      ❌ 没有找到有效候选")
            return None, None

        # 返回结果
        try:
            best_keypoints = keypoints_data.xy[best_idx].cpu().numpy()
            best_confidence = keypoints_data.conf[best_idx].cpu().numpy()
            
            # print(f"      ✅ 成功返回: 人物{best_idx+1}")
            # print(f"        关键点shape: {best_keypoints.shape}")
            # print(f"        置信度shape: {best_confidence.shape}")
            
            return best_keypoints, best_confidence
            
        except Exception as return_e:
            # print(f"      ❌ 返回结果异常: {return_e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    except Exception as main_e:
        # print(f"    ❌ 主函数异常: {main_e}")
        import traceback
        traceback.print_exc()
        return None, None

class VideoBadmintonDataset(Dataset):
    """
    VideoBadminton数据集加载器
    集成了稳定的人体检测逻辑 + GPU优化
    """
    
    def __init__(self, dataset_dir: str, max_samples_per_class: Optional[int] = None, use_gpu: bool = True):
        """
        初始化数据集
        
        Args:
            dataset_dir: 数据集目录 (如 data/split/train/)
            max_samples_per_class: 每个类别最大样本数，用于限制数据量
            use_gpu: 是否使用GPU优化模型
        """
        self.dataset_dir = dataset_dir
        self.max_samples_per_class = max_samples_per_class
        self.samples = self._collect_samples()
        
        print(f"📁 数据集目录: {dataset_dir}")
        print(f"📊 总样本数: {len(self.samples)}")
        self._print_class_distribution()
        
        # 🔧 GPU优化：根据硬件选择更合适的YOLOv8模型
        if use_gpu and torch.cuda.is_available():
            # GPU模式使用更大的模型，检测精度更高
            self.pose_model = YOLO('yolov8m-pose.pt')
            print("🎯 GPU模式：使用YOLOv8m-pose模型")
        else:
            # CPU模式使用轻量级模型
            self.pose_model = YOLO('yolov8n-pose.pt')
            print("🎯 CPU模式：使用YOLOv8n-pose模型")
    
    def _collect_samples(self) -> List[Tuple[str, int]]:
        """收集所有视频文件和对应标签"""
        samples = []
        
        # 遍历18个动作类别文件夹
        for class_id in range(18):
            # 查找对应的类别文件夹
            class_folders = glob.glob(os.path.join(self.dataset_dir, f"{class_id:02d}_*"))
            
            if not class_folders:
                continue
            
            class_folder = class_folders[0]  # 应该只有一个匹配的文件夹
            video_files = glob.glob(os.path.join(class_folder, "*.mp4"))
            
            # 限制每个类别的样本数量（如果指定了）
            if self.max_samples_per_class and len(video_files) > self.max_samples_per_class:
                video_files = random.sample(video_files, self.max_samples_per_class)
            
            # 添加到样本列表
            for video_file in video_files:
                samples.append((video_file, class_id))
        
        # 随机打乱样本顺序
        random.shuffle(samples)
        return samples
    
    def _print_class_distribution(self):
        """打印类别分布统计"""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("\n📈 类别分布:")
        for class_id in sorted(class_counts.keys()):
            class_name = RAW_CLASSES.get(class_id, f"Class_{class_id}")
            print(f"  {class_id:02d} - {class_name}: {class_counts[class_id]} 个样本")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个训练样本
        
        Returns:
            (关键点序列张量, 动作类别标签)
        """
        video_path, label = self.samples[idx]
        
        # 从视频中提取关键点序列
        keypoints_sequence = self._extract_keypoints_from_video(video_path)
        
        if not keypoints_sequence:
            # 如果提取失败，返回零张量
            zero_tensor = torch.zeros(MODEL_CONFIG['keypoints'] * 2 * MODEL_CONFIG['sequence_length'])
            return zero_tensor, label
        
        # 转换为固定长度的张量
        sequence_tensor = self._keypoints_to_tensor(keypoints_sequence)
        return sequence_tensor, label
    
    def _extract_keypoints_from_video(self, video_path: str) -> List[Keypoints]:
        """
        🔧 Linus式调试版：从视频提取关键点序列
        每个步骤都有详细日志，定位问题
        """
        video_name = os.path.basename(video_path)
        # print(f"\n🎬 开始处理视频: {video_name}")
        # print(f"📍 完整路径: {video_path}")
        
        # 步骤1: 视频加载
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        if not cap.isOpened():
            print(f"❌ 视频打开失败: {video_path}")
            return []
        
        # 获取视频基本信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # print(f"📊 视频信息: {width}x{height}, {total_frames}帧, {fps:.1f}fps")   
        
        frame_count = 0
        max_frames = TRAINING_CONFIG['max_frames_per_video']
        successful_detections = 0
        
        # print(f"🎯 将处理最多 {max_frames} 帧")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                # print(f"📹 第{frame_count+1}帧读取失败，视频结束")
                break
            
            # print(f"\n🔍 处理第{frame_count+1}帧:")
            # print(f"  原始帧尺寸: {frame.shape}")
            
            try:
                # 步骤2: 图像裁剪
                height, width = frame.shape[:2]
                crop_y1 = max(0, int(height * 0.02))
                crop_y2 = min(height, int(height * 0.98))
                crop_x1 = max(0, int(width * 0.02))
                crop_x2 = min(width, int(width * 0.98))
                
                # print(f"  裁剪区域: y[{crop_y1}:{crop_y2}], x[{crop_x1}:{crop_x2}]")
                
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                # print(f"  裁剪后尺寸: {cropped_frame.shape}")
                
                # 步骤3: 图像预处理
                try:
                    processed_frame = preprocess_frame_stable(cropped_frame)
                    # print(f"  预处理完成: {processed_frame.shape}")
                except Exception as preprocess_e:
                    # print(f"  ⚠️ 预处理失败: {preprocess_e}")
                    processed_frame = cropped_frame
                
                # 步骤4: 尺寸调整
                target_size = 640
                frame_resized = cv2.resize(processed_frame, (target_size, target_size))
                # print(f"  调整到目标尺寸: {frame_resized.shape}")
                
                # 步骤5: YOLO检测 - 关键环节
                # print(f"  🤖 开始YOLO检测...")
                # print(f"  YOLO模型类型: {type(self.pose_model)}")
                
                try:
                    results = self.pose_model(frame_resized, verbose=False, conf=0.03)
                    # print(f"  ✅ YOLO检测完成")
                    # print(f"  Results类型: {type(results)}")
                    # print(f"  Results长度: {len(results) if results else 'None'}")
                    
                    if results and len(results) > 0:
                        result = results[0]
                        # print(f"  第一个result类型: {type(result)}")
                        # print(f"  result属性: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                        # print(f"  是否有keypoints: {hasattr(result, 'keypoints')}")
                        
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            kp_data = result.keypoints
                            # print(f"  keypoints类型: {type(kp_data)}")
                            # print(f"  keypoints属性: {[attr for attr in dir(kp_data) if not attr.startswith('_')]}")
                            # print(f"  是否有xy: {hasattr(kp_data, 'xy')}")
                            # print(f"  是否有conf: {hasattr(kp_data, 'conf')}")
                            
                            # if hasattr(kp_data, 'xy'):
                            #     # print(f"  xy类型: {type(kp_data.xy)}")
                            #     # print(f"  检测到人数: {len(kp_data.xy)}")
                                
                            #     if len(kp_data.xy) > 0:
                            #         # print(f"  第一个人关键点:")
                            #         # print(f"    xy shape: {kp_data.xy[0].shape}")
                            #         # print(f"    xy类型: {type(kp_data.xy)}")
                            #         # if hasattr(kp_data, 'conf'):
                            #             # print(f"    conf shape: {kp_data.conf.shape}")
                            #             # print(f"    conf类型: {type(kp_data.conf)}")
                            #             # print(f"    置信度范围: {kp_data.conf.min():.3f}-{kp_data.conf.max():.3f}")
                            #     else:
                            # #         print(f"  ❌ xy数组为空")
                        #     # else:
                        #     #     print(f"  ❌ keypoints没有xy属性")
                        # else:
                        #     print(f"  ❌ result没有keypoints或keypoints为None")
                    else:
                        print(f"  ❌ YOLO返回空结果")
                    
                except Exception as yolo_e:
                    print(f"  ❌ YOLO检测异常: {yolo_e}")
                    import traceback
                    traceback.print_exc()
                    frame_count += 1
                    continue
                
                # 步骤6: 人体选择
                # print(f"  👤 开始人体选择...")
                try:
                    best_keypoints, best_confidence = select_nearest_person_keypoints_stable(
                        results, target_size, target_size
                    )
                    
                    if best_keypoints is not None and best_confidence is not None:
                        # print(f"  ✅ 人体选择成功")
                        # print(f"    关键点shape: {best_keypoints.shape}")
                        # print(f"    置信度shape: {best_confidence.shape}")
                        
                        # 步骤7: 质量检查
                        try:
                            avg_confidence = safe_float_robust(best_confidence.mean())
                            # print(f"    平均置信度: {avg_confidence:.3f}")
                            
                            if avg_confidence > 0.02:
                                # print(f"    ✅ 质量检查通过，添加到序列")
                                keypoints_list.append(Keypoints(
                                    points=best_keypoints,
                                    confidence=best_confidence
                                ))
                                successful_detections += 1
                            # else:
                                # print(f"    ❌ 质量检查失败，平均置信度{avg_confidence:.3f} <= 0.02")
                        except Exception as quality_e:
                            print(f"    ❌ 质量检查异常: {quality_e}")
                    # else:
                        # print(f"  ❌ 人体选择失败，返回None")
                        
                except Exception as select_e:
                    # print(f"  ❌ 人体选择异常: {select_e}")
                    import traceback
                    traceback.print_exc()
            
            except Exception as frame_e:
                print(f"  ❌ 处理第{frame_count+1}帧异常: {frame_e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
            
            # 早停检查
            if frame_count > 30 and successful_detections == 0:
                print(f"🛑 早停：处理了{frame_count}帧，成功0次")
                break
        
        cap.release()
        
        # 最终统计
        success_rate = successful_detections / max(frame_count, 1) * 100
        # print(f"\n📊 视频处理完成:")
        # print(f"  总处理帧数: {frame_count}")
        # print(f"  成功检测帧数: {successful_detections}")
        # print(f"  检测成功率: {success_rate:.1f}%")
        # print(f"  关键点序列长度: {len(keypoints_list)}")
        
        if success_rate < 5:
            print(f"⚠️  {video_name}: 检测成功率 {success_rate:.1f}% ({successful_detections}/{frame_count})")
        
        return keypoints_list
    
    def _keypoints_to_tensor(self, keypoints_sequence: List[Keypoints]) -> torch.Tensor:
        """
        将关键点序列转换为固定长度的张量
        
        Args:
            keypoints_sequence: 关键点序列
            
        Returns:
            固定长度的张量 (340维: 17关键点 × 2坐标 × 10帧)
        """
        target_length = MODEL_CONFIG['sequence_length']
        
        if len(keypoints_sequence) >= target_length:
            # 如果序列过长，均匀采样选择代表性帧
            indices = np.linspace(0, len(keypoints_sequence) - 1, target_length, dtype=int)
            selected_keypoints = [keypoints_sequence[i] for i in indices]
        else:
            # 如果序列过短，采用重复填充策略
            selected_keypoints = keypoints_sequence.copy()
            while len(selected_keypoints) < target_length:
                if keypoints_sequence:
                    # 重复最后一帧
                    selected_keypoints.append(keypoints_sequence[-1])
                else:
                    # 如果没有有效关键点，用零填充
                    zero_keypoints = Keypoints(
                        points=np.zeros((MODEL_CONFIG['keypoints'], 2)),
                        confidence=np.zeros(MODEL_CONFIG['keypoints'])
                    )
                    selected_keypoints.append(zero_keypoints)
        
        # 🔄 转换为张量：展平所有关键点坐标
        sequence_data = []
        for keypoints in selected_keypoints:
            sequence_data.extend(keypoints.points.flatten())
        
        return torch.FloatTensor(sequence_data)

class Trainer:
    """
    羽毛球动作分类模型训练器 - GPU优化+稳定性增强版本
    """
    
    def __init__(self, data_root: str = "data/split/", force_cpu: bool = False):
        """
        初始化训练器
        
        Args:
            data_root: 分割后的数据集根目录
            force_cpu: 强制使用CPU模式
        """
        self.data_root = data_root
        
        # 🔧 GPU优化配置
        if force_cpu:
            self.config = {
                'device': torch.device('cpu'),
                'batch_size': 16,
                'num_workers': 2,
                'prefetch_factor': 2,
                'pin_memory': False,
                'mixed_precision': False,
                'persistent_workers': False
            }
            print("🔧 强制使用CPU模式")
        else:
            self.config = get_optimal_config()
        
        self.device = self.config['device']
        
        print("🚀 初始化羽毛球动作分类训练器 (稳定增强版)")
        print(f"🔧 使用设备: {self.device}")
        print(f"📁 数据根目录: {data_root}")
        print(f"🎯 批次大小: {self.config['batch_size']}")
        print(f"🎯 工作线程: {self.config['num_workers']}")
        print(f"🎯 混合精度: {self.config['mixed_precision']}")
        print("🎯 集成了稳定的人体选择算法")
        
        # 🔧 暂时禁用混合精度训练确保稳定性
        if self.config['mixed_precision']:
            try:
                self.scaler = amp.GradScaler()
                print("⚡ 启用混合精度训练加速")
            except:
                self.config['mixed_precision'] = False
                print("⚠️ 混合精度初始化失败，禁用混合精度")
        
        # 验证数据目录
        self._validate_data_directories()
    
    def _validate_data_directories(self):
        """验证数据目录结构"""
        required_dirs = ['train', 'val', 'test']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.data_root, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"❌ 找不到必需的目录: {dir_path}")
        print("✅ 数据目录结构验证通过")
    
    def train(self, epochs: int = TRAINING_CONFIG['max_epochs'], 
              save_path: str = "badminton_model_stable.pth"):
        """
        训练模型 - GPU优化+稳定性增强版本
        
        Args:
            epochs: 训练轮数
            save_path: 模型保存路径
        """
        print("=" * 60)
        print("🏸 开始训练羽毛球动作分类模型 (稳定增强版)")
        print("=" * 60)
        
        training_start_time = time.time()
        
        # 加载数据集
        print("📂 正在加载数据集...")
        use_gpu_models = self.device.type == 'cuda'
        
        train_dataset = VideoBadmintonDataset(f"{self.data_root}/train/", use_gpu=use_gpu_models)
        val_dataset = VideoBadmintonDataset(f"{self.data_root}/val/", use_gpu=use_gpu_models)
        test_dataset = VideoBadmintonDataset(f"{self.data_root}/test/", use_gpu=use_gpu_models)
        
        print(f"✅ 数据集加载完成")
        print(f"   训练集: {len(train_dataset)} 个样本")
        print(f"   验证集: {len(val_dataset)} 个样本")
        print(f"   测试集: {len(test_dataset)} 个样本")
        
        # 🔧 GPU优化的数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            persistent_workers=self.config['persistent_workers'],
            prefetch_factor=self.config['prefetch_factor'],
            drop_last=True  # 确保批次大小一致，有利于BatchNorm
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            persistent_workers=self.config['persistent_workers'],
            prefetch_factor=self.config['prefetch_factor']
        )
        
        # 初始化模型
        detector = BadmintonDetector()
        model = detector.action_classifier.to(self.device)
        
        # 🔧 GPU优化：模型编译（PyTorch 2.0+）
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='default')  # 使用default模式确保稳定性
                print("⚡ 模型编译优化成功")
            except Exception as e:
                print(f"⚠️ 模型编译失败，使用标准模式: {e}")
        
        # 训练配置
        criterion = torch.nn.CrossEntropyLoss()
        
        # 🔧 GPU优化：使用AdamW优化器和学习率scaling
        base_lr = MODEL_CONFIG['learning_rate']
        scaled_lr = base_lr * (self.config['batch_size'] / 16)  # 根据批次大小调整学习率
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=scaled_lr,
            weight_decay=MODEL_CONFIG['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=TRAINING_CONFIG['lr_scheduler_patience'],
            factor=0.5
        )
        
        # 训练状态跟踪
        best_val_accuracy = 0.0
        epochs_without_improvement = 0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        print(f"🔧 稳定增强训练配置:")
        print(f"   批次大小: {self.config['batch_size']}")
        print(f"   基础学习率: {base_lr:.6f}")
        print(f"   缩放学习率: {scaled_lr:.6f}")
        print(f"   最大轮数: {epochs}")
        print(f"   混合精度: {self.config['mixed_precision']}")
        print("-" * 40)
        
        # 主训练循环
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            # 训练阶段
            if self.config['mixed_precision']:
                train_loss, train_acc = self._train_epoch_amp(model, train_loader, criterion, optimizer)
            else:
                train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证阶段
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # 更新学习率调度器
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            training_history['learning_rate'].append(current_lr)
            
            # 显示当前轮结果
            epoch_time = time.time() - epoch_start_time
            
            # 🔧 GPU内存监控
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.max_memory_allocated() / 1024**3
                gpu_memory_cached = torch.cuda.max_memory_reserved() / 1024**3
                gpu_info = f"| GPU内存: {gpu_memory_used:.1f}GB/{gpu_memory_cached:.1f}GB"
            else:
                gpu_info = ""
            
            print(f"📊 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"📊 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"⏱️  轮次耗时: {epoch_time:.1f}秒 {gpu_info} | 学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                epochs_without_improvement = 0
                
                torch.save(model.state_dict(), save_path)
                print(f"🎯 新的最佳模型！验证准确率: {val_acc:.2f}% (已保存)")
            else:
                epochs_without_improvement += 1
            
            # 早停检查
            if epochs_without_improvement >= TRAINING_CONFIG['early_stopping_patience']:
                print(f"⏹️  早停触发！{epochs_without_improvement} 轮无改进")
                break
            
            # 🔧 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 训练结束统计
        total_training_time = time.time() - training_start_time
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print(f"⏱️  总训练时间: {total_training_time:.1f}秒")
        print(f"🏆 最佳验证准确率: {best_val_accuracy:.2f}%")
        print(f"📁 模型已保存至: {save_path}")
        
        # 保存训练历史
        history_path = save_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"📊 训练历史已保存至: {history_path}")
        
        # 最终测试评估
        if os.path.exists(save_path):
            self._final_evaluation(test_dataset, save_path)
    
    def _train_epoch_amp(self, model, train_loader, criterion, optimizer) -> Tuple[float, float]:
        """训练一个epoch - 混合精度版本"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            # 🔧 使用传统的混合精度API确保兼容性
            try:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            except Exception as amp_e:
                # 如果混合精度失败，回退到标准模式
                print(f"⚠️ 混合精度失败，回退到标准模式: {amp_e}")
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 显示进度
            if batch_idx % 50 == 0:  # GPU训练更快，减少输出频率
                current_acc = 100.0 * correct / total
                print(f"  📦 批次 {batch_idx}/{len(train_loader)} | "
                      f"损失: {loss.item():.4f} | 准确率: {current_acc:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _train_epoch(self, model, train_loader, criterion, optimizer) -> Tuple[float, float]:
        """训练一个epoch - 标准版本"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 🔧 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 显示进度
            if batch_idx % 50 == 0:
                current_acc = 100.0 * correct / total
                print(f"  📦 批次 {batch_idx}/{len(train_loader)} | "
                      f"损失: {loss.item():.4f} | 准确率: {current_acc:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, model, val_loader, criterion) -> Tuple[float, float]:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 🔧 稳定的验证推理
                try:
                    if self.config['mixed_precision']:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                except Exception as val_e:
                    # 验证失败时使用标准模式
                    output = model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _final_evaluation(self, test_dataset, model_path):
        """最终测试集评估"""
        print("\n" + "=" * 60)
        print("🧪 最终评估 (使用测试集)")
        print("=" * 60)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        # 加载最佳模型
        detector = BadmintonDetector()
        model = detector.action_classifier.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # 评估
        correct = 0
        total = 0
        class_correct = [0] * 18
        class_total = [0] * 18
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 稳定的测试推理
                try:
                    if self.config['mixed_precision']:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                    else:
                        output = model(data)
                except:
                    output = model(data)
                
                _, predicted = torch.max(output, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 按类别统计
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_total[label] += 1
                    if predicted[i] == target[i]:
                        class_correct[label] += 1
        
        # 输出整体结果
        overall_accuracy = 100.0 * correct / total
        print(f"🎯 测试集整体准确率: {overall_accuracy:.2f}% ({correct}/{total})")
        
        # 输出各类别详细结果
        print("\n📊 各类别详细结果:")
        print("-" * 80)
        print(f"{'类别ID':<6} {'动作名称':<20} {'准确率':<10} {'样本数':<10}")
        print("-" * 80)
        
        for i in range(18):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                class_name = RAW_CLASSES.get(i, f"Class_{i}")
                print(f"{i:02d}     {class_name:<20} {acc:>6.2f}%     {class_total[i]:>4d}")
        
        print("-" * 80)
        print("✅ 最终评估完成")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练羽毛球动作分类模型 (稳定增强版)")
    parser.add_argument("--data", default="data/split/", 
                       help="分割后的数据集根目录")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG['max_epochs'], 
                       help="训练轮数")
    parser.add_argument("--output", default="badminton_model_stable.pth", 
                       help="模型输出路径")
    parser.add_argument("--batch-size", type=int, default=None, 
                       help="批次大小（留空自动优化）")
    parser.add_argument("--lr", type=float, default=MODEL_CONFIG['learning_rate'], 
                       help="学习率")
    parser.add_argument("--cpu", action="store_true", 
                       help="强制使用CPU模式")
    
    args = parser.parse_args()
    
    # 更新配置
    if args.batch_size:
        MODEL_CONFIG['batch_size'] = args.batch_size
    MODEL_CONFIG['learning_rate'] = args.lr
    
    # 设置随机种子确保可重现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 🔧 GPU优化设置
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = False  # 允许非确定性以提高性能
        torch.backends.cudnn.benchmark = True       # 优化卷积性能
    
    # 开始训练
    trainer = Trainer(args.data, force_cpu=args.cpu)
    trainer.train(epochs=args.epochs, save_path=args.output)

if __name__ == "__main__":
    main()
