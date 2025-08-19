"""
模型训练器 - 训练羽毛球动作分类模型
Linus原则：工具要简单、可靠、可预测
集成了成功的人体选择算法
"""

import os
import glob
import cv2
import torch
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

def preprocess_frame(frame):
    """
    提高图像质量，增强人体检测效果
    """
    # 提高对比度和亮度
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    # 锐化滤波去除运动模糊
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame

def safe_float(value):
    """
    安全的数值转换，处理所有numpy类型
    集成自test_person.py的成功实现
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
    集成自test_person.py的成功实现，专门针对训练数据优化
    
    判断标准：
    1. 关键点包围盒面积最大（人体在画面中最大）
    2. 关键点质心位置最靠近画面底部（更靠近摄像头）
    3. 综合评分选择最佳候选
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

class VideoBadmintonDataset(Dataset):
    """
    VideoBadminton数据集加载器
    集成了成功的人体检测逻辑
    """
    
    def __init__(self, dataset_dir: str, max_samples_per_class: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            dataset_dir: 数据集目录 (如 data/split/train/)
            max_samples_per_class: 每个类别最大样本数，用于限制数据量
        """
        self.dataset_dir = dataset_dir
        self.max_samples_per_class = max_samples_per_class
        self.samples = self._collect_samples()
        
        print(f"📁 数据集目录: {dataset_dir}")
        print(f"📊 总样本数: {len(self.samples)}")
        self._print_class_distribution()
        
        # 初始化姿态检测器（只用于提取关键点）
        self.pose_model = YOLO('yolov8n-pose.pt')
    
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
        从视频提取关键点序列 - 集成test_person.py的成功实现
        
        优化点：
        1. 智能裁剪去除过多背景
        2. 图像预处理提高检测率
        3. 优先选择靠近摄像头的人（核心改进）
        4. 安全的错误处理
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        if not cap.isOpened():
            return []
        
        frame_count = 0
        max_frames = TRAINING_CONFIG['max_frames_per_video']
        successful_detections = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 📐 智能裁剪：去除过多背景，突出人物区域
                height, width = frame.shape[:2]
                
                # 裁剪参数：保留中央区域，去掉上下左右的背景
                crop_y1, crop_y2 = int(height * 0.15), int(height * 0.90)
                crop_x1, crop_x2 = int(width * 0.10), int(width * 0.90)
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # 🔧 图像预处理：提高对比度和清晰度
                processed_frame = preprocess_frame(cropped_frame)
                
                # 📏 调整到模型输入尺寸
                target_size = 640
                frame_resized = cv2.resize(processed_frame, (target_size, target_size))
                
                # 🤖 YOLOv8姿态检测：降低检测阈值提高召回率
                results = self.pose_model(frame_resized, verbose=False, conf=0.1)
                
                # 🎯 核心改进：使用test_person.py中成功的人体选择算法
                best_keypoints, best_confidence = select_nearest_person_keypoints(
                    results, target_size, target_size
                )
                
                if best_keypoints is not None and best_confidence is not None:
                    # 检查整体质量：平均置信度必须达到最低要求
                    avg_confidence = safe_float(best_confidence.mean())
                    if avg_confidence > 0.05:  # 降低阈值，提高检测成功率
                        keypoints_list.append(Keypoints(
                            points=best_keypoints,
                            confidence=best_confidence
                        ))
                        successful_detections += 1
                
            except Exception as e:
                # 跳过处理失败的帧，继续处理下一帧
                pass
            
            frame_count += 1
            
            # 🎯 优化：如果检测成功率太低，提前终止
            if frame_count > 20 and successful_detections == 0:
                break
        
        cap.release()
        
        # 📊 检测统计（只对失败率高的视频输出警告）
        success_rate = successful_detections / max(frame_count, 1) * 100
        if success_rate < 10:  # 成功率低于10%时输出警告
            video_name = os.path.basename(video_path)
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
    羽毛球动作分类模型训练器
    """
    
    def __init__(self, data_root: str = "data/split/"):
        """
        初始化训练器
        
        Args:
            data_root: 分割后的数据集根目录
        """
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🚀 初始化羽毛球动作分类训练器")
        print(f"🔧 使用设备: {self.device}")
        print(f"📁 数据根目录: {data_root}")
        print("🎯 集成了优化的人体选择算法")
        
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
              save_path: str = "badminton_model.pth"):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            save_path: 模型保存路径
        """
        print("=" * 60)
        print("🏸 开始训练羽毛球动作分类模型")
        print("=" * 60)
        
        training_start_time = time.time()
        
        # 加载数据集
        print("📂 正在加载数据集...")
        train_dataset = VideoBadmintonDataset(f"{self.data_root}/train/")
        val_dataset = VideoBadmintonDataset(f"{self.data_root}/val/")
        test_dataset = VideoBadmintonDataset(f"{self.data_root}/test/")
        
        print(f"✅ 数据集加载完成")
        print(f"   训练集: {len(train_dataset)} 个样本")
        print(f"   验证集: {len(val_dataset)} 个样本")
        print(f"   测试集: {len(test_dataset)} 个样本")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # 初始化模型
        detector = BadmintonDetector()
        model = detector.action_classifier.to(self.device)
        
        # 训练配置
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=MODEL_CONFIG['learning_rate'],
            weight_decay=MODEL_CONFIG['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=TRAINING_CONFIG['lr_scheduler_patience'],
            factor=0.5
        )
        
        # 训练状态跟踪
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        print(f"🔧 训练配置:")
        print(f"   批次大小: {MODEL_CONFIG['batch_size']}")
        print(f"   学习率: {MODEL_CONFIG['learning_rate']}")
        print(f"   权重衰减: {MODEL_CONFIG['weight_decay']}")
        print(f"   最大轮数: {epochs}")
        print("-" * 40)
        
        # 主训练循环
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            # 训练阶段
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
            print(f"📊 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"📊 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"⏱️  轮次耗时: {epoch_time:.1f}秒 | 学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                if TRAINING_CONFIG['save_best_only']:
                    torch.save(model.state_dict(), save_path)
                    print(f"🎯 新的最佳模型！验证准确率: {val_acc:.2f}% (已保存)")
            else:
                epochs_without_improvement += 1
            
            # 早停检查
            if epochs_without_improvement >= TRAINING_CONFIG['early_stopping_patience']:
                print(f"⏹️  早停触发！{epochs_without_improvement} 轮无改进")
                break
        
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
    
    def _train_epoch(self, model, train_loader, criterion, optimizer) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 显示进度
            if batch_idx % 20 == 0:  # 减少输出频率
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
                data, target = data.to(self.device), target.to(self.device)
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
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False
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
        class_total =  [0] * 18
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 记录预测和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
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
    
    parser = argparse.ArgumentParser(description="训练羽毛球动作分类模型")
    parser.add_argument("--data", default="data/split/", 
                       help="分割后的数据集根目录")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG['max_epochs'], 
                       help="训练轮数")
    parser.add_argument("--output", default="badminton_model.pth", 
                       help="模型输出路径")
    parser.add_argument("--batch-size", type=int, default=MODEL_CONFIG['batch_size'], 
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=MODEL_CONFIG['learning_rate'], 
                       help="学习率")
    
    args = parser.parse_args()
    
    # 更新配置
    MODEL_CONFIG['batch_size'] = args.batch_size
    MODEL_CONFIG['learning_rate'] = args.lr
    
    # 设置随机种子确保可重现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 开始训练
    trainer = Trainer(args.data)
    trainer.train(epochs=args.epochs, save_path=args.output)

if __name__ == "__main__":
    main()
