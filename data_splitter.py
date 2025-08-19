"""
数据集分割工具
Linus原则：测试集是神圣的，训练时绝对不能碰
"""

import os
import glob
import shutil
import random
from typing import List
from config import DATA_SPLIT_RATIOS

def split_videobadminton_dataset(
    source_dir: str = "data/VideoBadminton_Dataset/",
    output_dir: str = "data/split/",
    train_ratio: float = DATA_SPLIT_RATIOS['train'],
    val_ratio: float = DATA_SPLIT_RATIOS['val'], 
    test_ratio: float = DATA_SPLIT_RATIOS['test'],
    random_seed: int = 42
):
    """
    分割VideoBadminton数据集为训练集、验证集、测试集
    
    Args:
        source_dir: 原始数据集目录
        output_dir: 输出目录
        train_ratio: 训练集比例 (默认70%)
        val_ratio: 验证集比例 (默认15%)
        test_ratio: 测试集比例 (默认15%)
        random_seed: 随机种子，确保结果可重现
    """
    
    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"比例总和必须等于1.0，当前为{total_ratio}"
    
    # 设置随机种子确保可重现
    random.seed(random_seed)
    
    print("开始分割VideoBadminton数据集...")
    print(f"训练集: {train_ratio*100:.1f}%")
    print(f"验证集: {val_ratio*100:.1f}%")
    print(f"测试集: {test_ratio*100:.1f}%")
    print("-" * 50)
    
    # 创建输出目录结构
    _create_output_directories(source_dir, output_dir)
    
    total_files = 0
    split_summary = {'train': 0, 'val': 0, 'test': 0}
    
    # 对每个动作类别进行分割
    for class_id in range(18):
        class_folder = _find_class_folder(source_dir, class_id)
        if not class_folder:
            print(f"⚠️  警告: 找不到类别 {class_id:02d} 的文件夹")
            continue
        
        # 获取该类别的所有视频文件
        video_files = glob.glob(os.path.join(class_folder, "*.mp4"))
        if not video_files:
            print(f"⚠️  警告: 类别 {class_id:02d} 中没有视频文件")
            continue
        
        random.shuffle(video_files)  # 随机打乱文件顺序
        
        # 计算各个数据集的文件数量
        total_count = len(video_files)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        test_count = total_count - train_count - val_count  # 剩余的分给测试集
        
        # 分割文件列表
        train_files = video_files[:train_count]
        val_files = video_files[train_count:train_count + val_count]
        test_files = video_files[train_count + val_count:]
        
        # 复制文件到对应目录
        class_name = os.path.basename(class_folder)
        _copy_files(train_files, f"{output_dir}/train/{class_name}")
        _copy_files(val_files, f"{output_dir}/val/{class_name}")
        _copy_files(test_files, f"{output_dir}/test/{class_name}")
        
        # 更新统计信息
        total_files += total_count
        split_summary['train'] += len(train_files)
        split_summary['val'] += len(val_files)
        split_summary['test'] += len(test_files)
        
        print(f"类别 {class_id:02d} ({class_name}): "
              f"训练{len(train_files)} | 验证{len(val_files)} | 测试{len(test_files)}")
    
    # 输出最终统计
    print("-" * 50)
    print("✅ 数据集分割完成！")
    print(f"总文件数: {total_files}")
    print(f"训练集: {split_summary['train']} 个文件")
    print(f"验证集: {split_summary['val']} 个文件")
    print(f"测试集: {split_summary['test']} 个文件")
    print(f"输出目录: {output_dir}")

def _create_output_directories(source_dir: str, output_dir: str):
    """创建输出目录结构"""
    for split in ['train', 'val', 'test']:
        for class_id in range(18):
            class_name = _get_class_name(source_dir, class_id)
            target_dir = f"{output_dir}/{split}/{class_id:02d}_{class_name}"
            os.makedirs(target_dir, exist_ok=True)

def _find_class_folder(source_dir: str, class_id: int) -> str:
    """找到对应类别的文件夹"""
    pattern = os.path.join(source_dir, f"{class_id:02d}_*")
    folders = glob.glob(pattern)
    return folders[0] if folders else ""

def _get_class_name(source_dir: str, class_id: int) -> str:
    """获取类别名称"""
    folder = _find_class_folder(source_dir, class_id)
    if folder:
        basename = os.path.basename(folder)
        # 去掉前面的数字前缀 "00_Short Serve" -> "Short Serve"
        if '_' in basename:
            return basename.split('_', 1)[1]
    return f"class_{class_id}"

def _copy_files(file_list: List[str], target_dir: str):
    """复制文件列表到目标目录"""
    if not file_list:
        return
    
    os.makedirs(target_dir, exist_ok=True)
    for file_path in file_list:
        filename = os.path.basename(file_path)
        target_path = os.path.join(target_dir, filename)
        try:
            shutil.copy2(file_path, target_path)
        except Exception as e:
            print(f"⚠️  复制文件失败 {file_path}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分割VideoBadminton数据集")
    parser.add_argument("--source", default="data/VideoBadminton_Dataset/", 
                       help="原始数据集目录")
    parser.add_argument("--output", default="data/split/", 
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子")
    
    args = parser.parse_args()
    
    split_videobadminton_dataset(
        source_dir=args.source,
        output_dir=args.output,
        random_seed=args.seed
    )
