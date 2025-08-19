"""
配置文件 - 所有常量和配置参数
Linus原则：配置集中管理，避免魔法数字散布在代码中
"""

# 原始数据集的18个分类 - 保持与VideoBadminton数据集一致
RAW_CLASSES = {
    0: "Short Serve",           # 短发球
    1: "Cross Court Flight",    # 斜线球飞行
    2: "Lift",                  # 挑球
    3: "Tap Smash",            # 轻扣杀
    4: "Block",                # 拦网
    5: "Drop Shot",            # 吊球
    6: "Push Shot",            # 推球
    7: "Transitional Slice",   # 过渡性削球
    8: "Cut",                  # 切削球
    9: "Rush Shot",            # 突击球
    10: "Defensive Clear",     # 防守高远球
    11: "Defensive Drive",     # 防守抽球
    12: "Clear",               # 高远球
    13: "Long Serve",          # 长发球
    14: "Smash",               # 扣杀
    15: "Flat Shot",           # 平抽球
    16: "Rear Court Flat Drive", # 后场平抽球
    17: "Short Flat Shot"      # 短平球
}

# 用户友好的4大类映射 - 简化复杂的18类为用户可理解的分类
CATEGORY_MAPPING = {
    # 发球类：短发球、长发球
    0: 0, 13: 0,
    # 进攻类：轻扣杀、吊球、突击球、扣杀
    3: 1, 5: 1, 9: 1, 14: 1,
    # 防守类：挑球、防守高远球、防守抽球、高远球
    2: 2, 10: 2, 11: 2, 12: 2,
    # 控制类：其余所有技术
    1: 3, 4: 3, 6: 3, 7: 3, 8: 3, 15: 3, 16: 3, 17: 3
}

# 四大类别名称
CATEGORIES = ["发球", "进攻", "防守", "控制"]

# 模型配置参数
MODEL_CONFIG = {
    'input_size': (640, 640),     # YOLOv8标准输入尺寸
    'keypoints': 17,              # COCO格式人体关键点数量
    'sequence_length': 10,        # 动作序列帧数
    'batch_size': 16,             # 批次大小（根据显存调整）
    'learning_rate': 0.001,       # 学习率
    'weight_decay': 1e-5,         # 权重衰减
    'dropout_rate': 0.3           # Dropout比例
}

# 质量评估规则 - 基于人体工学的硬编码规则
# 这比黑盒神经网络更可解释、更可控
QUALITY_RULES = {
    0: {  # 发球类
        'elbow_angle': (90, 150),      # 肘关节角度范围
        'knee_bend': (10, 30),         # 膝关节弯曲角度
        'racket_height': (0.8, 1.2)    # 球拍高度比例
    },
    1: {  # 进攻类
        'elbow_angle': (120, 160),     # 进攻时肘关节更大角度
        'shoulder_rotation': (30, 60), # 肩部旋转角度
        'weight_transfer': True        # 重心转移
    },
    2: {  # 防守类
        'stance_width': (0.6, 1.0),    # 站位宽度
        'racket_angle': (45, 90),      # 球拍角度
        'ready_position': True         # 准备姿势
    },
    3: {  # 控制类
        'wrist_stability': True,       # 手腕稳定性
        'gentle_swing': True,          # 温和挥拍
        'precision_grip': True         # 精准握拍
    }
}

# 数据集分割比例
DATA_SPLIT_RATIOS = {
    'train': 0.7,      # 训练集70%
    'val': 0.15,       # 验证集15%
    'test': 0.15       # 测试集15%
}

# 训练相关配置
TRAINING_CONFIG = {
    'max_epochs': 100,             # 最大训练轮数
    'early_stopping_patience': 10, # 早停耐心值
    'lr_scheduler_patience': 5,     # 学习率调度器耐心值
    'min_confidence_threshold': 0.3, # 关键点置信度阈值
    'max_frames_per_video': 50,    # 每个视频最大处理帧数
    'save_best_only': True         # 只保存最佳模型
}
