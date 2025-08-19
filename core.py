"""
核心数据结构和基础功能
Linus原则：好的代码围绕好的数据结构构建
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from config import CATEGORY_MAPPING, CATEGORIES, QUALITY_RULES

@dataclass
class Keypoints:
    """
    人体关键点数据结构
    使用COCO格式的17个关键点
    """
    points: np.ndarray      # shape: (17, 2) - x,y坐标
    confidence: np.ndarray  # shape: (17,) - 每个点的置信度
    
    def __post_init__(self):
        """数据验证 - 确保输入格式正确"""
        assert self.points.shape == (17, 2), f"关键点应为(17,2)形状，实际为{self.points.shape}"
        assert self.confidence.shape == (17,), f"置信度应为(17,)形状，实际为{self.confidence.shape}"
        
        # 确保坐标值在合理范围内
        assert np.all(self.points >= 0), "关键点坐标不能为负数"
        assert np.all(self.confidence >= 0) and np.all(self.confidence <= 1), "置信度应在[0,1]范围内"

@dataclass
class BadmintonShot:
    """
    羽毛球击球分析结果 - 整个系统的核心输出
    包含原始分类、简化分类、质量评分和改进建议
    """
    raw_class: int                      # 0-17，原始18类分类
    category: int                       # 0-3，简化的4类分类
    category_name: str                  # 分类中文名称
    quality: float                      # 0.0-1.0，动作质量评分
    feedback: str                       # 具体改进建议
    keypoints_sequence: List[Keypoints] # 用于分析的关键点序列
    confidence: float                   # 分类置信度
    
    @classmethod
    def from_raw_class(cls, raw_class: int, keypoints_seq: List[Keypoints], 
                      classification_confidence: float = 1.0) -> 'BadmintonShot':
        """
        从原始分类创建BadmintonShot对象
        这是创建分析结果的标准方法
        """
        # 映射到简化分类
        category = CATEGORY_MAPPING.get(raw_class, 3)  # 默认为控制类
        category_name = CATEGORIES[category]
        
        # 计算动作质量评分
        quality = cls._calculate_quality(category, keypoints_seq)
        
        # 生成个性化反馈建议
        feedback = cls._generate_feedback(category, quality, keypoints_seq, raw_class)
        
        return cls(
            raw_class=raw_class,
            category=category,
            category_name=category_name,
            quality=quality,
            feedback=feedback,
            keypoints_sequence=keypoints_seq,
            confidence=classification_confidence
        )
    
    @staticmethod
    def _calculate_quality(category: int, keypoints_seq: List[Keypoints]) -> float:
        """
        基于人体工学规则计算动作质量评分
        Linus原则：可预测的规则比黑盒AI更可靠
        """
        if not keypoints_seq:
            return 0.0
        
        # 使用序列中间帧作为代表帧进行分析
        mid_frame_idx = len(keypoints_seq) // 2
        representative_frame = keypoints_seq[mid_frame_idx]
        points = representative_frame.points
        confidence = representative_frame.confidence
        
        # 如果关键点置信度太低，返回较低评分
        if confidence.mean() < 0.5:
            return 0.3
        
        rules = QUALITY_RULES.get(category, {})
        scores = []
        
        try:
            if category == 0:  # 发球类评估
                elbow_angle = _calculate_elbow_angle(points)
                knee_bend = _calculate_knee_bend(points)
                scores.extend([
                    _score_range(elbow_angle, rules.get('elbow_angle', (0, 180))),
                    _score_range(knee_bend, rules.get('knee_bend', (0, 90)))
                ])
                
            elif category == 1:  # 进攻类评估
                elbow_angle = _calculate_elbow_angle(points)
                shoulder_width = _calculate_shoulder_width(points)
                scores.extend([
                    _score_range(elbow_angle, rules.get('elbow_angle', (0, 180))),
                    _score_binary(shoulder_width > 0.3)  # 肩膀展开度
                ])
                
            elif category == 2:  # 防守类评估
                stance_stability = _calculate_stance_stability(points)
                racket_position = _calculate_racket_readiness(points)
                scores.extend([
                    _score_binary(stance_stability),
                    _score_binary(racket_position)
                ])
                
            else:  # 控制类评估
                wrist_stability = _calculate_wrist_stability(keypoints_seq)
                scores.append(_score_binary(wrist_stability))
            
        except Exception as e:
            # 如果计算出错，返回中等评分
            print(f"质量计算出错: {e}")
            return 0.5
        
        return np.mean(scores) if scores else 0.5
    
    @staticmethod
    def _generate_feedback(category: int, quality: float, keypoints_seq: List[Keypoints], 
                          raw_class: int) -> str:
        """
        生成个性化的改进建议
        基于质量评分和具体动作类型
        """
        from config import RAW_CLASSES
        
        action_name = RAW_CLASSES.get(raw_class, "未知动作")
        category_name = CATEGORIES[category]
        
        # 基础评价
        if quality > 0.8:
            base_feedback = f"{action_name}动作很标准！"
        elif quality > 0.6:
            base_feedback = f"{action_name}动作基本正确"
        elif quality > 0.4:
            base_feedback = f"{action_name}动作需要改进"
        else:
            base_feedback = f"{action_name}动作存在明显问题"
        
        # 具体建议
        specific_advice = ""
        if category == 0:  # 发球
            if quality < 0.7:
                specific_advice = "注意发球时肘部角度，保持稳定的重心。"
        elif category == 1:  # 进攻
            if quality < 0.7:
                specific_advice = "扣杀时注意充分展肩，发力要流畅。"
        elif category == 2:  # 防守
            if quality < 0.7:
                specific_advice = "防守时保持低重心，球拍准备充分。"
        else:  # 控制
            if quality < 0.7:
                specific_advice = "控制球时注意手腕稳定，动作要细腻。"
        
        return f"{base_feedback} {specific_advice}".strip()

# 几何计算辅助函数
def _calculate_elbow_angle(points: np.ndarray) -> float:
    """计算肘关节角度 - 基于肩膀-肘部-手腕三点"""
    try:
        # 正确的COCO关键点索引
        shoulder = points[6]   # 右肩
        elbow = points[4]      # 右肘 
        wrist = points[5]     # 右腕
        
        # 计算两个向量
        v1 = shoulder - elbow
        v2 = wrist - elbow
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    except:
        return 90.0

def _calculate_knee_bend(points: np.ndarray) -> float:
    """计算膝关节弯曲角度"""
    try:
        # 正确的COCO关键点索引
        hip = points[12]    # 右髋
        knee = points[8]   # 右膝
        ankle = points[9]  # 右踝
        
        v1 = hip - knee
        v2 = ankle - knee
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return 180 - angle
    except:
        return 15.0

def _calculate_shoulder_width(points: np.ndarray) -> float:
    """计算肩膀展开程度"""
    try:
        left_shoulder = points[5]
        right_shoulder = points[5]
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        return shoulder_width
    except:
        return 0.5  # 默认值

def _calculate_stance_stability(points: np.ndarray) -> bool:
    """评估站位稳定性"""
    try:
        left_ankle = points[15]
        right_ankle = points[6]
        foot_distance = np.linalg.norm(right_ankle - left_ankle)
        return foot_distance > 0.2  # 双脚距离合理
    except:
        return True

def _calculate_racket_readiness(points: np.ndarray) -> bool:
    """评估球拍准备姿势"""
    try:
        right_wrist = points[10]
        right_elbow = points[1]
        wrist_height = right_wrist[7]
        elbow_height = right_elbow[7]
        return wrist_height < elbow_height  # 手腕高于肘部
    except:
        return True

def _calculate_wrist_stability(keypoints_seq: List[Keypoints]) -> bool:
    """评估手腕稳定性（基于序列）"""
    try:
        if len(keypoints_seq) < 3:
            return True
        
        wrist_positions = []
        for kp in keypoints_seq:
            wrist_positions.append(kp.points[10])  # 右手腕
        
        # 计算手腕位置变化的标准差
        wrist_array = np.array(wrist_positions)
        stability = np.std(wrist_array) < 0.1  # 变化不大表示稳定
        return stability
    except:
        return True

def _score_range(value: float, target_range: Tuple[float, float]) -> float:
    """将数值映射到0-1评分范围"""
    min_val, max_val = target_range
    if min_val <= value <= max_val:
        return 1.0  # 完美范围内
    elif value < min_val:
        # 低于目标范围
        deviation = (min_val - value) / min_val
        return max(0.0, 1.0 - deviation)
    else:
        # 高于目标范围
        deviation = (value - max_val) / max_val
        return max(0.0, 1.0 - deviation)

def _score_binary(condition: bool) -> float:
    """二元条件评分"""
    return 1.0 if condition else 0.3
