"""
羽毛球视频动作分析应用
Linus原则：简单、直接、有效
输入视频 -> 输出分析结果
"""

import cv2
import argparse
import os
import json
from typing import List, Dict
from datetime import datetime

from detector import BadmintonDetector
from core import BadmintonShot
from config import RAW_CLASSES, CATEGORIES

class VideoAnalyzer:
    """
    视频分析器 - 专门处理视频文件的动作分析
    """
    
    def __init__(self, model_path: str):
        """
        初始化分析器
        
        Args:
            model_path: 训练好的模型路径
        """
        print("🚀 初始化羽毛球视频分析器...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
        
        # 初始化检测器
        self.detector = BadmintonDetector(action_model_path=model_path)
        print(f"✅ 模型加载成功: {model_path}")
    
    def analyze_video(self, video_path: str, output_path: str = None) -> List[BadmintonShot]:
        """
        分析视频中的羽毛球动作
        
        Args:
            video_path: 输入视频路径
            output_path: 结果保存路径 (可选)
            
        Returns:
            分析结果列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ 视频文件不存在: {video_path}")
        
        print(f"📹 开始分析视频: {video_path}")
        print("-" * 50)
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ 无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 视频信息:")
        print(f"   总帧数: {total_frames}")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"   时长: {duration:.1f} 秒")
        print("-" * 50)
        
        cap.release()
        
        # 重置检测器缓冲区
        self.detector.reset_buffer()
        
        # 开始分析
        results = self.detector.process_video(video_path)
        
        print(f"\n✅ 视频分析完成！")
        print(f"🎯 检测到 {len(results)} 个动作")
        
        # 打印分析结果
        self._print_results(results)
        
        # 保存结果（如果指定了输出路径）
        if output_path:
            self._save_results(results, video_path, output_path)
        
        return results
    
    def _print_results(self, results: List[BadmintonShot]):
        """打印分析结果到控制台"""
        if not results:
            print("😔 未检测到任何动作")
            return
        
        print("\n" + "=" * 80)
        print("🏸 动作分析结果")
        print("=" * 80)
        
        # 统计各类别数量
        category_counts = {}
        quality_scores = []
        
        for i, shot in enumerate(results, 1):
            # 显示单个动作结果
            raw_action = RAW_CLASSES.get(shot.raw_class, "未知动作")
            
            print(f"\n🎯 动作 #{i}")
            print(f"   原始分类: {raw_action}")
            print(f"   简化分类: {shot.category_name}")
            print(f"   质量评分: {shot.quality:.2f}/1.0")
            print(f"   分类置信度: {shot.confidence:.2f}")
            print(f"   改进建议: {shot.feedback}")
            
            # 统计数据
            category_counts[shot.category_name] = category_counts.get(shot.category_name, 0) + 1
            quality_scores.append(shot.quality)
        
        # 显示总体统计
        print("\n" + "=" * 80)
        print("📊 总体统计")
        print("=" * 80)
        
        print(f"🎯 动作类别分布:")
        for category, count in category_counts.items():
            percentage = (count / len(results)) * 100
            print(f"   {category}: {count} 次 ({percentage:.1f}%)")
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            max_quality = max(quality_scores)
            min_quality = min(quality_scores)
            
            print(f"\n📈 质量评分统计:")
            print(f"   平均质量: {avg_quality:.2f}/1.0")
            print(f"   最高质量: {max_quality:.2f}/1.0")
            print(f"   最低质量: {min_quality:.2f}/1.0")
            
            # 给出整体评价
            if avg_quality >= 0.8:
                overall_assessment = "优秀！动作质量很高 🏆"
            elif avg_quality >= 0.6:
                overall_assessment = "良好，还有提升空间 📈"
            elif avg_quality >= 0.4:
                overall_assessment = "一般，需要加强练习 💪"
            else:
                overall_assessment = "较差，建议寻求专业指导 📚"
            
            print(f"\n🏸 整体评价: {overall_assessment}")
    
    def _save_results(self, results: List[BadmintonShot], video_path: str, output_path: str):
        """保存分析结果到JSON文件"""
        try:
            # 准备结果数据
            analysis_data = {
                "video_info": {
                    "input_path": video_path,
                    "analysis_time": datetime.now().isoformat(),
                    "total_actions": len(results)
                },
                "actions": [],
                "summary": {}
            }
            
            category_counts = {}
            quality_scores = []
            
            # 处理每个动作
            for i, shot in enumerate(results):
                action_data = {
                    "action_id": i + 1,
                    "raw_class": shot.raw_class,
                    "raw_class_name": RAW_CLASSES.get(shot.raw_class, "未知动作"),
                    "category": shot.category,
                    "category_name": shot.category_name,
                    "quality_score": round(shot.quality, 3),
                    "classification_confidence": round(shot.confidence, 3),
                    "feedback": shot.feedback
                }
                analysis_data["actions"].append(action_data)
                
                # 统计数据
                category_counts[shot.category_name] = category_counts.get(shot.category_name, 0) + 1
                quality_scores.append(shot.quality)
            
            # 添加总结统计
            if quality_scores:
                analysis_data["summary"] = {
                    "category_distribution": category_counts,
                    "quality_statistics": {
                        "average": round(sum(quality_scores) / len(quality_scores), 3),
                        "maximum": round(max(quality_scores), 3),
                        "minimum": round(min(quality_scores), 3)
                    }
                }
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 分析结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"⚠️  保存结果时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="羽毛球视频动作分析工具")
    
    # 必需参数
    parser.add_argument("--video", "-v", required=True, 
                       help="输入视频文件路径")
    parser.add_argument("--model", "-m", required=True,
                       help="训练好的模型文件路径")
    
    # 可选参数
    parser.add_argument("--output", "-o", 
                       help="结果保存路径 (JSON格式)")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细输出")
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = VideoAnalyzer(args.model)
        
        # 生成默认输出路径（如果未指定）
        if not args.output:
            video_name = os.path.splitext(os.path.basename(args.video))[0]
            args.output = f"{video_name}_analysis.json"
        
        # 分析视频
        results = analyzer.analyze_video(args.video, args.output)
        
        print(f"\n🎉 分析完成！共检测到 {len(results)} 个动作")
        
    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
