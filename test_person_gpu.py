"""
GPU优化版本 - 羽毛球人体关键点检测和骨架绘制 (最终稳定版)
解决所有API兼容性问题，确保在各种PyTorch版本下都能稳定运行

核心策略：
1. 使用传统稳定的API
2. 增强错误处理和容错性
3. 保持原版test_person.py的成功算法
4. 简化混合精度处理，优先保证功能
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import time
import gc
from typing import Optional, Tuple, List, Dict, Any

class GPUOptimizedDetector:
    """GPU优化的人体检测器 - 稳定版"""
    
    def __init__(self, model_path: str = 'yolov8m-pose.pt', device: Optional[str] = None):
        """初始化GPU优化检测器"""
        # 设备配置
        self.device = self._setup_device(device)
        self.use_amp = False  # 暂时禁用混合精度，确保稳定性
        
        print(f"🚀 初始化GPU优化检测器 (稳定版)")
        print(f"🔧 使用设备: {self.device}")
        print(f"⚡ 混合精度: {self.use_amp} (为确保稳定性暂时禁用)")
        
        # 根据设备选择合适的模型
        if self.device.type == 'cuda':
            if 'n-pose' in model_path:
                model_path = model_path.replace('n-pose', 'm-pose')
            print(f"🎯 GPU模式：使用更大模型 {model_path}")
        else:
            print(f"🎯 CPU模式：使用轻量模型 {model_path}")
        
        # 初始化YOLO模型
        try:
            self.model = YOLO(model_path)
            if self.device.type == 'cuda':
                # 简化GPU设置，避免复杂API
                torch.backends.cudnn.benchmark = True
                self._simple_warmup()
            
            print(f"✅ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """设置设备"""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🎯 自动检测到GPU: {gpu_name} ({vram_gb:.1f}GB)")
            else:
                device = 'cpu'
                print("⚠️ 未检测到GPU，使用CPU模式")
        
        device = torch.device(device)
        
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("❌ CUDA不可用，回退到CPU模式")
            device = torch.device('cpu')
        
        return device
    
    def _simple_warmup(self):
        """简化的GPU预热"""
        try:
            print("🔥 GPU预热中...")
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 简单预热，不使用混合精度
            with torch.no_grad():
                _ = self.model(dummy_frame, verbose=False)
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print("✅ GPU预热完成")
        except Exception as e:
            print(f"⚠️ GPU预热失败: {e}")

def safe_float_robust(value):
    """
    最强健的数值转换函数
    """
    try:
        # 处理各种numpy类型
        if hasattr(value, 'item'):  # numpy标量
            return float(value.item())
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            elif value.size == 1:
                return float(value.flat[0])
            else:
                # 多元素数组，记录警告但不崩溃
                print(f"    ⚠️ 多元素数组转标量: shape={value.shape}, 取首元素")
                return float(value.flat[0])
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            # 其他类型尝试直接转换
            return float(value)
    except Exception as e:
        print(f"    ❌ 数值转换失败: {type(value)}, {e}, 返回0.0")
        return 0.0

def select_nearest_person_robust(results, frame_height=640, frame_width=640):
    """
    最稳健的人体选择算法
    基于原版test_person.py，增加全方位错误处理
    """
    try:
        if not results or len(results) == 0:
            return None, None, 0

        result = results[0]

        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return None, None, 0

        keypoints_data = result.keypoints

        if len(keypoints_data.xy) == 0:
            return None, None, 0

        best_idx = None
        max_score = 0.0
        person_scores = []
        max_possible_area = float(frame_height * frame_width)

        print(f"    🔍 发现 {len(keypoints_data.xy)} 个人物候选")

        for i in range(len(keypoints_data.xy)):
            try:
                # 安全获取数据
                coords = keypoints_data.xy[i].cpu().numpy().astype(np.float64)
                confidence = keypoints_data.conf[i].cpu().numpy().astype(np.float64)

                # 检查数据有效性
                if coords.shape[0] == 0 or confidence.shape[0] == 0:
                    person_scores.append((i, 0.0, "empty_data"))
                    continue

                # 过滤低置信度关键点
                valid_mask = confidence > 0.2  # 降低阈值
                valid_points = coords[valid_mask]
                valid_conf = confidence[valid_mask]

                if len(valid_points) < 2:  # 最低要求2个关键点
                    person_scores.append((i, 0.0, "insufficient_keypoints"))
                    continue

                # 安全计算包围盒
                try:
                    min_vals = np.min(valid_points, axis=0)
                    max_vals = np.max(valid_points, axis=0)
                    
                    bbox_width = safe_float_robust(max_vals[0] - min_vals)
                    bbox_height = safe_float_robust(max_vals[1] - min_vals[1])
                    bbox_area = max(bbox_width * bbox_height, 1.0)  # 避免0面积
                    
                    # 计算质心
                    centroid_y = safe_float_robust(np.mean(valid_points[:, 1]))
                    position_score = max(0.0, min(1.0, centroid_y / frame_height))
                    
                    # 计算平均置信度
                    avg_confidence = safe_float_robust(np.mean(valid_conf))
                    
                    # 综合评分（保持原算法）
                    area_weight = 0.5
                    position_weight = 0.3
                    confidence_weight = 0.2
                    
                    normalized_area = min(1.0, np.log10(bbox_area + 1) / np.log10(max_possible_area + 1))
                    normalized_position = position_score
                    normalized_confidence = avg_confidence
                    
                    composite_score = (
                        area_weight * normalized_area +
                        position_weight * normalized_position +
                        confidence_weight * normalized_confidence
                    )
                    
                    person_scores.append((
                        i,
                        composite_score,
                        f"area:{bbox_area:.0f}_pos:{centroid_y:.0f}_conf:{avg_confidence:.2f}_pts:{len(valid_points)}"
                    ))
                    
                    if composite_score > max_score:
                        max_score = composite_score
                        best_idx = i
                        
                except Exception as calc_e:
                    person_scores.append((i, 0.0, f"calc_error: {str(calc_e)[:30]}"))
                    continue
                    
            except Exception as person_e:
                person_scores.append((i, 0.0, f"person_error: {str(person_e)[:30]}"))
                continue

        # 调试输出
        print(f"    候选人评分:")
        for idx, score, details in person_scores:
            marker = "👑" if idx == best_idx else "  "
            print(f"    {marker} 人物{idx}: 得分{score:.3f} ({details})")

        if best_idx is None:
            return None, None, 0.0

        # 安全返回结果
        try:
            best_keypoints = keypoints_data.xy[best_idx].cpu().numpy()
            best_confidence = keypoints_data.conf[best_idx].cpu().numpy()
            return best_keypoints, best_confidence, max_score
        except Exception as return_e:
            print(f"    ❌ 返回结果时出错: {return_e}")
            return None, None, 0.0
            
    except Exception as main_e:
        print(f"  ❌ 人体选择主函数出错: {main_e}")
        return None, None, 0.0

def preprocess_frame_simple(frame):
    """简化的图像预处理"""
    try:
        # 提高对比度和亮度
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        # 锐化滤波
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    except Exception as e:
        print(f"    ⚠️ 预处理失败: {e}, 返回原图")
        return frame

def extract_keypoints_and_draw_stable(video_path: str, 
                                     output_dir: str, 
                                     model_path: str = 'yolov8m-pose.pt',
                                     max_frames: int = 50,
                                     device: Optional[str] = None):
    """
    最稳定版本的关键点提取和绘制
    """
    
    # 初始化检测器
    detector = GPUOptimizedDetector(model_path, device)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return
    except Exception as e:
        print(f"❌ 视频加载失败: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"🎬 处理视频: {os.path.basename(video_path)}")
    print(f"📊 视频信息: {width}x{height}, {total_frames}帧, {fps:.1f}fps")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 最大处理帧数: {max_frames}")
    print("-" * 50)

    frame_idx = 0
    saved_idx = 0
    detection_stats = []
    
    total_start_time = time.time()

    # 骨架连接定义
    skeleton_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),        # 头部
        (5, 6), (5, 11), (6, 12), (11, 12),    # 躯干
        (5, 7), (7, 9),                        # 左臂
        (6, 8), (8, 10),                       # 右臂
        (11, 13), (13, 15),                    # 左腿
        (12, 14), (14, 16)                     # 右腿
    ]

    while frame_idx < max_frames and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n🔍 处理帧 {frame_idx + 1}/{min(max_frames, total_frames)}")

        try:
            # 保守的智能裁剪
            original_height, original_width = frame.shape[:2]
            crop_y1 = max(0, int(original_height * 0.02))  # 非常保守的裁剪
            crop_y2 = min(original_height, int(original_height * 0.98))
            crop_x1 = max(0, int(original_width * 0.02))
            crop_x2 = min(original_width, int(original_width * 0.98))
            
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # 预处理
            processed_frame = preprocess_frame_simple(cropped_frame)

            # 调整尺寸
            target_size = 640
            frame_resized = cv2.resize(processed_frame, (target_size, target_size))

            # 稳定的YOLO检测
            try:
                with torch.no_grad():
                    results = detector.model(
                        frame_resized, 
                        verbose=False, 
                        conf=0.03  # 进一步降低阈值
                    )
            except Exception as detection_e:
                print(f"  ❌ YOLO检测失败: {detection_e}")
                continue

            # 选择最佳人体
            best_keypoints, best_confidence, best_score = select_nearest_person_robust(
                results, target_size, target_size
            )

            if best_keypoints is not None and best_confidence is not None and best_score > 0:
                print(f"  ✅ 检测成功，最佳候选得分: {best_score:.3f}")

                # 创建绘制图像
                draw_img = frame_resized.copy()

                # 处理关键点
                valid_keypoints = []
                valid_confidences = []

                for i in range(min(len(best_keypoints), 17)):  # 确保不超过17个关键点
                    try:
                        x = safe_float_robust(best_keypoints[i][0])
                        y = safe_float_robust(best_keypoints[i][1])
                        conf = safe_float_robust(best_confidence[i]) if i < len(best_confidence) else 0.0

                        # 更宽松的条件
                        if conf > 0.1 and 0 <= x < target_size and 0 <= y < target_size:
                            valid_keypoints.append((i, x, y, conf))
                            valid_confidences.append(conf)
                    except Exception as kp_e:
                        continue

                print(f"    📍 有效关键点: {len(valid_keypoints)}/17")

                if len(valid_keypoints) >= 2:  # 至少2个关键点才绘制
                    # 绘制关键点
                    for i, x, y, conf in valid_keypoints:
                        if conf > 0.6:
                            color = (0, 255, 0)    # 绿色
                        elif conf > 0.3:
                            color = (0, 255, 255)  # 黄色
                        else:
                            color = (0, 0, 255)    # 红色

                        cv2.circle(draw_img, (int(x), int(y)), 4, color, -1)
                        cv2.putText(draw_img, str(i), (int(x)-5, int(y)-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

                    # 绘制骨架
                    kp_dict = {i: (x, y, conf) for i, x, y, conf in valid_keypoints}

                    for (p1, p2) in skeleton_pairs:
                        if p1 in kp_dict and p2 in kp_dict:
                            x1, y1, conf1 = kp_dict[p1]
                            x2, y2, conf2 = kp_dict[p2]

                            if conf1 > 0.15 and conf2 > 0.15:
                                cv2.line(draw_img, (int(x1), int(y1)), (int(x2), int(y2)),
                                        (0, 0, 255), 2)

                    # 添加信息
                    avg_conf = np.mean(valid_confidences) if valid_confidences else 0

                    info_text = [
                        f"Frame: {frame_idx + 1}",
                        f"Score: {best_score:.3f}",
                        f"Conf: {avg_conf:.2f}",
                        f"Points: {len(valid_keypoints)}/17"
                    ]

                    for i, text in enumerate(info_text):
                        cv2.putText(draw_img, text, (10, 25 + i * 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 保存图片
                    output_filename = f"frame_{saved_idx:04d}_score_{best_score:.3f}.jpg"
                    output_path = os.path.join(output_dir, output_filename)

                    if cv2.imwrite(output_path, draw_img):
                        print(f"  💾 已保存: {output_filename}")
                        saved_idx += 1

                        detection_stats.append({
                            'frame': frame_idx,
                            'score': float(best_score),
                            'confidence': float(avg_conf),
                            'valid_points': len(valid_keypoints)
                        })
                    else:
                        print(f"  ❌ 保存失败: {output_filename}")
                else:
                    print(f"  ⚠️ 有效关键点不足，跳过绘制")
            else:
                print(f"  ❌ 未检测到有效人体")

        except Exception as frame_e:
            print(f"  ⚠️ 处理帧 {frame_idx + 1} 时出错: {frame_e}")

        frame_idx += 1

        # GPU内存管理
        if detector.device.type == 'cuda' and frame_idx % 5 == 0:
            torch.cuda.empty_cache()

    cap.release()

    # 最终统计
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 50)
    print("🎉 处理完成！")
    print(f"📊 总处理帧数: {frame_idx}")
    print(f"✅ 成功检测帧数: {saved_idx}")
    
    if frame_idx > 0:
        success_rate = saved_idx/frame_idx*100
        print(f"📈 检测成功率: {success_rate:.1f}%")
        print(f"⏱️  总处理时间: {total_time:.1f}秒")
        print(f"🚀 平均FPS: {frame_idx/total_time:.1f}")
    
    print(f"📁 输出图片保存到: {output_dir}")

    if detection_stats:
        avg_score = sum(s['score'] for s in detection_stats) / len(detection_stats)
        avg_conf = sum(s['confidence'] for s in detection_stats) / len(detection_stats)
        avg_points = sum(s['valid_points'] for s in detection_stats) / len(detection_stats)

        print(f"📊 平均评分: {avg_score:.3f}")
        print(f"📊 平均置信度: {avg_conf:.3f}")
        print(f"📊 平均有效关键点: {avg_points:.1f}/17")
    
    # 成功率评价
    if saved_idx > 0:
        success_rate = saved_idx/frame_idx*100
        if success_rate >= 50:
            print(f"🏆 检测成功率很高: {success_rate:.1f}%")
        elif success_rate >= 20:
            print(f"👍 检测成功率良好: {success_rate:.1f}%")
        else:
            print(f"📈 检测成功率较低: {success_rate:.1f}%，可能需要调整参数")

    # 清理
    if detector.device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU优化的羽毛球人体关键点检测 (最稳定版)")
    parser.add_argument("--video", default="data/split/train/14_Smash/2022-08-30_19-10-16_dataset_set1_130_011066_011084_A_14.mp4",
                       help="输入视频路径")
    parser.add_argument("--output", default="debug_output_stable",
                       help="输出目录")
    parser.add_argument("--model", default="yolov8m-pose.pt",
                       help="YOLO模型路径")
    parser.add_argument("--max-frames", type=int, default=50,
                       help="最大处理帧数")
    parser.add_argument("--device", choices=['auto', 'cuda', 'cpu'], default='auto',
                       help="指定设备")
    parser.add_argument("--cpu", action="store_true",
                       help="强制使用CPU模式")
    
    args = parser.parse_args()
    
    # 设备选择
    if args.cpu:
        device = 'cpu'
    elif args.device == 'auto':
        device = None
    else:
        device = args.device
    
    # 检查视频文件
    if os.path.exists(args.video):
        extract_keypoints_and_draw_stable(
            video_path=args.video,
            output_dir=args.output,
            model_path=args.model,
            max_frames=args.max_frames,
            device=device
        )
    else:
        print(f"❌ 视频文件不存在: {args.video}")
        print("请确认视频路径正确，或使用以下命令指定视频：")
        print(f"python {__file__} --video /path/to/your/video.mp4")

if __name__ == "__main__":
    main()
