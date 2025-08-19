import cv2
from ultralytics import YOLO
import numpy as np
import os

def preprocess_frame(frame):
    """提高图像质量，增强人体检测"""
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame

def safe_float(value):
    """
    安全的数值转换，处理所有numpy类型
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

def select_nearest_person(results, frame_height=640, frame_width=640):
    """从YOLOv8姿态检测结果中选择最靠近摄像头的人"""
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
    
    for i in range(len(keypoints_data.xy)):
        try:
            coords = keypoints_data.xy[i].cpu().numpy().astype(np.float64)
            confidence = keypoints_data.conf[i].cpu().numpy().astype(np.float64)
            
            valid_mask = confidence > 0.3
            valid_points = coords[valid_mask]
            valid_conf = confidence[valid_mask]
            
            if len(valid_points) < 5:
                person_scores.append((i, 0.0, "insufficient_keypoints"))
                continue
            
            min_xy = valid_points.min(axis=0)
            max_xy = valid_points.max(axis=0)
            
            bbox_width = safe_float(max_xy[0] - min_xy[0])
            bbox_height = safe_float(max_xy[1] - min_xy[1])
            bbox_area = bbox_width * bbox_height
            
            centroid_y = safe_float(valid_points[:, 1].mean())
            position_score = centroid_y / frame_height
            
            avg_confidence = safe_float(valid_conf.mean())
            
            # 综合评分
            area_weight = 0.5
            position_weight = 0.3
            confidence_weight = 0.2
            
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
            
            person_scores.append((
                i, 
                composite_score, 
                f"area:{bbox_area:.0f}_pos:{centroid_y:.0f}_conf:{avg_confidence:.2f}"
            ))
            
            if composite_score > max_score:
                max_score = composite_score
                best_idx = i
                
        except Exception as e:
            print(f"  ⚠️ 处理人物{i}时出错: {e}")
            person_scores.append((i, 0.0, f"error: {str(e)}"))
            continue
    
    # 调试输出
    print(f"  候选人评分:")
    for idx, score, details in person_scores:
        marker = "👑" if idx == best_idx else "  "
        print(f"  {marker} 人物{idx}: 得分{score:.3f} ({details})")
    
    if best_idx is None:
        return None, None, 0.0
    
    best_keypoints = keypoints_data.xy[best_idx].cpu().numpy()
    best_confidence = keypoints_data.conf[best_idx].cpu().numpy()
    
    return best_keypoints, best_confidence, max_score

def extract_keypoints_and_draw(video_path, output_dir, model_path='yolov8n-pose.pt', max_frames=50):
    """从视频中提取关键点并绘制骨架"""
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
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
            # 智能裁剪
            original_height, original_width = frame.shape[:2]
            crop_y1 = int(original_height * 0.15)
            crop_y2 = int(original_height * 0.90)
            crop_x1 = int(original_width * 0.10)
            crop_x2 = int(original_width * 0.90)
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 预处理
            processed_frame = preprocess_frame(cropped_frame)
            
            # 调整尺寸
            target_size = 640
            frame_resized = cv2.resize(processed_frame, (target_size, target_size))
            
            # YOLOv8检测
            results = model(frame_resized, verbose=False, conf=0.1)
            
            # 选择最佳人体
            best_keypoints, best_confidence, best_score = select_nearest_person(
                results, target_size, target_size
            )
            
            if best_keypoints is not None and best_confidence is not None:
                print(f"  ✅ 检测成功，最佳候选得分: {best_score:.3f}")
                
                # 创建绘制图像
                draw_img = frame_resized.copy()
                
                # 🔧 关键修正：安全处理所有坐标转换
                valid_keypoints = []
                valid_confidences = []
                
                # 预处理关键点，确保都是可用的标量
                for i in range(len(best_keypoints)):
                    try:
                        x = safe_float(best_keypoints[i][0])
                        y = safe_float(best_keypoints[i][1])
                        conf = safe_float(best_confidence[i])
                        
                        if conf > 0.3 and 0 <= x < target_size and 0 <= y < target_size:
                            valid_keypoints.append((i, x, y, conf))
                            valid_confidences.append(conf)
                    except Exception as e:
                        print(f"    ⚠️ 跳过关键点{i}: {e}")
                        continue
                
                # 绘制关键点
                for i, x, y, conf in valid_keypoints:
                    # 根据置信度设置颜色
                    if conf > 0.7:
                        color = (0, 255, 0)    # 绿色
                    elif conf > 0.5:
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
                        
                        if conf1 > 0.3 and conf2 > 0.3:
                            cv2.line(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    (0, 0, 255), 2)
                
                # 添加信息文本
                avg_conf = np.mean(valid_confidences) if valid_confidences else 0
                
                info_text = [
                    f"Frame: {frame_idx + 1}",
                    f"Score: {best_score:.3f}",
                    f"Avg Conf: {avg_conf:.2f}",
                    f"Valid Points: {len(valid_keypoints)}/17"
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
                print(f"  ❌ 未检测到有效人体")
        
        except Exception as e:
            print(f"  ⚠️ 处理帧 {frame_idx + 1} 时出错: {e}")
        
        frame_idx += 1
    
    cap.release()
    
    # 输出统计
    print("\n" + "=" * 50)
    print("🎉 处理完成！")
    print(f"📊 总处理帧数: {frame_idx}")
    print(f"✅ 成功检测帧数: {saved_idx}")
    
    if frame_idx > 0:
        print(f"📈 检测成功率: {saved_idx/frame_idx*100:.1f}%")
    
    print(f"📁 输出图片保存到: {output_dir}")
    
    if detection_stats:
        avg_score = sum(s['score'] for s in detection_stats) / len(detection_stats)
        avg_conf = sum(s['confidence'] for s in detection_stats) / len(detection_stats)
        avg_points = sum(s['valid_points'] for s in detection_stats) / len(detection_stats)
        
        print(f"📊 平均评分: {avg_score:.3f}")
        print(f"📊 平均置信度: {avg_conf:.3f}")
        print(f"📊 平均有效关键点: {avg_points:.1f}/17")

if __name__ == "__main__":
    test_video = "data/split/train/14_Smash/2022-08-30_19-10-16_dataset_set1_130_011066_011084_A_14.mp4"
    test_output = "debug_output"
    
    if os.path.exists(test_video):
        extract_keypoints_and_draw(test_video, test_output, max_frames=18)
    else:
        print("❌ 测试视频不存在")
