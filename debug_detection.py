import cv2
from ultralytics import YOLO
import numpy as np

def debug_single_video(video_path):
    """调试单个视频的检测情况 - 修正版"""
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(video_path)
    
    print(f"🔍 调试视频: {video_path}")
    
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 视频信息: {width}x{height}, {total_frames}帧, {fps}fps")
    
    # 测试多种检测策略
    strategies = [
        ("原始帧", lambda f: f),
        ("裁剪帧", lambda f: f[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]),
        ("放大裁剪", lambda f: cv2.resize(f[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.8)], (640, 640)))
    ]
    
    for strategy_name, preprocess in strategies:
        print(f"\n🧪 测试策略: {strategy_name}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
        
        detections = 0
        test_frames = min(20, total_frames)
        
        for i in range(test_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                processed_frame = preprocess(frame)
                if processed_frame.size == 0:
                    continue
                
                # 🔧 修正：适配新版ultralytics API
                results = model(processed_frame, verbose=False, conf=0.01)
                
                # 🔧 关键修正：results是列表，取第一个元素
                if results and len(results) > 0:
                    result = results[0]  # 取第一个结果
                    
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints_data = result.keypoints
                        
                        if len(keypoints_data.xy) > 0:
                            detections += 1
                            print(f"  ✅ 帧 {i}: 检测到 {len(keypoints_data.xy)} 个人体")
                            
                            # 显示检测详情
                            for j, kp_tensor in enumerate(keypoints_data.xy):
                                if j < len(keypoints_data.conf):
                                    conf = keypoints_data.conf[j].cpu().numpy()
                                    print(f"    人物{j}: 平均置信度 {conf.mean():.3f}")
                        else:
                            print(f"  ❌ 帧 {i}: 检测到keypoints但无坐标数据")
                    else:
                        print(f"  ❌ 帧 {i}: 无keypoints检测")
                else:
                    print(f"  ❌ 帧 {i}: 无结果返回")
                    
            except Exception as e:
                print(f"  ⚠️ 帧 {i}: 处理出错 {e}")
        
        success_rate = detections / test_frames * 100
        print(f"📈 {strategy_name} 成功率: {success_rate:.1f}% ({detections}/{test_frames})")
    
    cap.release()

# 测试一个具体的失败视频
debug_single_video("data/split/train/14_Smash/2022-08-30_19-10-16_dataset_set1_130_011066_011084_A_14.mp4")
