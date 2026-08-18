import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import glob
import time
import argparse
from src.pipeline import TrafficIntelligencePipeline

def create_sample_video_if_needed(output_path="Data/Reports/sample_traffic_video.mp4"):
    """Creates a smooth simulated video stream from sample traffic images if no video file is provided."""
    sample_imgs = glob.glob(r"Data/Reports/visual_samples/traffic_signs/*.png") + glob.glob(r"Data/Reports/visual_samples/traffic_lights/*.jpg")
    if not sample_imgs:
        return None
        
    print("[*] Generating simulated traffic video stream from sample road frames...")
    first_img = cv2.imread(sample_imgs[0])
    h, w = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 10, (w, h))
    
    for img_p in sample_imgs:
        img = cv2.imread(img_p)
        if img is None:
            continue
        resized = cv2.resize(img, (w, h))
        # Write each frame multiple times to simulate a smooth video stream
        for _ in range(8):
            out.write(resized)
            
    out.release()
    print(f"[✓] Created simulated video at: {output_path}")
    return output_path

def run_live_video_demo(source="sample", output_path=None, show_window=True, conf=0.18, imgsz=640):
    print("="*70)
    print("🚗🚦 VIETNAMESE TRAFFIC INTELLIGENCE — LIVE VIDEO STREAM DEMO")
    print(f"[*] Detection Settings: Sensitivity Conf={conf}, Inference Size={imgsz}")
    print("="*70)
    
    if source == "sample" or not os.path.exists(source) and source != "webcam":
        video_src = create_sample_video_if_needed()
        base_name = "sample_traffic_video"
    else:
        video_src = 0 if source == "webcam" else source
        base_name = os.path.splitext(os.path.basename(source))[0] if source != "webcam" else "webcam_demo"
        
    if not video_src:
        print("[!] Error: No video source found.")
        return
        
    if output_path is None:
        output_path = f"Data/Reports/{base_name}_annotated.mp4"
        
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"[!] Error: Cannot open video: {video_src}")
        return
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    pipeline = TrafficIntelligencePipeline()
    
    window_name = "🚦 VIETNAMESE TRAFFIC INTELLIGENCE (Press 'Q' to Exit)"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, w), min(720, h))
        
    print(f"\n[*] Playing and analyzing video live ({w}x{h} @ {fps} FPS)...")
    print("[*] Press 'Q' or 'ESC' on the video window anytime to stop.\n")
    
    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Process frame with High-Sensitivity parameters
        annotated_frame, data = pipeline.process_frame(frame, conf_threshold=conf, imgsz=imgsz)
        
        # Calculate real-time FPS
        elapsed = time.time() - start_time
        curr_fps = frame_idx / max(0.001, elapsed)
        cv2.putText(annotated_frame, f"FPS: {curr_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        out.write(annotated_frame)
        
        if show_window:
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q') or key == 27:
                print("[*] User stopped live video demo.")
                break
                
    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()
        
    print(f"\n[✓] Finished processing {frame_idx} frames at average {curr_fps:.1f} FPS!")
    print(f"[✓] Saved annotated video output to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="sample", help="Path to .mp4 video file, 'webcam', or 'sample'")
    parser.add_argument("--conf", type=float, default=0.18, help="Detection confidence threshold (default: 0.18 for high sensitivity)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (640 or 832 for distant small objects)")
    parser.add_argument("--no-display", action="store_true", help="Do not show pop-up window")
    args = parser.parse_args()
    
    run_live_video_demo(source=args.source, output_path=None, show_window=not args.no_display, conf=args.conf, imgsz=args.imgsz)
