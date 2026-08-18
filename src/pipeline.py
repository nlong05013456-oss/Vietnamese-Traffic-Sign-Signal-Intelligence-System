import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from src.temporal_tracker import TrafficLightTracker
from src.digit_localizer import DigitLocalizer
from PIL import Image

class TrafficIntelligencePipeline:
    """
    End-to-End Multi-Task Computer Vision Pipeline:
    1. Sign Detection (YOLO11n @ 50 epochs GPU weights)
    2. Traffic Light Detection (YOLO11n @ 50 epochs GPU weights)
    3. Dedicated Countdown Box Finder & Digit Recognition (PyTorch CNN)
    4. Temporal State Tracking & Driver HUD Overlay
    """
    def __init__(self, signs_model_path=None, lights_model_path=None, digit_weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default trained model paths
        default_signs = r"d:\HocTap\Bien_bao\runs\detect\traffic_signs_yolo11n_gpu\weights\best.pt"
        default_lights = r"d:\HocTap\Bien_bao\runs\detect\traffic_lights_yolo11n_gpu\weights\best.pt"
        default_digit = r"d:\HocTap\Bien_bao\runs\classify\countdown_digits_baseline\best_digit_classifier.pth"
        
        p_signs = signs_model_path if (signs_model_path and os.path.exists(signs_model_path)) else default_signs
        p_lights = lights_model_path if (lights_model_path and os.path.exists(lights_model_path)) else default_lights
        p_digit = digit_weights_path if (digit_weights_path and os.path.exists(digit_weights_path)) else default_digit
        
        self.signs_model = YOLO(p_signs if os.path.exists(p_signs) else "yolo11n.pt")
        self.lights_model = YOLO(p_lights if os.path.exists(p_lights) else "yolo11n.pt")
        self.digit_localizer = DigitLocalizer(weights_path=p_digit)
        self.tracker = TrafficLightTracker()
        
        self.sign_labels = {
            0: "Cam nguoc chieu",
            1: "Cam dung va do",
            2: "Cam re",
            3: "Gioi han toc do",
            4: "Cam con lai",
            5: "Nguy hiem",
            6: "Hieu lenh"
        }

    def _find_countdown_around_light(self, frame, light_xyxy, light_color="Green"):
        """
        Searches the spatial neighborhood around a detected Traffic Light
        for active 7-segment LED countdown digit displays.
        """
        h_img, w_img = frame.shape[:2]
        lx1, ly1, lx2, ly2 = light_xyxy
        lw = lx2 - lx1
        lh = ly2 - ly1
        
        # Define search neighborhood: Traffic timer is typically directly on the Left or Right
        margin_x = int(lw * 1.6)
        margin_y = int(lh * 0.4)
        
        rx1 = max(0, lx1 - margin_x)
        ry1 = max(0, ly1 - margin_y)
        rx2 = min(w_img, lx2 + margin_x)
        ry2 = min(h_img, ly2 + margin_y)
        
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None, None
            
        # Segment glowing LED color (Green or Red)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if light_color == "Green":
            mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([90, 255, 255]))
        else:
            mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
            mask = mask1 | mask2
            
        # Find glowing segment clusters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Skip the traffic light bulb itself (inside light_xyxy)
            abs_x1 = rx1 + bx
            abs_y1 = ry1 + by
            abs_x2 = abs_x1 + bw
            abs_y2 = abs_y1 + bh
            
            # Check if this contour is outside the light bulb and has digit-like aspect ratio
            is_inside_light = (abs_x1 >= lx1 and abs_x2 <= lx2 and abs_y1 >= ly1 and abs_y2 <= ly2)
            if not is_inside_light and bw >= 10 and bh >= 14:
                best_box = (abs_x1 - 4, abs_y1 - 4, abs_x2 + 4, abs_y2 + 4)
                break
                
        if best_box:
            # Crop the countdown box
            cx1 = max(0, best_box[0])
            cy1 = max(0, best_box[1])
            cx2 = min(w_img, best_box[2])
            cy2 = min(h_img, best_box[3])
            timer_crop = frame[cy1:cy2, cx1:cx2]
            
            if timer_crop.size > 0:
                val, confs, _ = self.digit_localizer.recognize_countdown(timer_crop)
                return val, [cx1, cy1, cx2, cy2]
                
        return None, None

    def process_frame(self, frame, conf_threshold=0.20, imgsz=640):
        """
        Process a single image or video frame.
        Enhanced for High-Sensitivity & Distant Small-Object Detection.
        """
        h, w = frame.shape[:2]
        output_data = {
            "traffic_signs": [],
            "traffic_light": None,
            "countdown_timer": None,
            "system_status": "ONLINE"
        }
        
        annotated = frame.copy()
        
        # 1. Detect Traffic Signs (High sensitivity)
        sign_results = self.signs_model.predict(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)
        for r in sign_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                sign_name = self.sign_labels.get(cls_id, f"Sign {cls_id}")
                output_data["traffic_signs"].append({
                    "class": sign_name,
                    "confidence": round(conf, 2),
                    "box": xyxy.tolist()
                })
                
                # Draw sign on frame
                cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 140, 255), 2)
                cv2.putText(annotated, f"{sign_name} ({conf:.2f})", (xyxy[0], max(20, xyxy[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        # 2. Detect Traffic Lights & Linked Countdown Display
        detected_light = None
        light_conf = 0.0
        countdown_val = None
        
        light_results = self.lights_model.predict(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)
        for r in light_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                light_name = "Green" if cls_id == 0 else "Red"
                detected_light = light_name
                light_conf = conf
                
                col = (0, 255, 0) if light_name == "Green" else (0, 0, 255)
                cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), col, 2)
                cv2.putText(annotated, f"LIGHT: {light_name} ({conf:.2f})", (xyxy[0], max(20, xyxy[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                
                # Search for adjacent Countdown LED Box
                val, timer_box = self._find_countdown_around_light(frame, xyxy, light_name)
                if val is not None:
                    countdown_val = val
                    output_data["countdown_timer"] = {"value": val, "box": timer_box}
                    # Draw Timer Bounding Box in Cyan
                    cv2.rectangle(annotated, (timer_box[0], timer_box[1]), (timer_box[2], timer_box[3]), (255, 255, 0), 2)
                    cv2.putText(annotated, f"TIMER: {val}s", (timer_box[0], max(20, timer_box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                break
                
        # 3. Update Temporal State Tracker
        tracker_summary = self.tracker.update(detected_light, light_conf, countdown_val=countdown_val)
        output_data["traffic_light"] = tracker_summary
        
        # 4. Render Driver HUD Dashboard in Top-Right
        self._render_hud(annotated, output_data)
        
        return annotated, output_data

    def _render_hud(self, img, data):
        """Renders ADAS Intelligence Overlay Box"""
        hud_w, hud_h = 320, 130
        h, w = img.shape[:2]
        x1, y1 = w - hud_w - 20, 20
        x2, y2 = x1 + hud_w, y1 + hud_h
        
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return
            
        sub_img = img[y1:y2, x1:x2]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 1.0)
        img[y1:y2, x1:x2] = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # HUD Text
        cv2.putText(img, "TRAFFIC ADAS INTELLIGENCE", (x1 + 12, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        
        tl = data.get("traffic_light", {})
        state = tl.get("state", "UNKNOWN")
        state_col = (0, 255, 0) if state == "Green" else ((0, 0, 255) if state == "Red" else (200, 200, 200))
        
        cv2.putText(img, f"Light: {state} ({tl.get('confidence', 0.0)})", (x1 + 12, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_col, 1, cv2.LINE_AA)
        cv2.putText(img, f"Timer: {tl.get('remaining_time', 'N/A')}", (x1 + 12, y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        signs_count = len(data.get("traffic_signs", []))
        cv2.putText(img, f"Active Signs: {signs_count} detected", (x1 + 12, y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
