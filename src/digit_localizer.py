import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.train_countdown_digits import CountdownDigitCNN

class DigitLocalizer:
    """
    Dedicated Digit Localization & Reading Pipeline:
    Task 1: Segment & Localize individual digits inside Countdown ROI
    Task 2: Sort localized boxes Left-to-Right
    Task 3: Classify each digit via PyTorch CNN (0-9)
    Task 4: Reconstruct full multi-digit countdown string (e.g. '53', '27', '03')
    """
    def __init__(self, weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CountdownDigitCNN(num_classes=10).to(self.device)
        
        default_weights = r"d:\HocTap\Bien_bao\runs\classify\countdown_digits_baseline\best_digit_classifier.pth"
        path_to_load = weights_path if weights_path and os.path.exists(weights_path) else default_weights
        
        if os.path.exists(path_to_load):
            self.model.load_state_dict(torch.load(path_to_load, map_location=self.device))
            self.model.eval()
            print(f"[✓] Digit Localizer loaded classifier weights: {path_to_load}")
        else:
            print("[!] Warning: Digit classifier weights not found, using untrained backbone.")
            
        self.transform = transforms.Compose([
            transforms.Resize((96, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def localize_digit_boxes(self, roi_img):
        """
        Localizes individual digit bounding boxes from a Countdown ROI.
        Handles both 1-digit and 2-digit displays.
        """
        if roi_img is None or roi_img.size == 0:
            return []
            
        h, w = roi_img.shape[:2]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img
        
        # Adaptive thresholding to extract active LED bars
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw * bh >= (h * w) * 0.08:
                # If the bounding box is wide (contains 2 merged digits), split horizontally
                if bw >= bh * 0.70:
                    mid = bw // 2
                    boxes.append((bx, by, mid, bh))
                    boxes.append((bx + mid, by, bw - mid, bh))
                else:
                    boxes.append((bx, by, bw, bh))
                    
        # If thresholding missed or timer is standard rectangular 2-digit display
        if len(boxes) == 0:
            mid_w = w // 2
            boxes = [(0, 0, mid_w, h), (mid_w, 0, w - mid_w, h)]
            
        # Sort Left to Right
        boxes = sorted(boxes, key=lambda b: b[0])
        return boxes

    def recognize_countdown(self, roi_img):
        """
        End-to-End: Localize -> Classify -> Reconstruct string (e.g. '53')
        Returns: (reconstructed_int_or_str, confidence_list, debug_boxes)
        """
        boxes = self.localize_digit_boxes(roi_img)
        if not boxes:
            return None, [], []
            
        digits_read = []
        confidences = []
        
        for (bx, by, bw, bh) in boxes:
            # Crop with safe boundary
            crop = roi_img[max(0, by):min(roi_img.shape[0], by+bh), max(0, bx):min(roi_img.shape[1], bx+bw)]
            if crop.size == 0:
                continue
            
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_cls = torch.max(probs, 1)
                
            digits_read.append(str(pred_cls.item()))
            confidences.append(float(conf.item()))
            
        if digits_read:
            full_str = "".join(digits_read)
            try:
                val = int(full_str)
            except ValueError:
                val = full_str
            return val, confidences, boxes
        return None, [], []
