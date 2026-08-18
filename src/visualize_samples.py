import os
import glob
import random
import cv2

def render_bounding_boxes(dataset_type="signs", count=15):
    if dataset_type == "signs":
        img_dir = r"d:\HocTap\Bien_bao\Data\Processed\traffic_signs_yolo\train\images"
        lbl_dir = r"d:\HocTap\Bien_bao\Data\Processed\traffic_signs_yolo\train\labels"
        out_dir = r"d:\HocTap\Bien_bao\Data\Reports\visual_samples\traffic_signs"
        names = {
            0: "Cam nguoc chieu",
            1: "Cam dung & do",
            2: "Cam re",
            3: "Gioi han toc do",
            4: "Cam con lai",
            5: "Nguy hiem",
            6: "Hieu lenh"
        }
        color_map = {
            0: (0, 0, 255),
            1: (0, 165, 255),
            2: (0, 0, 200),
            3: (255, 0, 0),
            4: (180, 0, 180),
            5: (0, 255, 255),
            6: (255, 200, 0)
        }
    else:
        img_dir = r"d:\HocTap\Bien_bao\Data\Processed\traffic_lights_yolo\train\images"
        lbl_dir = r"d:\HocTap\Bien_bao\Data\Processed\traffic_lights_yolo\train\labels"
        out_dir = r"d:\HocTap\Bien_bao\Data\Reports\visual_samples\traffic_lights"
        names = {0: "Green", 1: "Red"}
        color_map = {0: (0, 255, 0), 1: (0, 0, 255)}

    os.makedirs(out_dir, exist_ok=True)
    all_imgs = glob.glob(os.path.join(img_dir, "*.*"))
    if not all_imgs:
        return
        
    random.seed(42)
    selected = random.sample(all_imgs, min(count, len(all_imgs)))
    
    for idx, img_p in enumerate(selected):
        bname = os.path.basename(img_p)
        base_no_ext = os.path.splitext(bname)[0]
        lbl_p = os.path.join(lbl_dir, f"{base_no_ext}.txt")
        
        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        if os.path.exists(lbl_p):
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                x1 = max(0, int((xc - bw/2) * w))
                y1 = max(0, int((yc - bh/2) * h))
                x2 = min(w, int((xc + bw/2) * w))
                y2 = min(h, int((yc + bh/2) * h))
                
                col = color_map.get(cls_id, (0, 255, 0))
                lbl_name = names.get(cls_id, str(cls_id))
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                
                # Draw text background and text
                (tw, th), _ = cv2.getTextSize(lbl_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(th + 4, y1 - 4)
                cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), col, -1)
                cv2.putText(img, lbl_name, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
        out_file = os.path.join(out_dir, f"sample_{idx:02d}_{bname}")
        cv2.imwrite(out_file, img)

    print(f"[✓] Rendered {len(selected)} visual verification samples to: {out_dir}")

def main():
    print("\n" + "="*60)
    print("[4] GENERATING VISUAL VERIFICATION SAMPLES")
    print("="*60)
    render_bounding_boxes("signs", count=15)
    render_bounding_boxes("lights", count=15)

if __name__ == "__main__":
    main()
