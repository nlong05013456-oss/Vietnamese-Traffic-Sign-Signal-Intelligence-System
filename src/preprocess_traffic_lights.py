import os
import glob
import shutil
import cv2

def clean_and_standardize_traffic_lights():
    print("\n" + "="*60)
    print("[2] PREPROCESSING: VIETNAM TRAFFIC LIGHTS DATASET")
    print("="*60)
    
    raw_dir = r"d:\HocTap\Bien_bao\Data\Raw\traffic_lights_vn"
    out_base = r"d:\HocTap\Bien_bao\Data\Processed\traffic_lights_yolo"
    
    for split_raw, split_target in [("train", "train"), ("valid", "val"), ("test", "test")]:
        os.makedirs(os.path.join(out_base, split_target, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_base, split_target, "labels"), exist_ok=True)
        
        src_img_dir = os.path.join(raw_dir, split_raw, "images")
        src_lbl_dir = os.path.join(raw_dir, split_raw, "labels")
        
        dst_img_dir = os.path.join(out_base, split_target, "images")
        dst_lbl_dir = os.path.join(out_base, split_target, "labels")
        
        img_files = glob.glob(os.path.join(src_img_dir, "*.*"))
        print(f"[*] Processing split '{split_target}': {len(img_files)} images...")
        
        cleaned_boxes = 0
        total_boxes = 0
        
        for img_p in img_files:
            bname = os.path.basename(img_p)
            base_no_ext = os.path.splitext(bname)[0]
            
            src_lbl = os.path.join(src_lbl_dir, f"{base_no_ext}.txt")
            dst_img = os.path.join(dst_img_dir, bname)
            dst_lbl = os.path.join(dst_lbl_dir, f"{base_no_ext}.txt")
            
            # Copy image
            shutil.copy2(img_p, dst_img)
            
            if not os.path.exists(src_lbl):
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    f.write("")
                continue
                
            clean_lines = []
            with open(src_lbl, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                    
                if bw <= 0 or bh <= 0:
                    continue
                    
                # Fix / clip coordinates to valid [0, 1] range
                orig_xc, orig_yc, orig_bw, orig_bh = xc, yc, bw, bh
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                bw = min(max(bw, 0.0001), 1.0)
                bh = min(max(bh, 0.0001), 1.0)
                
                # Check if box extends beyond boundaries
                if xc - bw/2 < 0:
                    bw = 2 * xc
                if xc + bw/2 > 1.0:
                    bw = 2 * (1.0 - xc)
                if yc - bh/2 < 0:
                    bh = 2 * yc
                if yc + bh/2 > 1.0:
                    bh = 2 * (1.0 - yc)
                    
                if (orig_xc != xc or orig_yc != yc or orig_bw != bw or orig_bh != bh):
                    cleaned_boxes += 1
                    
                clean_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                total_boxes += 1
                
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(clean_lines))
                
        print(f"    - Done '{split_target}': {total_boxes} valid bboxes (cleaned/clipped {cleaned_boxes} edge bboxes).")
        
    yaml_content = f"""path: d:/HocTap/Bien_bao/Data/Processed/traffic_lights_yolo
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: Green
  1: Red
"""
    yaml_path = os.path.join(out_base, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[✓] Created Traffic Lights YAML at: {yaml_path}")

if __name__ == "__main__":
    clean_and_standardize_traffic_lights()
