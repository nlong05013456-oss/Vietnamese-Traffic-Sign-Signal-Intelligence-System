import os
import glob
import shutil
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def convert_zalo_to_yolo():
    print("="*60)
    print("[1] PREPROCESSING: ZALO TRAFFIC SIGNS 2020 -> YOLO")
    print("="*60)
    
    raw_dir = r"d:\HocTap\Bien_bao\Data\Raw\traffic_signs_zalo\traffic_train"
    csv_path = os.path.join(raw_dir, "annotation.csv")
    img_dir = os.path.join(raw_dir, "images")
    
    out_base = r"d:\HocTap\Bien_bao\Data\Processed\traffic_signs_yolo"
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_base, split, "labels"), exist_ok=True)
        
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} annotations across {df['file_name'].nunique()} unique images.")
    
    # 1. Group Split by street_id (70% train, 15% val, 15% test) to prevent Data Leakage
    unique_imgs_df = df.drop_duplicates(subset=['file_name'])[['file_name', 'street_id']].reset_index(drop=True)
    
    # First split: 70% train, 30% temp (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(gss1.split(unique_imgs_df, groups=unique_imgs_df['street_id']))
    
    train_files = set(unique_imgs_df.iloc[train_idx]['file_name'])
    temp_df = unique_imgs_df.iloc[temp_idx].reset_index(drop=True)
    
    # Second split: split temp into 50% val, 50% test (15% and 15% of total)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['street_id']))
    
    val_files = set(temp_df.iloc[val_idx]['file_name'])
    test_files = set(temp_df.iloc[test_idx]['file_name'])
    
    print(f"[*] Group Split completed:")
    print(f"    - Train images: {len(train_files)}")
    print(f"    - Val images:   {len(val_files)}")
    print(f"    - Test images:  {len(test_files)}")
    
    # Map category_id (1..7) -> YOLO class index (0..6)
    # 1: Cấm ngược chiều -> 0
    # 2: Cấm dừng và đỗ   -> 1
    # 3: Cấm rẽ           -> 2
    # 4: Giới hạn tốc độ  -> 3
    # 5: Cấm còn lại      -> 4
    # 6: Nguy hiểm        -> 5
    # 7: Hiệu lệnh        -> 6
    cat_to_yolo = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    
    # Group annotations by file_name
    grouped = df.groupby('file_name')
    
    split_counts = {"train": 0, "val": 0, "test": 0}
    obj_counts = {"train": 0, "val": 0, "test": 0}
    
    # Process each image
    for f_name, rows in grouped:
        if f_name in train_files:
            split = "train"
        elif f_name in val_files:
            split = "val"
        else:
            split = "test"
            
        src_img = os.path.join(img_dir, f_name)
        if not os.path.exists(src_img):
            continue
            
        dst_img = os.path.join(out_base, split, "images", f_name)
        base_no_ext = os.path.splitext(f_name)[0]
        dst_lbl = os.path.join(out_base, split, "labels", f"{base_no_ext}.txt")
        
        # Get image dimension
        img_w = float(rows.iloc[0]['width'])
        img_h = float(rows.iloc[0]['height'])
        
        yolo_lines = []
        for _, r in rows.iterrows():
            cat_id = int(r['category_id'])
            yolo_cls = cat_to_yolo.get(cat_id, 0)
            
            bbox_str = str(r['bbox']).strip("[]").split(",")
            xmin = float(bbox_str[0].strip())
            ymin = float(bbox_str[1].strip())
            bw = float(bbox_str[2].strip())
            bh = float(bbox_str[3].strip())
            
            if bw <= 0 or bh <= 0:
                continue
                
            # Convert to center coords & normalize to [0, 1]
            xc = (xmin + bw / 2.0) / img_w
            yc = (ymin + bh / 2.0) / img_h
            nw = bw / img_w
            nh = bh / img_h
            
            # Clip bounds to [0, 1]
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 0.0001), 1.0)
            nh = min(max(nh, 0.0001), 1.0)
            
            yolo_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            obj_counts[split] += 1
            
        # Copy image and write label
        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            
        split_counts[split] += 1
        
    print(f"[+] Traffic Signs processing complete:")
    for sp in ["train", "val", "test"]:
        print(f"    - {sp}: {split_counts[sp]} images, {obj_counts[sp]} labels")
        
    # Write data.yaml
    yaml_content = f"""path: d:/HocTap/Bien_bao/Data/Processed/traffic_signs_yolo
train: train/images
val: val/images
test: test/images

nc: 7
names:
  0: Cam_nguoc_chieu
  1: Cam_dung_va_do
  2: Cam_re
  3: Gioi_han_toc_do
  4: Cam_con_lai
  5: Nguy_hiem
  6: Hieu_lenh
"""
    yaml_path = os.path.join(out_base, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[✓] Created YAML config at: {yaml_path}")

if __name__ == "__main__":
    convert_zalo_to_yolo()
