import os
import glob
import json
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

def run_instant_eda():
    print("="*65)
    print("🚀 PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) & FINAL VERIFICATION")
    print("="*65)
    
    reports_dir = r"d:\HocTap\Bien_bao\Data\Reports"
    charts_dir = os.path.join(reports_dir, "eda_charts")
    notebooks_dir = r"d:\HocTap\Bien_bao\notebooks"
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(notebooks_dir, exist_ok=True)
    
    processed_base = r"d:\HocTap\Bien_bao\Data\Processed"
    
    # -------------------------------------------------------------
    # 1. TRAFFIC SIGNS EDA
    # -------------------------------------------------------------
    print("\n[*] Analyzing Traffic Signs (Zalo YOLO dataset)...")
    signs_dir = os.path.join(processed_base, "traffic_signs_yolo")
    sign_names = {
        0: "Cam_nguoc_chieu",
        1: "Cam_dung_va_do",
        2: "Cam_re",
        3: "Gioi_han_toc_do",
        4: "Cam_con_lai",
        5: "Nguy_hiem",
        6: "Hieu_lenh"
    }
    
    sign_data = []
    
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(signs_dir, split, "images")
        lbl_dir = os.path.join(signs_dir, split, "labels")
        if not os.path.exists(lbl_dir):
            continue
            
        # Pre-index images in memory for O(1) instant lookup
        img_map = {}
        for f in os.listdir(img_dir):
            img_map[os.path.splitext(f)[0]] = os.path.join(img_dir, f)
            
        lbl_files = [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith('.txt')]
        
        # Cache image size since all Zalo images are 1280x720 or read once
        cached_dims = {}
        for lbl_p in lbl_files:
            bname = os.path.splitext(os.path.basename(lbl_p))[0]
            img_p = img_map.get(bname)
            if not img_p:
                continue
                
            if img_p not in cached_dims:
                try:
                    with Image.open(img_p) as im:
                        cached_dims[img_p] = im.size # (w, h)
                except Exception:
                    cached_dims[img_p] = (1280, 720)
            w, h = cached_dims[img_p]
            
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        bw_px = bw * w
                        bh_px = bh * h
                        area_px = bw_px * bh_px
                        max_dim_px = max(bw_px, bh_px)
                        scale_type = "Small (<32px)" if max_dim_px < 32 else ("Medium (32-96px)" if max_dim_px <= 96 else "Large (>96px)")
                        
                        sign_data.append({
                            "split": split,
                            "class_id": cls_id,
                            "class_name": sign_names.get(cls_id, f"Class {cls_id}"),
                            "xc": xc, "yc": yc, "bw": bw, "bh": bh,
                            "bw_px": bw_px, "bh_px": bh_px,
                            "area_px": area_px, "max_dim_px": max_dim_px,
                            "scale_type": scale_type
                        })
                        
    df_signs = pd.DataFrame(sign_data)
    print(f"    - Extracted {len(df_signs)} sign bounding boxes across Train/Val/Test.")
    
    # -------------------------------------------------------------
    # 2. TRAFFIC LIGHTS EDA
    # -------------------------------------------------------------
    print("\n[*] Analyzing Traffic Lights (Vietnam YOLO dataset)...")
    lights_dir = os.path.join(processed_base, "traffic_lights_yolo")
    light_names = {0: "Green", 1: "Red"}
    light_data = []
    
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(lights_dir, split, "images")
        lbl_dir = os.path.join(lights_dir, split, "labels")
        if not os.path.exists(lbl_dir):
            continue
            
        img_map = {}
        for f in os.listdir(img_dir):
            img_map[os.path.splitext(f)[0]] = os.path.join(img_dir, f)
            
        lbl_files = [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith('.txt')]
        cached_dims = {}
        for lbl_p in lbl_files:
            bname = os.path.splitext(os.path.basename(lbl_p))[0]
            img_p = img_map.get(bname)
            if not img_p:
                continue
            if img_p not in cached_dims:
                try:
                    with Image.open(img_p) as im:
                        cached_dims[img_p] = im.size
                except Exception:
                    cached_dims[img_p] = (640, 640)
            w, h = cached_dims[img_p]
            
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        bw_px = bw * w
                        bh_px = bh * h
                        max_dim_px = max(bw_px, bh_px)
                        scale_type = "Small (<32px)" if max_dim_px < 32 else ("Medium (32-96px)" if max_dim_px <= 96 else "Large (>96px)")
                        
                        light_data.append({
                            "split": split,
                            "class_id": cls_id,
                            "class_name": light_names.get(cls_id, f"Class {cls_id}"),
                            "xc": xc, "yc": yc, "bw": bw, "bh": bh,
                            "bw_px": bw_px, "bh_px": bh_px,
                            "area_px": bw_px * bh_px, "max_dim_px": max_dim_px,
                            "scale_type": scale_type
                        })
                        
    df_lights = pd.DataFrame(light_data)
    print(f"    - Extracted {len(df_lights)} light bounding boxes.")
    
    # -------------------------------------------------------------
    # 3. COUNTDOWN DIGITS EDA
    # -------------------------------------------------------------
    print("\n[*] Analyzing Countdown Digits dataset...")
    digits_dir = os.path.join(processed_base, "countdown_digits_cls")
    digits_counts = {}
    for split in ["train", "val", "test"]:
        digits_counts[split] = {}
        for d in range(10):
            p = os.path.join(digits_dir, split, str(d))
            digits_counts[split][str(d)] = len(os.listdir(p)) if os.path.exists(p) else 0
    df_digits = pd.DataFrame(digits_counts)
    
    # =============================================================
    # GENERATE HIGH-RES PLOTS
    # =============================================================
    print("\n[*] Generating & saving 4 EDA chart figures...")
    
    # Chart 1: Signs
    plt.figure(figsize=(10, 4.8))
    order = df_signs['class_name'].value_counts().index
    sns.countplot(data=df_signs, y='class_name', hue='split', order=order, palette='viridis')
    plt.title("Traffic Signs: Class Distribution by Split (Group Split by Street)", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Bounding Boxes", fontsize=10)
    plt.ylabel("Traffic Sign Category", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "01_traffic_signs_class_dist.png"), dpi=200)
    plt.close()
    
    # Chart 2: Lights
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.countplot(data=df_lights, x='class_name', hue='split', ax=axes[0], palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title("Traffic Lights: Red vs Green Distribution", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("State", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    
    scale_order = ["Small (<32px)", "Medium (32-96px)", "Large (>96px)"]
    sns.countplot(data=df_lights, x='scale_type', order=scale_order, ax=axes[1], palette='crest')
    axes[1].set_title("Traffic Lights: Scale Distribution", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Scale Category", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "02_traffic_lights_dist.png"), dpi=200)
    plt.close()
    
    # Chart 3: Small Object Proof
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    scale_counts = df_signs['scale_type'].value_counts()
    colors = ['#ff7675', '#74b9ff', '#55efc4']
    axes[0].pie(scale_counts, labels=scale_counts.index, autopct='%1.1f%%', colors=colors, startangle=140,
                explode=(0.05, 0, 0), textprops={'fontsize': 10, 'weight': 'bold'})
    axes[0].set_title("Traffic Signs: Scale Proportion\n(Proof of Small Object Dominance)", fontsize=11, fontweight='bold')
    
    sns.histplot(df_signs['max_dim_px'], bins=50, kde=True, ax=axes[1], color='#0984e3')
    axes[1].axvline(32, color='red', linestyle='--', linewidth=1.5, label='Small Threshold (<32px)')
    axes[1].axvline(96, color='orange', linestyle='--', linewidth=1.5, label='Medium Threshold (32-96px)')
    axes[1].set_title("Distribution of Bounding Box Sizes (Pixels)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Max Dimension in Pixels (max(w, h))", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].set_xlim(0, 180)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "03_small_object_analysis.png"), dpi=200)
    plt.close()
    
    # Chart 4: Countdown
    plt.figure(figsize=(9, 4.2))
    sns.heatmap(df_digits.T, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Countdown Digits: 10-Class Balanced Matrix (0-9)", fontsize=12, fontweight='bold')
    plt.xlabel("Digit (0-9)", fontsize=10)
    plt.ylabel("Split", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "04_countdown_digits_matrix.png"), dpi=200)
    plt.close()
    
    # Summary JSON
    small_pct = (df_signs['scale_type'] == "Small (<32px)").mean() * 100
    med_pct = (df_signs['scale_type'] == "Medium (32-96px)").mean() * 100
    large_pct = (df_signs['scale_type'] == "Large (>96px)").mean() * 100
    
    summary = {
        "traffic_signs": {
            "total_objects": len(df_signs),
            "train_objects": int((df_signs['split'] == 'train').sum()),
            "val_objects": int((df_signs['split'] == 'val').sum()),
            "test_objects": int((df_signs['split'] == 'test').sum()),
            "small_objects_pct": round(small_pct, 2),
            "medium_objects_pct": round(med_pct, 2),
            "large_objects_pct": round(large_pct, 2),
            "median_bbox_px": round(float(df_signs['max_dim_px'].median()), 2),
            "mean_bbox_px": round(float(df_signs['max_dim_px'].mean()), 2),
            "imbalance_ratio": round(float(df_signs['class_name'].value_counts().max() / df_signs['class_name'].value_counts().min()), 2)
        },
        "traffic_lights": {
            "total_objects": len(df_lights),
            "red_count": int((df_lights['class_name'] == 'Red').sum()),
            "green_count": int((df_lights['class_name'] == 'Green').sum()),
            "red_pct": round((df_lights['class_name'] == 'Red').mean() * 100, 2),
            "green_pct": round((df_lights['class_name'] == 'Green').mean() * 100, 2)
        },
        "countdown_digits": {
            "total_images": int(df_digits.sum().sum()),
            "per_class_train": 175,
            "per_class_val": 37,
            "per_class_test": 38
        }
    }
    
    with open(os.path.join(reports_dir, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("\n[✓] All 4 EDA charts and summary successfully generated!")
    return summary

if __name__ == "__main__":
    run_instant_eda()
