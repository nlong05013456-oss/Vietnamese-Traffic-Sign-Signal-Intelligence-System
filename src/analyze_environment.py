import os
import glob
from PIL import Image, ImageStat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

def analyze_environmental_diversity():
    print("="*65)
    print("🌤️ RUNNING ENVIRONMENTAL DIVERSITY & ILLUMINATION ANALYSIS")
    print("="*65)
    
    processed_dir = r"d:\HocTap\Bien_bao\Data\Processed"
    reports_dir = r"d:\HocTap\Bien_bao\Data\Reports\eda_charts"
    os.makedirs(reports_dir, exist_ok=True)
    
    sign_imgs = glob.glob(os.path.join(processed_dir, "traffic_signs_yolo", "**", "images", "*.*"), recursive=True)
    light_imgs = glob.glob(os.path.join(processed_dir, "traffic_lights_yolo", "**", "images", "*.*"), recursive=True)
    
    def get_fast_stats(img_list, label_type, sample_limit=300):
        records = []
        for p in img_list[:sample_limit]:
            try:
                with Image.open(p) as im:
                    # Convert to grayscale thumbnail for instant intensity calculation
                    gray = im.convert('L')
                    stat = ImageStat.Stat(gray)
                    mean_val = stat.mean[0]
                    std_val = stat.stddev[0]
                    
                    if mean_val < 65:
                        env_tag = "Night / Low-Light"
                    elif mean_val > 165:
                        env_tag = "Bright / Sun Glare"
                    else:
                        env_tag = "Daytime / Normal"
                        
                    records.append({
                        "dataset": label_type,
                        "brightness": mean_val,
                        "contrast": std_val,
                        "environment": env_tag
                    })
            except Exception:
                continue
        return pd.DataFrame(records)
        
    df_signs_env = get_fast_stats(sign_imgs, "Traffic Signs", 300)
    df_lights_env = get_fast_stats(light_imgs, "Traffic Lights", 300)
    df_env = pd.concat([df_signs_env, df_lights_env], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    
    sns.kdeplot(data=df_env, x="brightness", hue="dataset", fill=True, common_norm=False, palette=["#3498db", "#e74c3c"], ax=axes[0])
    axes[0].axvline(65, color='black', linestyle='--', alpha=0.7, label='Low-light (<65)')
    axes[0].axvline(165, color='orange', linestyle='--', alpha=0.7, label='High-glare (>165)')
    axes[0].set_title("Illumination (Brightness) Distribution across Datasets", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Mean Pixel Intensity (0 = Dark, 255 = Bright)", fontsize=10)
    axes[0].legend()
    
    sns.countplot(data=df_env, x="environment", hue="dataset", palette=["#3498db", "#e74c3c"], ax=axes[1])
    axes[1].set_title("Environmental Diversity Breakdown", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Lighting Condition", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    
    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "05_environmental_diversity.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    
    print(f"[✓] Environmental Diversity chart saved to: {chart_path}")
    for ds in ["Traffic Signs", "Traffic Lights"]:
        sub = df_env[df_env["dataset"] == ds]
        print(f"\n[+] {ds} Environment Breakdown:")
        for env, cnt in sub["environment"].value_counts().items():
            print(f"    - {env}: {cnt} images ({cnt/len(sub)*100:.1f}%)")

if __name__ == "__main__":
    analyze_environmental_diversity()
