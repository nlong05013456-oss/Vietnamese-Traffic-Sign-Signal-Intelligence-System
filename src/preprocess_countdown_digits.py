import os
import glob
import shutil
import random

def split_countdown_digits():
    print("\n" + "="*60)
    print("[3] PREPROCESSING: COUNTDOWN TIMER DIGITS")
    print("="*60)
    
    raw_dir = r"d:\HocTap\Bien_bao\Data\Raw\countdown_timer_digits"
    out_base = r"d:\HocTap\Bien_bao\Data\Processed\countdown_digits_cls"
    
    random.seed(42)
    
    total_copied = 0
    for d in range(10):
        digit_str = str(d)
        src_digit_dir = os.path.join(raw_dir, digit_str)
        if not os.path.exists(src_digit_dir):
            continue
            
        imgs = glob.glob(os.path.join(src_digit_dir, "*.png"))
        random.shuffle(imgs)
        
        n_total = len(imgs)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]
        }
        
        for sp, file_list in splits.items():
            dst_dir = os.path.join(out_base, sp, digit_str)
            os.makedirs(dst_dir, exist_ok=True)
            for src_p in file_list:
                dst_p = os.path.join(dst_dir, os.path.basename(src_p))
                shutil.copy2(src_p, dst_p)
                total_copied += 1
                
        print(f"  - Digit '{digit_str}': {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
    print(f"[✓] Successfully split {total_copied} digit images into {out_base}!")

if __name__ == "__main__":
    split_countdown_digits()
