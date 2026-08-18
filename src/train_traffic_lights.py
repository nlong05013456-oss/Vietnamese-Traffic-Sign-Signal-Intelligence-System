import os
import argparse
from ultralytics import YOLO
import torch

def train_traffic_lights(epochs=50, imgsz=640, batch=16, device='cpu', workers=4):
    print("="*65)
    print("🚀 PHASE 3: TRAINING TRAFFIC LIGHTS BASELINE (YOLO11n)")
    print("="*65)
    
    data_yaml = r"d:\HocTap\Bien_bao\Data\Processed\traffic_lights_yolo\data.yaml"
    project_dir = r"d:\HocTap\Bien_bao\runs\detect"
    exp_name = "traffic_lights_yolo11n_baseline"
    
    if torch.cuda.is_available() and device == 'auto':
        device_to_use = 0
        print(f"[*] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device_to_use = device if device != 'auto' else 'cpu'
        print(f"[*] Training on device: {device_to_use}")

    print("[*] Initializing YOLO11n pretrained model...")
    model = YOLO("yolo11n.pt")
    
    print(f"[*] Starting training: epochs={epochs}, imgsz={imgsz}, batch={batch}...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device_to_use,
        workers=workers,
        project=project_dir,
        name=exp_name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        plots=True,
        save=True
    )
    
    print("\n[✓] Baseline Traffic Lights Training Completed!")
    print(f"    - Best Model Weights: {os.path.join(project_dir, exp_name, 'weights', 'best.pt')}")
    print(f"    - Results & Curves:   {os.path.join(project_dir, exp_name)}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11n Traffic Lights Baseline")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', '0', etc.")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()
    
    train_traffic_lights(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers)
