# 🚦 Vietnamese Traffic Intelligence System (ADAS & MLOps)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Computer Vision + MLOps system for **Vietnamese Road Infrastructure**: traffic sign detection, traffic light state recognition (Red/Green), 7-segment countdown digit reading, temporal tracking across video, and REST serving via **FastAPI** + **Docker**.

> **⚠️ Project status:** actively in development. Baseline models are trained and evaluated; results below are honest, current numbers — not final production metrics. See [Current Progress](#-current-progress) for what's done vs. in progress.

---

## 📸 System Architecture

```
                       DASHCAM / VIDEO STREAM
                                │
                                ▼
                      ┌──────────────────┐
                      │   Frame Input     │
                      └────────┬─────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       🪧 Sign             🚦 Light           ⏱️ Countdown
       Detector            Detector           Classifier
       (YOLO11n)           (YOLO11n)          (PyTorch CNN)
            │                  │                  │
            ▼                  ▼                  ▼
       7 VN Classes        RED / GREEN         0 to 9 Digits
      (P.127, W.205...)   (Task-Aligned)      (Left-Right Merge)
            │                  │                  │
            └───────────┬──────┴──────────────────┘
                        ▼
            ┌────────────────────────┐
            │   Temporal Tracker     │  <- (State Smoothing)
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │   Driver HUD Overlay   │
            │   & Structured JSON    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │    FastAPI + Docker    │
            └────────────────────────┘
```

---

## ✅ Current Progress

| Component | Status | Notes |
|---|---|---|
| Data audit & cleaning | ✅ Done | 10,252 files scanned, 0 corrupt |
| Group split (anti-leakage by `street_id`) | ✅ Done | 70/15/15 train/val/test |
| EDA (class distribution, small-object analysis, environmental diversity) | ✅ Done | See `Data/Reports/eda_charts/` |
| Countdown digit CNN | ✅ Trained | 90.0% accuracy on 380 held-out test images |
| Digit localization module (`digit_localizer.py`) | ✅ Built | Otsu thresholding + left-to-right sort; **not yet benchmarked end-to-end with the classifier** |
| Traffic sign detector (YOLO11n) | 🟡 Baseline trained, needs improvement | See metrics below — recall is low, next step is higher input resolution |
| Traffic light detector (YOLO11n) | ⬜ Not yet trained | Script ready (`train_traffic_lights.py`), no results yet |
| Temporal tracker / smoothing | 🟡 Built, not validated on real video | Logic implemented, needs testing against a real detector output stream |
| FastAPI service | 🟡 Scaffolded | Endpoints defined; **not yet serving a fully trained sign+light model** |
| Docker | 🟡 Scaffolded | Builds, not yet validated with final model weights |
| AWS deployment | ⬜ Not started | Planned as a short, one-time demo deploy — not a persistent service |

---

## 📊 Baseline Results (real, measured — not targets)

### Traffic Sign Detector — YOLO11n, 640px, 50 epochs (GPU)

Best epoch so far (peaked around epoch 26 of 50):

| Metric | Value |
|---|---|
| Precision | 0.686 |
| Recall | 0.384 |
| mAP50 | 0.432 |
| mAP50-95 | 0.253 |

**Honest read:** precision is decent (when the model says "sign," it's often right), but recall is low — it misses a majority of signs present. This tracks directly with the EDA finding that **76.05% of signs are small objects (<32px)**, which is hard for a 640px-input nano model to pick up. Next planned step: re-run at `imgsz=832` and compare small-object recall before considering this a finished baseline.

Per-class breakdown showed the weakest class was `Cam_nguoc_chieu` (no-entry sign), with recall well below the others — flagged for further error analysis, not yet root-caused.

### Countdown Digit Classifier — PyTorch CNN

| Metric | Value |
|---|---|
| Test Accuracy | 90.0% (380 test images) |

**Caveat:** this accuracy is measured on cleanly cropped digits. End-to-end accuracy (through `digit_localizer.py`'s automatic cropping) has not yet been separately measured and may be lower.

### Traffic Light Detector

Not yet trained — no metrics to report.

---

## 📊 Exploratory Data Analysis (EDA) Summary

| Dataset | Total Samples | Key Findings | Split |
|---|---|---|---|
| 🪧 Traffic Signs | 11,000 bboxes | Small (<32px): 76.05% · Medium (32-96px): 21.29% · Large (>96px): 2.66% · Median size: 17.0px | Train 2,636 / Val 734 / Test 1,130 imgs, group-split by `street_id` |
| 🚦 Traffic Lights | 4,883 bboxes | Red: 53.49% (2,612) · Green: 46.51% (2,271) · Ratio 1.15:1 | Train 2,513 / Val 103 / Test 50 imgs |
| ⏱️ Countdown Digits | 2,500 images | 10-class balanced (250 imgs/digit) | Train 1,750 / Val 370 / Test 380 |

Known data quality note: several dozen images had duplicate bounding-box labels removed automatically during training (visible in training logs) — flagged for cleanup in the source annotation pipeline, not yet fixed at the root.

---

## 💡 Engineering Philosophy

Every architectural decision follows this chain, and is only advanced to the next stage once there's evidence to support it:

```
Objective → Requirement → EDA Evidence → Model Selection → Baseline → Evaluation → Error Analysis → Improvement → Deployment
```

### Why 3 independent modules instead of 1 unified model?
Traffic signs (7-class detection, small objects), traffic lights (2-state detection, different geometry), and countdown digits (10-class classification on cropped ROIs) have different output types, losses, and metrics. Keeping them separate makes debugging, evaluation, and independent iteration much easier — a unified multi-head model is only worth considering later, and only if benchmarks show a real benefit.

### Why YOLO11n as the detection baseline?
Real-time inference and deployment (FastAPI/Docker) were requirements from the start, so a one-stage, anchor-free detector made sense over something like Faster R-CNN. Starting with the nano variant (not a larger one) gives a fast baseline to find out how far the current dataset can go before committing more compute.

### Data-centric decisions
- **Group split by `street_id`**: frames from the same recording route are never split across train/val/test, avoiding leakage from near-duplicate consecutive frames.
- **Small-object dominance (76.05%)** directly informed the decision to test higher input resolution (832px) rather than assuming the baseline resolution would be sufficient.

---

## 📂 Project Structure

```text
├── configs/
│   ├── traffic_signs_yolo11.yaml
│   └── traffic_lights_yolo11.yaml
├── Data/
│   ├── Processed/
│   └── Reports/
│       ├── eda_charts/
│       └── visual_samples/
├── notebooks/
│   ├── 01_data_eda_and_validation.ipynb
│   └── 02_train_and_evaluate_models.ipynb
├── runs/
│   ├── classify/countdown_digits_baseline/
│   │   └── best_digit_classifier.pth      <- 90.0% acc
│   └── detect/traffic_signs_yolo11n_gpu/
│       └── weights/best.pt                <- mAP50 0.432 (baseline, in progress)
├── src/
│   ├── app.py                             <- FastAPI service (scaffolded)
│   ├── data_audit.py
│   ├── eda_analysis.py
│   ├── pipeline.py
│   ├── temporal_tracker.py
│   ├── digit_localizer.py
│   ├── train_countdown_digits.py
│   ├── train_traffic_lights.py            <- not yet run
│   └── train_traffic_signs.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Setup
```bash
git clone https://github.com/<your-username>/vietnamese-traffic-intelligence.git
cd vietnamese-traffic-intelligence
pip install -r requirements.txt
```

### 2. Data audit & EDA
```bash
python src/data_audit.py
python src/eda_analysis.py
```

### 3. Train models
```bash
python src/train_countdown_digits.py

# Sign detector — baseline currently at mAP50 0.432, see notes above
python src/train_traffic_signs.py --epochs 50 --batch 16 --imgsz 640

# Light detector — not yet run
python src/train_traffic_lights.py --epochs 50 --batch 16 --imgsz 640
```

### 4. Run inference pipeline
```bash
python src/pipeline.py
```

### 5. Launch FastAPI service
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI: `http://localhost:8000/docs`

---

## 🐳 Docker

```bash
docker build -t vietnamese-traffic-intelligence:latest .
docker run -p 8000:8000 vietnamese-traffic-intelligence:latest
```

---

## 🗺️ Next Steps

- [ ] Re-train sign detector at `imgsz=832`, compare small-object recall against the 640px baseline
- [ ] Investigate low recall on `Cam_nguoc_chieu` class specifically
- [ ] Clean up duplicate bounding-box labels at the source
- [ ] Train traffic light detector, get first mAP numbers
- [ ] Measure end-to-end digit accuracy (localizer + classifier together, not classifier alone)
- [ ] Validate temporal tracker against real video, not just unit logic
- [ ] Wire FastAPI to the actual trained weights and test `/predict` end-to-end
- [ ] One-time AWS demo deploy (with billing alerts set), or HuggingFace Spaces as a lower-risk alternative

---

## 📜 License
MIT License.
