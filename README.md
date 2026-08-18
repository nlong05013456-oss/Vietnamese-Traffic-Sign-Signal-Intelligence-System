# 🚦 Vietnamese Traffic Intelligence System (ADAS & MLOps)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Computer Vision and MLOps system tailored for **Vietnamese Road Infrastructure**. The system detects traffic signs, recognizes traffic light states (Red/Green), performs digital 7-segment countdown timer classification ($00 \to 99$), tracks temporal transitions over video streams, and serves real-time predictions via a **FastAPI** REST service containerized with **Docker**.

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
            │   Temporal Tracker     │  <- (ByteTrack / State Machine)
            │   & State Smoothing    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │   Driver HUD Overlay   │  <- (Real-time Video Display)
            │   & Structured JSON    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │    FastAPI + Docker    │  <- (MLOps Production Serving)
            └────────────────────────┘
```

---

## 💡 Engineering Philosophy & Key Design Decisions

Every architectural decision is evidence-driven:
$$\text{Objective} \longrightarrow \text{Requirement} \longrightarrow \text{EDA Evidence} \longrightarrow \text{Model Selection} \longrightarrow \text{Optimization} \longrightarrow \text{Deployment}$$

### 1. Why 3 Independent Modules Instead of 1 Unified Model?
* **Traffic Signs (Detection):** 7 distinct classes, high spatial variance, small-scale object detection ($<32\text{px}$).
* **Traffic Lights (Detection & State):** 2 states (`Red`, `Green`), vertical rectangular geometry ($1:3$).
* **Countdown Digits (Classification):** 10 classes ($0 \to 9$), localized ROI, digital LED segments.
* *Benefit:* Independent loss functions, task-specific metrics, modular debugging, decoupled deployments.

### 2. Why YOLO11 for Detection?
* Real-time inference requirements ($\ge 30\text{ FPS}$) on edge devices/CPU dashcams.
* Task-Aligned Assigner and Anchor-Free regression head handle high scale variance without manual anchor clustering.

### 3. Data-Centric AI & Anti-Leakage
* **Group Split by `street_id`:** Frame sequences from the same recording route are strictly kept within a single split ($70\%$ Train, $15\%$ Val, $15\%$ Test) to eliminate temporal data leakage.
* **Small Object Dominance:** EDA proved **$76.05\%$ of traffic signs are Small Objects ($<32\text{px}$)** with a median size of **$17.0\text{px}$**, requiring tailored feature pyramid tuning.

---

## 📊 Exploratory Data Analysis (EDA) Summary

| Dataset | Total Samples | Key Findings | Resolution & Split |
| :--- | :---: | :--- | :--- |
| **🪧 Traffic Signs** | $11,000$ bboxes | • Small (<32px): **76.05%**<br>• Medium (32-96px): **21.29%**<br>• Large (>96px): **2.66%** | $1280 \times 720$<br>Group Split by `street_id` |
| **🚦 Traffic Lights** | $4,883$ bboxes | • Red: **53.49%** ($2,612$ obj)<br>• Green: **46.51%** ($2,271$ obj)<br>• Balanced ratio: $1.15 : 1$ | $640 \times 640$<br>Train: $2,513$ \| Val: $103$ \| Test: $50$ |
| **⏱️ Countdown Digits** | $2,500$ images | • 10-Class balanced matrix ($250$ imgs/digit)<br>• Verified 7-segment LED layouts | $96 \times 64$<br>Train: $1,750$ \| Val: $370$ \| Test: $380$ |

---

## 📂 Project Structure

```text
├── configs/
│   ├── traffic_signs_yolo11.yaml     <- Sign detector hyperparameters
│   └── traffic_lights_yolo11.yaml    <- Traffic light detector config
├── Data/
│   ├── Processed/                    <- Frozen, standardized YOLO datasets
│   └── Reports/                      <- Data Audit & EDA Charts
│       ├── eda_charts/               <- High-res statistical plots
│       └── visual_samples/           <- Annotated visual verification samples
├── notebooks/
│   ├── 01_data_eda_and_validation.ipynb
│   └── 02_train_and_evaluate_models.ipynb
├── runs/
│   └── classify/countdown_digits_baseline/
│       └── best_digit_classifier.pth <- Trained PyTorch CNN weights (90.0% Acc)
├── src/
│   ├── app.py                        <- FastAPI Production REST Service
│   ├── data_audit.py                 <- Automated Data-Centric audit engine
│   ├── eda_analysis.py               <- Statistical EDA & plotting script
│   ├── pipeline.py                   <- Multi-task End-to-End inference engine
│   ├── temporal_tracker.py           <- Temporal smoothing & countdown state tracker
│   ├── train_countdown_digits.py     <- PyTorch digit classifier training
│   ├── train_traffic_lights.py       <- YOLO11n traffic lights trainer
│   └── train_traffic_signs.py        <- YOLO11n traffic signs trainer
├── Dockerfile                        <- Containerization specification
├── requirements.txt                  <- Production dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/vietnamese-traffic-intelligence.git
cd vietnamese-traffic-intelligence

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Audit & EDA
```bash
# Execute automated Data Audit
python src/data_audit.py

# Generate EDA Statistical Visualizations
python src/eda_analysis.py
```

### 3. Train Models
```bash
# Train Countdown Digit Classifier
python src/train_countdown_digits.py

# Train YOLO11 Traffic Signs Detector
python src/train_traffic_signs.py --epochs 50 --batch 16 --imgsz 640

# Train YOLO11 Traffic Lights Detector
python src/train_traffic_lights.py --epochs 50 --batch 16 --imgsz 640
```

### 4. Run Real-time Inference Pipeline
```bash
python src/pipeline.py
```

### 5. Launch FastAPI MLOps Service
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive Swagger UI at: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

Build and run the entire system in an isolated container:

```bash
# Build the Docker image
docker build -t vietnamese-traffic-intelligence:latest .

# Run the container
docker run -p 8000:8000 vietnamese-traffic-intelligence:latest
```

---

## 💬 Machine Learning Engineering Interview Q&A

**Q: Why separate the system into 3 distinct models rather than one multi-head network?**  
> *A:* Multi-task networks often suffer from gradient conflict (destructive interference) when balancing bounding-box regression losses with fine-grained classification losses across radically different spatial resolutions. Decoupling into 3 modular components allows independent metric optimization, separate inference scaling, and isolated debugging.

**Q: How do you handle Data Leakage in sequential traffic data?**  
> *A:* Consecutive dashcam frames from the same vehicle run share identical backgrounds. Random splitting creates near-duplicate leakage between Train and Test, inflating test metrics. We implemented a **Group Split by `street_id`**, ensuring that entire road segments are strictly assigned to either Train, Val, or Test.

**Q: What is the limitation regarding Traffic Light timing?**  
> *A:* When a camera observes a solid Red light without a visible digital countdown display, the system cannot output a ground-truth remaining time. We explicitly treat this as an **Estimated Remaining Time** derived from historical cycle observations rather than an exact reading.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
