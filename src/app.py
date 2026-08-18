import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import io
from PIL import Image
from src.pipeline import TrafficIntelligencePipeline

# Initialize FastAPI app
app = FastAPI(
    title="🚦 Vietnamese Traffic Intelligence API",
    description="Real-time Computer Vision Service for Traffic Signs, Traffic Lights, and Countdown Timers (ADAS System).",
    version="1.0.0"
)

# Initialize End-to-End Pipeline
pipeline = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    print("[*] Loading Traffic Intelligence Pipeline models...")
    pipeline = TrafficIntelligencePipeline(
        digit_weights_path=r"d:\HocTap\Bien_bao\runs\classify\countdown_digits_baseline\best_digit_classifier.pth"
    )
    print("[✓] Pipeline loaded and ready for inference!")

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "Vietnamese Traffic Intelligence Service",
        "version": "1.0.0",
        "models": {
            "traffic_signs_detector": "YOLO11n (7 Classes)",
            "traffic_lights_detector": "YOLO11n (Red / Green)",
            "countdown_digit_classifier": "PyTorch CNN (10 Classes)"
        }
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs the full CV + Temporal pipeline,
    and returns structured traffic intelligence JSON.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)[:, :, ::-1] # RGB to BGR for OpenCV
        
        annotated_frame, result_data = pipeline.process_frame(frame)
        return JSONResponse(content=result_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/visualize")
async def predict_image_visualize(file: UploadFile = File(...)):
    """
    Runs inference and returns the annotated image with BBoxes & Driver HUD overlay.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)[:, :, ::-1]
        
        annotated_frame, _ = pipeline.process_frame(frame)
        
        # Encode to JPEG
        _, encoded_img = cv2.imencode(".jpg", annotated_frame)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
