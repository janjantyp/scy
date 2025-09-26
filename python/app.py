# app.py
from fastapi import Body
import requests
from ultralytics import YOLO
import io
import base64
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
from PIL import Image
import cv2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- use "*" only in dev for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"msg": "api is working!"}


# Ultralytics YOLO

# Will be set on startup
MODEL: Optional[YOLO] = None


class Box(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: str


class PredictionResponse(BaseModel):
    boxes: List[Box]
    # cannotated_image_base64: Optional[str] = None  # data:image/png;base64,...
    width: int
    height: int


@app.on_event("startup")
def load_model():
    global MODEL
    # Change to your model path if needed (e.g. "runs/train/weights/best.pt" or "yolov8n.pt")
    MODEL = YOLO("best_mAP50.pt")
    # Optionally run a dry run to warm up (small image)
    # MODEL.predict(source=np.zeros((640,640,3), dtype=np.uint8), imgsz=640, conf=0.25)


def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return np.array(image)


def draw_boxes(image: np.ndarray, boxes: List[Dict[str, Any]], names: Dict[int, str]) -> np.ndarray:
    img = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, [b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
        conf = b["confidence"]
        cls = b["class_id"]
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        # Draw rectangle and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # text background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


@app.post("/usc_api/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, description="Confidence threshold (0-1)"),
    iou: float = Query(0.45, description="NMS IoU threshold (0-1)"),
    imgsz: int = Query(
        640, description="Resize short side to this (recommended 320/640/1280)")
):
    """
    Upload an image file (multipart/form-data) and get YOLO predictions.
    Returns bounding boxes and an annotated image (base64 PNG).
    """
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    content = await file.read()
    try:
        img = read_imagefile(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run prediction
    # Ultralytics API: results = MODEL.predict(source=..., imgsz=..., conf=..., iou=...)
    results = MODEL.predict(source=img, imgsz=imgsz,
                            conf=conf, iou=iou, verbose=False)

    if len(results) == 0:
        # No predictions (rare)
        boxes = []
    else:
        res = results[0]  # single image
        # res.boxes: contains xyxy, conf, cls
        boxes = []
        # res.boxes.xyxy is a tensor-like; convert to numpy
        if hasattr(res, "boxes") and res.boxes is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()  # shape (n,4)
            confs = res.boxes.conf.cpu().numpy()  # shape (n,)
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # shape (n,)
            names = MODEL.model.names if hasattr(
                MODEL, "model") and hasattr(MODEL.model, "names") else {}
            for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, cls_ids):
                boxes.append({
                    "xmin": float(x1),
                    "ymin": float(y1),
                    "xmax": float(x2),
                    "ymax": float(y2),
                    "confidence": float(c),
                    "class_id": int(cls),
                    "class_name": names.get(cls, str(cls))
                })

    # Annotated image
    annotated_np = draw_boxes(img, boxes, MODEL.model.names if hasattr(
        MODEL, "model") and hasattr(MODEL.model, "names") else {})
    # Convert BGR/RGB: our img is RGB from PIL -> np array; cv2 uses


@app.post("/usc_api/predict_url", response_model=PredictionResponse)
async def predict_image_from_url(
    url: str = Body(..., embed=True),
    conf: float = Query(0.25),
    iou: float = Query(0.45),
    imgsz: int = Query(640)
):
    """
    Provide an image URL (e.g. Google Street View) and get YOLO predictions.
    """
    print("Received URL:", url)
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not download image: {e}")

    try:
        img = read_imagefile(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    results = MODEL.predict(source=img, imgsz=imgsz,
                            conf=conf, iou=iou, verbose=False)

    boxes = []
    if len(results):
        res = results[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        names = MODEL.model.names if hasattr(
            MODEL, "model") and hasattr(MODEL.model, "names") else {}
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, cls_ids):
            boxes.append({
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
                "confidence": float(c),
                "class_id": int(cls),
                "class_name": names.get(cls, str(cls))
            })

    annotated_np = draw_boxes(
        img, boxes, MODEL.model.names if hasattr(MODEL, "model") else {})
    _, buffer = cv2.imencode(".png", annotated_np[:, :, ::-1])
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    height, width = img.shape[:2]
    return PredictionResponse(boxes=boxes, annotated_image_base64=data_uri, width=width, height=height)
