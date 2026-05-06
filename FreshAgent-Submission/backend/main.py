"""
main.py – FastAPI backend for FreshAgent Adulteration Detection System.

Endpoints:
  GET  /                        → serves the web dashboard
  POST /api/upload-manual       → manual image upload from browser
  POST /api/esp32-stream        → image POST from ESP32-CAM
  GET  /api/latest-esp32        → returns the last ESP32 result (for polling)
  GET  /api/health              → health check
"""
import os
import io
import sys
import base64
import json
import socket
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ─── Path resolution ───────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"
ONNX_PATH        = MODELS_DIR / "freshagent.onnx"
PTH_PATH         = MODELS_DIR / "best_model.pth"
VEGGIE_ONNX_PATH = MODELS_DIR / "veggieagent.onnx"
VEGGIE_PTH_PATH  = MODELS_DIR / "veggieagent_best.pth"

sys.path.insert(0, str(BASE_DIR / "backend"))
from inference import FreshAgentInference, generate_gradcam
from veggie_inference import VeggieAgentInference, generate_veggie_gradcam

# ─── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FreshAgent – Fruits & Vegetables Adulteration Detection API",
    version="1.0.0",
    description="Detects Fresh / Rotten / Formalin-Mixed conditions in fruits using deep learning.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve static frontend ─────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ─── Global inference engines ─────────────────────────────────────────────────
inference_engine: Optional[FreshAgentInference] = None
veggie_engine: Optional[VeggieAgentInference] = None
latest_esp32_result: dict = {}
latest_phone_result: dict = {}


class PhoneFrame(BaseModel):
    image_b64: str


@app.on_event("startup")
async def startup_event():
    global inference_engine, veggie_engine
    # ─── Fruit model ───────────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        inference_engine = FreshAgentInference(str(ONNX_PATH))
    else:
        print(f"\n[WARNING] Fruit ONNX model not found at {ONNX_PATH}")
        print("  Train: python training/train.py --data_dir <path>")
        print("  Export: python training/export.py --checkpoint models/best_model.pth\n")
    # ─── Veggie model ──────────────────────────────────────────────────────────
    if VEGGIE_ONNX_PATH.exists():
        veggie_engine = VeggieAgentInference(str(VEGGIE_ONNX_PATH))
    else:
        print(f"[INFO] VeggieAgent ONNX not found at {VEGGIE_ONNX_PATH}")
        print("  Train: train_veggie.bat <path_to_veggie_dataset>\n")


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def run_prediction(img: Image.Image, generate_cam: bool = True) -> dict:
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and export the model first."
        )
    result = inference_engine.predict(img)
    result["timestamp"] = datetime.now().isoformat()

    # Grad-CAM (condition class index)
    from inference import COND_CLASSES
    cond_idx = COND_CLASSES.index(result["condition"])
    if generate_cam and PTH_PATH.exists():
        result["gradcam"] = generate_gradcam(str(PTH_PATH), img, target_class_idx=cond_idx)
    else:
        result["gradcam"] = ""

    # Human-readable safety label
    cond = result["condition"]
    if cond == "Fresh":
        result["safety"]       = "SAFE - Fresh fruit. No adulteration detected."
        result["safety_class"] = "safe-class"
    elif cond == "Rotten":
        result["safety"]       = "UNSAFE - Fruit is rotten. Not suitable for consumption."
        result["safety_class"] = "warn-class"
    else:  # Formalin_Mixed
        result["safety"]       = "DANGER - Formalin treatment detected! Do not consume."
        result["safety_class"] = "danger-class"

    return result


def run_veggie_prediction(img: Image.Image, generate_cam: bool = True) -> dict:
    if veggie_engine is None:
        raise HTTPException(
            status_code=503,
            detail="VeggieAgent model not loaded. Please train and export the veggie model first: train_veggie.bat <dataset_path>"
        )
    result = veggie_engine.predict(img)
    result["timestamp"] = datetime.now().isoformat()

    # Grad-CAM
    from veggie_inference import COND_CLASSES
    cond_idx = COND_CLASSES.index(result["condition"])
    if generate_cam and VEGGIE_PTH_PATH.exists():
        result["gradcam"] = generate_veggie_gradcam(str(VEGGIE_PTH_PATH), img, target_class_idx=cond_idx)
    else:
        result["gradcam"] = ""

    # Human-readable safety label
    cond = result["condition"]
    if cond == "Fresh":
        result["safety"]       = "SAFE - Fresh vegetable. No adulteration detected."
        result["safety_class"] = "safe-class"
    elif cond == "Rotten":
        result["safety"]       = "UNSAFE - Vegetable is rotten. Not suitable for consumption."
        result["safety_class"] = "warn-class"
    else:  # Adulterated
        result["safety"]       = "DANGER - Adulteration detected! Do not consume."
        result["safety_class"] = "danger-class"

    return result


def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>FreshAgent API running. Frontend not found.</h1>")


@app.get("/style.css")
async def serve_css():
    f = FRONTEND_DIR / "style.css"
    return FileResponse(str(f), media_type="text/css")


@app.get("/app.js")
async def serve_js():
    f = FRONTEND_DIR / "app.js"
    return FileResponse(str(f), media_type="application/javascript")


@app.get("/phone-cam", response_class=HTMLResponse)
async def serve_phone_cam():
    """Serve the self-contained phone camera page."""
    page = FRONTEND_DIR / "phone_cam.html"
    if page.exists():
        return HTMLResponse(content=page.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Phone camera page not found.</h1>", status_code=404)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": inference_engine is not None,
        "onnx_path": str(ONNX_PATH),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/upload-manual")
async def upload_manual(file: UploadFile = File(...)):
    """
    Accept an image file uploaded manually from the web dashboard.
    Returns prediction results including Grad-CAM heatmap (base64).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    data = await file.read()
    img  = pil_from_bytes(data)
    result = run_prediction(img, generate_cam=True)
    result["source"]   = "manual_upload"
    result["filename"] = file.filename
    return JSONResponse(content=result)


@app.post("/api/esp32-stream")
async def esp32_stream(file: UploadFile = File(None), image_b64: str = Form(None)):
    """
    Accepts image from ESP32-CAM either as:
    - Multipart form file upload, or
    - Base64-encoded string in form field 'image_b64'
    
    The ESP32 does NOT run any model. It only sends raw JPEG bytes.
    """
    global latest_esp32_result

    if file is not None:
        data = await file.read()
    elif image_b64:
        # Strip data URI header if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        data = base64.b64decode(image_b64)
    else:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'image_b64'.")

    img    = pil_from_bytes(data)
    result = run_prediction(img, generate_cam=False)   # skip CAM for speed in streaming
    result["source"] = "esp32_cam"
    latest_esp32_result = result
    return JSONResponse(content=result)


@app.get("/api/latest-esp32")
async def get_latest_esp32():
    """
    Polling endpoint: returns the last result received from the ESP32-CAM.
    Frontend polls this every 2-3 seconds to update the live view.
    """
    if not latest_esp32_result:
        return JSONResponse(content={"status": "no_data", "message": "No ESP32 data received yet."})
    return JSONResponse(content=latest_esp32_result)


@app.post("/api/upload-veggie")
async def upload_veggie(file: UploadFile = File(...)):
    """
    Accept a vegetable image uploaded from the web dashboard.
    Uses VeggieAgent (separate model) — fruit model is not involved at all.
    Returns prediction: veggie name, condition, confidence, Grad-CAM heatmap.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    data = await file.read()
    img  = pil_from_bytes(data)
    result = run_veggie_prediction(img, generate_cam=True)
    result["source"]   = "manual_upload_veggie"
    result["filename"] = file.filename
    return JSONResponse(content=result)


@app.get("/api/veggie-status")
async def veggie_status():
    """Check whether the VeggieAgent model is loaded and ready."""
    return {
        "veggie_model_loaded": veggie_engine is not None,
        "onnx_path": str(VEGGIE_ONNX_PATH),
        "timestamp": datetime.now().isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phone Camera Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/phone-stream")
async def phone_stream(frame: PhoneFrame):
    """
    Accept a frame from the phone camera page.
    The phone sends base64-encoded JPEG via JSON body.
    """
    global latest_phone_result

    b64 = frame.image_b64
    # Strip data URI header if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)

    img    = pil_from_bytes(data)
    result = run_prediction(img, generate_cam=False)   # skip CAM for speed
    result["source"] = "phone_cam"

    # Include the image for dashboard preview
    result["image_b64"] = base64.b64encode(data).decode("ascii")
    latest_phone_result = result
    return JSONResponse(content=result)


@app.get("/api/latest-phone")
async def get_latest_phone():
    """
    Polling endpoint: returns the last result received from a phone camera.
    Dashboard polls this to update the phone live view.
    """
    if not latest_phone_result:
        return JSONResponse(content={"status": "no_data", "message": "No phone camera data received yet."})
    return JSONResponse(content=latest_phone_result)


def _get_lan_ip() -> str:
    """Get this machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/api/server-info")
async def server_info():
    """Return server's LAN IP and port for QR code generation."""
    return {
        "lan_ip": _get_lan_ip(),
        "port": 9090,
        "phone_cam_url": f"http://{_get_lan_ip()}:9090/phone-cam",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/qr-code")
async def get_qr_code():
    """Generate a QR code PNG image for the phone camera URL (server-side)."""
    import qrcode
    url = f"http://{_get_lan_ip()}:9090/phone-cam"
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=3)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0c1528", back_color="#ffffff")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png", headers={"Cache-Control": "no-cache"})


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
