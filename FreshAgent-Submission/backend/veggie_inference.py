"""
veggie_inference.py – ONNX Runtime inference for VeggieAgent vegetable detection.

Mirrors the structure of inference.py (fruits) but uses VEGGIE_CLASSES and
the veggieagent.onnx model. Fruit model is never touched.
"""
import io
import os
import sys
import base64
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as T

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "veggie_training"))
from veggie_dataset import VEGGIE_CLASSES, COND_CLASSES, IMG_SIZE, COND_DISPLAY

# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PREPROCESS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert PIL Image → normalised numpy array (1, 3, H, W) for ONNX Runtime."""
    tensor = PREPROCESS(img.convert("RGB"))
    return tensor.unsqueeze(0).numpy()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ──────────────────────────────────────────────────────────────────────────────
class VeggieAgentInference:
    """
    Wraps the ONNX Runtime session for fast server inference (vegetables).
    Completely separate from FreshAgentInference — no shared state.
    """
    def __init__(self, onnx_path: str):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"VeggieAgent ONNX model not found at '{onnx_path}'.\n"
                "Please train the model first:\n"
                "  train_veggie.bat <path_to_veggie_dataset>\n"
                "Or manually:\n"
                "  python veggie_training/train.py --data_dir <path>\n"
                "  python veggie_training/export.py --checkpoint models/veggieagent_best.pth"
            )
        providers = ["CPUExecutionProvider"]
        self.session    = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[VeggieInference] ONNX model loaded: {onnx_path}")

    def predict(self, img: Image.Image) -> dict:
        """
        Run inference on a PIL Image.
        Returns a dict with veggie name, condition, confidence scores.
        """
        input_arr = preprocess_image(img)
        veggie_logits, cond_logits = self.session.run(None, {self.input_name: input_arr})

        veggie_probs = softmax(veggie_logits)[0]
        cond_probs   = softmax(cond_logits)[0]

        veggie_idx = int(np.argmax(veggie_probs))
        cond_idx   = int(np.argmax(cond_probs))

        return {
            "veggie":            VEGGIE_CLASSES[veggie_idx],
            "condition":         COND_CLASSES[cond_idx],
            "condition_display": COND_DISPLAY.get(COND_CLASSES[cond_idx], COND_CLASSES[cond_idx]),
            "veggie_confidence": float(round(veggie_probs[veggie_idx] * 100, 2)),
            "cond_confidence":   float(round(cond_probs[cond_idx]    * 100, 2)),
            "veggie_probs": {VEGGIE_CLASSES[i]: round(float(p)*100, 2) for i, p in enumerate(veggie_probs)},
            "cond_probs":   {COND_CLASSES[i]:   round(float(p)*100, 2) for i, p in enumerate(cond_probs)},
        }


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM for vegetables (uses PyTorch .pth checkpoint)
# ──────────────────────────────────────────────────────────────────────────────
def generate_veggie_gradcam(pth_path: str, img: Image.Image, target_class_idx: int = None) -> str:
    """
    Generate a Grad-CAM heatmap for a vegetable image.
    Returns a base64-encoded PNG string.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        sys.path.insert(0, str(BASE_DIR / "veggie_training"))
        from veggie_model import VeggieAgentModel
    except ImportError:
        return ""

    if not os.path.exists(pth_path):
        return ""

    model = VeggieAgentModel(pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    target_layers = [model.backbone.conv_head]

    tensor  = PREPROCESS(img.convert("RGB")).unsqueeze(0)
    rgb_img = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0

    class CondHead(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): _, cl = self.m(x); return cl

    wrapped = CondHead(model)

    with GradCAM(model=wrapped, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam[0], use_rgb=True)

    pil_out = Image.fromarray(visualization)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
