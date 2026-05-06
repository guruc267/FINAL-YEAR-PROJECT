"""
inference.py – ONNX Runtime inference + Grad-CAM via pytorch-grad-cam.

Provides:
  - FreshAgentInference: loads ONNX model, preprocesses images, returns predictions
  - generate_gradcam: runs Grad-CAM on the PyTorch model for heatmap visualization
"""
import io
import os
import sys
import base64
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from dataset import FRUIT_CLASSES, COND_CLASSES, IMG_SIZE

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
class FreshAgentInference:
    """
    Wraps the ONNX Runtime session for fast server inference.
    Falls back gracefully if ONNX model is not yet available.
    """
    def __init__(self, onnx_path: str):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found at '{onnx_path}'.\n"
                "Please train the model first:\n"
                "  python training/train.py --data_dir <dataset_path>\n"
                "Then export:\n"
                "  python training/export.py --checkpoint models/best_model.pth"
            )
        providers = ["CPUExecutionProvider"]  # ONNX Runtime on CPU (optimised)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[Inference] ONNX model loaded: {onnx_path}")

    def predict(self, img: Image.Image) -> dict:
        """
        Run inference on a PIL Image.
        Returns a dict with all prediction details.
        """
        input_arr = preprocess_image(img)
        fruit_logits, cond_logits = self.session.run(None, {self.input_name: input_arr})

        fruit_probs = softmax(fruit_logits)[0]
        cond_probs  = softmax(cond_logits)[0]

        fruit_idx  = int(np.argmax(fruit_probs))
        cond_idx   = int(np.argmax(cond_probs))

        return {
            "fruit":            FRUIT_CLASSES[fruit_idx],
            "condition":        COND_CLASSES[cond_idx],
            "fruit_confidence": float(round(fruit_probs[fruit_idx] * 100, 2)),
            "cond_confidence":  float(round(cond_probs[cond_idx]  * 100, 2)),
            "fruit_probs":  {FRUIT_CLASSES[i]: round(float(p)*100, 2) for i, p in enumerate(fruit_probs)},
            "cond_probs":   {COND_CLASSES[i]:  round(float(p)*100, 2) for i, p in enumerate(cond_probs)},
        }


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM (uses PyTorch model for explainability; separate from ONNX inference)
# ──────────────────────────────────────────────────────────────────────────────
def generate_gradcam(pth_path: str, img: Image.Image, target_class_idx: int = None) -> str:
    """
    Generate a Grad-CAM heatmap overlaid on the original image.
    Returns a base64-encoded PNG string.
    Requires the .pth checkpoint (ONNX doesn't support gradient hooks).
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from model import FreshAgentModel
    except ImportError:
        return ""

    if not os.path.exists(pth_path):
        return ""

    model = FreshAgentModel(pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    # Target layer: last block of the EfficientNetV2 backbone
    target_layers = [model.backbone.conv_head]

    tensor = PREPROCESS(img.convert("RGB")).unsqueeze(0)
    rgb_img = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0

    # Use the condition head logits for the CAM
    class CondHead(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): _, cl = self.m(x); return cl

    wrapped = CondHead(model)

    with GradCAM(model=wrapped, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam[0], use_rgb=True)

    # Convert to base64 PNG
    pil_out = Image.fromarray(visualization)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
