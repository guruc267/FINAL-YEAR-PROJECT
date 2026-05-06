"""
export.py – Export VeggieAgent PyTorch checkpoint → ONNX format.

Usage:
  python veggie_training/export.py --checkpoint models/veggieagent_best.pth --output models/veggieagent.onnx
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from veggie_dataset import VEGGIE_CLASSES, COND_CLASSES
from veggie_model import VeggieAgentModel


def export_onnx(checkpoint_path: str, output_path: str):
    print(f"[Export] Loading checkpoint: {checkpoint_path}")
    model = VeggieAgentModel(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")

    # Handle both raw state_dict and wrapped checkpoint formats
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[Export] Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["veggie_logits", "cond_logits"],
        dynamic_axes={
            "image":         {0: "batch_size"},
            "veggie_logits": {0: "batch_size"},
            "cond_logits":   {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[Export] Done! File size: {size_mb:.1f} MB")
    print(f"[Export] Veggie classes : {VEGGIE_CLASSES}  ({len(VEGGIE_CLASSES)} total)")
    print(f"[Export] Cond classes   : {COND_CLASSES}")

    # Quick validation with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"image": dummy_input.numpy()})
        print(f"[Export] ONNX validation OK — veggie_logits shape: {out[0].shape}, cond_logits shape: {out[1].shape}")
    except Exception as e:
        print(f"[Export] ONNX validation skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VeggieAgent to ONNX")
    parser.add_argument("--checkpoint", type=str, default="models/veggieagent_best.pth", help="Path to .pth checkpoint")
    parser.add_argument("--output",     type=str, default="models/veggieagent.onnx",     help="Output ONNX file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_onnx(args.checkpoint, args.output)
