"""
export.py – Export trained PyTorch model to ONNX for fast server-side inference.

Usage:
  python training/export.py --checkpoint ./models/best_model.pth --output ./models/freshagent.onnx
"""
import argparse
import sys
import os

import torch
import onnx
import onnxruntime as ort
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FreshAgentModel
from dataset import FRUIT_CLASSES, COND_CLASSES, IMG_SIZE


def export_to_onnx(checkpoint_path: str, output_path: str, opset: int = 17):
    print(f"[Export] Loading checkpoint: {checkpoint_path}")
    model = FreshAgentModel(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    input_names  = ["input_image"]
    output_names = ["fruit_logits", "cond_logits"]

    print(f"[Export] Exporting to ONNX (opset {opset}) …")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_image":  {0: "batch_size"},
            "fruit_logits": {0: "batch_size"},
            "cond_logits":  {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[Export] ONNX model validated OK: {output_path}")

    # Quick inference smoke test with ONNX Runtime
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_in = {"input_image": dummy_input.numpy()}
    fl, cl = sess.run(None, ort_in)
    print(f"[Export] ONNX Runtime smoke test PASSED")
    print(f"         Fruit logits shape : {fl.shape}")
    print(f"         Cond  logits shape : {cl.shape}")
    print(f"\n[Export] Done! Use this ONNX file for the FastAPI backend.")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output",     type=str, default="./models/freshagent.onnx")
    parser.add_argument("--opset",      type=int, default=17)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_to_onnx(args.checkpoint, args.output, args.opset)
