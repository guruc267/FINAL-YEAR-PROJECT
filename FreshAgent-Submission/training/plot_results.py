"""
plot_results.py – Generate training curves and confusion matrices from saved artifacts.

Usage:
  python training/plot_results.py --output_dir ./models
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import FRUIT_CLASSES, COND_CLASSES


def plot_training_curves(history: list, save_dir: str):
    epochs = [r["epoch"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FreshAgent – Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train", color="#22d3ee")
    axes[0].plot(epochs, [r["val_loss"]   for r in history], label="Val",   color="#8b5cf6")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    # Condition F1
    axes[1].plot(epochs, [r["train_cond_f1"] for r in history], label="Train", color="#22d3ee")
    axes[1].plot(epochs, [r["val_cond_f1"]   for r in history], label="Val",   color="#8b5cf6")
    axes[1].set_title("Condition F1-Score (macro)"); axes[1].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)

    # Fruit F1
    axes[2].plot(epochs, [r["train_fruit_f1"] for r in history], label="Train", color="#22d3ee")
    axes[2].plot(epochs, [r["val_fruit_f1"]   for r in history], label="Val",   color="#8b5cf6")
    axes[2].set_title("Fruit F1-Score (macro)"); axes[2].legend()
    axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.set_facecolor("#111827"); ax.grid(alpha=0.15)
    fig.patch.set_facecolor("#0a0e1a")
    for text in fig.findobj(matplotlib.text.Text): text.set_color("white")

    path = os.path.join(save_dir, "training_curves.png")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"Saved: {save_path}")


def main(args):
    d = args.output_dir
    history_path = os.path.join(d, "train_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f: history = json.load(f)
        plot_training_curves(history, d)

    cond_cm_path = os.path.join(d, "cond_cm.npy")
    if os.path.exists(cond_cm_path):
        cm = np.load(cond_cm_path)
        names = [c.replace("_", " ") for c in COND_CLASSES]
        plot_confusion_matrix(cm, names, "Condition Confusion Matrix (Test Set)",
                              os.path.join(d, "cond_confusion_matrix.png"))

    fruit_cm_path = os.path.join(d, "fruit_cm.npy")
    if os.path.exists(fruit_cm_path):
        cm = np.load(fruit_cm_path)
        plot_confusion_matrix(cm, FRUIT_CLASSES, "Fruit Confusion Matrix (Test Set)",
                              os.path.join(d, "fruit_confusion_matrix.png"))

    print("\n[Done] All plots saved to:", d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./models")
    main(parser.parse_args())
