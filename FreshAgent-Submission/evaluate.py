"""
evaluate.py - Full model evaluation with comprehensive graphical metrics.

Generates:
  1. Confusion Matrix - Condition (Fresh / Rotten / Formalin_Mixed)
  2. Confusion Matrix - Fruit Type (Apple / Banana / Grape / Mango / Orange)
  3. Per-Class Precision / Recall / F1 bar charts
  4. ROC Curves (one-vs-rest) for Condition classes
  5. Confidence Distribution histograms
  6. Summary metrics panel

Usage:
  python evaluate.py --data_dir "Augmented-Resized Image" --model_path models/best_model.pth
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))
from dataset import build_dataloaders, FRUIT_CLASSES, COND_CLASSES, IMG_SIZE
from model import FreshAgentModel

# ── Style (white / IEEE-print friendly) ───────────────────────────────────────
BG      = "#ffffff"
SURFACE = "#f8f9fa"
ACCENT  = "#1a73e8"
ACCENT2 = "#7c4dff"
SAFE    = "#1b8a5a"
WARN    = "#e67e00"
DANGER  = "#c0392b"
TEXT    = "#1a1a2e"
MUTED   = "#555577"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#dddddd",
    "grid.alpha":        0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

COND_COLORS  = [SAFE, WARN, DANGER]
FRUIT_COLORS = [ACCENT, WARN, ACCENT2, SAFE, "#e74c3c"]


# ── Device & Model ────────────────────────────────────────────────────────────
def load_model(model_path, device):
    model = FreshAgentModel(pretrained=False)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    all_fruit_true, all_cond_true = [], []
    all_fruit_pred, all_cond_pred = [], []
    all_cond_probs = []

    for batch in loader:
        imgs         = batch["image"].to(device)
        fruit_labels = batch["fruit_label"]
        cond_labels  = batch["cond_label"]
        fruit_logits, cond_logits = model(imgs)
        fruit_probs = F.softmax(fruit_logits.cpu(), dim=1)
        cond_probs  = F.softmax(cond_logits.cpu(),  dim=1)

        all_fruit_true.extend(fruit_labels.numpy())
        all_cond_true.extend(cond_labels.numpy())
        all_fruit_pred.extend(fruit_probs.argmax(dim=1).numpy())
        all_cond_pred.extend(cond_probs.argmax(dim=1).numpy())
        all_cond_probs.append(cond_probs.numpy())

    return (
        np.array(all_fruit_true), np.array(all_cond_true),
        np.array(all_fruit_pred), np.array(all_cond_pred),
        np.vstack(all_cond_probs)
    )


# ── Plot helpers ───────────────────────────────────────────────────────────────
def save_confusion_matrix(cm, class_names, title, save_path, colors):
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 1.4), max(6, len(class_names) * 1.2)))
    fig.patch.set_facecolor(BG)

    # Normalize for color, keep raw counts for annotation
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    annot   = np.where(cm > 0, cm.astype(str), "")

    sns.heatmap(
        cm_norm, annot=cm, fmt="d",
        cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1, linecolor="#1e2d4a",
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, pad=16)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=12)
    ax.set_ylabel("Actual",    color=MUTED, fontsize=12)
    ax.tick_params(colors=TEXT)
    plt.colorbar(ax.collections[0], ax=ax).set_label("Normalized", color=MUTED)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def save_per_class_metrics(true, pred, class_names, title, save_path, colors):
    p, r, f, _ = precision_recall_fscore_support(true, pred, labels=range(len(class_names)), zero_division=0)
    x = np.arange(len(class_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 2), 6))
    fig.patch.set_facecolor(BG)

    b1 = ax.bar(x - w,     p, w, label="Precision", color=ACCENT,  alpha=0.85, edgecolor="none")
    b2 = ax.bar(x,         r, w, label="Recall",    color=ACCENT2, alpha=0.85, edgecolor="none")
    b3 = ax.bar(x + w,     f, w, label="F1-Score",  color=SAFE,    alpha=0.85, edgecolor="none")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in class_names], color=TEXT)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color=MUTED)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, pad=14)
    ax.legend(facecolor=SURFACE, edgecolor="#1e2d4a", labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def save_roc_curves(true, probs, class_names, save_path):
    n_classes = len(class_names)
    true_bin  = label_binarize(true, classes=range(n_classes))
    colors    = [SAFE, WARN, DANGER, ACCENT, ACCENT2]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG)

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(true_bin[:, i], probs[:, i])
        roc_auc      = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, linestyle="--", label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.03])
    ax.set_xlabel("False Positive Rate", color=MUTED)
    ax.set_ylabel("True Positive Rate",  color=MUTED)
    ax.set_title("ROC Curves – Condition Detection (One-vs-Rest)",
                 fontsize=13, fontweight="bold", color=TEXT, pad=14)
    ax.legend(facecolor=SURFACE, edgecolor="#1e2d4a", labelcolor=TEXT, loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def save_confidence_distribution(cond_probs, cond_true, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Confidence Distribution per Condition Class", fontsize=13, fontweight="bold", color=TEXT)

    colors = [SAFE, WARN, DANGER]
    for i, (name, color) in enumerate(zip(COND_CLASSES, colors)):
        ax = axes[i]
        correct_mask = cond_true == i
        probs_class  = cond_probs[correct_mask, i]
        probs_wrong  = cond_probs[~correct_mask, i]

        ax.hist(probs_class, bins=30, color=color, alpha=0.75, label="Correct", edgecolor="none")
        ax.hist(probs_wrong, bins=30, color=DANGER, alpha=0.4,  label="Incorrect", edgecolor="none")
        ax.set_title(name, color=TEXT, fontweight="bold")
        ax.set_xlabel("Confidence", color=MUTED)
        ax.set_ylabel("Count", color=MUTED)
        ax.legend(facecolor=SURFACE, edgecolor="#1e2d4a", labelcolor=TEXT, fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def save_summary_panel(fruit_true, fruit_pred, cond_true, cond_pred, save_path):
    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "Condition\nAccuracy":  accuracy_score(cond_true,  cond_pred),
        "Condition\nF1 (macro)": f1_score(cond_true, cond_pred, average="macro"),
        "Fruit\nAccuracy":      accuracy_score(fruit_true, fruit_pred),
        "Fruit\nF1 (macro)":   f1_score(fruit_true, fruit_pred, average="macro"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("FreshAgent – Model Performance Summary", fontsize=14, fontweight="bold", color=TEXT, y=1.02)

    gradients = [SAFE, ACCENT, ACCENT2, WARN]
    for ax, (label, value), color in zip(axes, metrics.items(), gradients):
        ax.set_facecolor(SURFACE)
        pct = value * 100
        # Big number
        ax.text(0.5, 0.55, f"{pct:.2f}%",
                ha="center", va="center", fontsize=30, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.22, label,
                ha="center", va="center", fontsize=11, color=MUTED,
                transform=ax.transAxes)
        # Progress arc
        theta = np.linspace(0, 2 * np.pi * value, 200)
        x_arc = 0.5 + 0.4 * np.cos(theta - np.pi/2)
        y_arc = 0.5 + 0.4 * np.sin(theta - np.pi/2)
        ax.plot(x_arc, y_arc, color=color, lw=5, transform=ax.transAxes, solid_capstyle="round")
        circle_bg = plt.Circle((0.5, 0.5), 0.4, color="#1e2d4a", fill=False, lw=5, transform=ax.transAxes)
        ax.add_patch(circle_bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"[Eval] Device: {device}")

    print("[Eval] Loading dataset (test split)...")
    _, _, test_loader, _ = build_dataloaders(
        root_dir=args.data_dir, batch_size=32, num_workers=0, seed=42
    )

    print("[Eval] Loading model...")
    model = load_model(args.model_path, device)

    print("[Eval] Running inference on test set (this may take ~10 min)...")
    fruit_true, cond_true, fruit_pred, cond_pred, cond_probs = run_inference(model, test_loader, device)

    print("\n[Eval] === CONDITION METRICS ===")
    print(classification_report(cond_true, cond_pred, target_names=COND_CLASSES))
    print("\n[Eval] === FRUIT METRICS ===")
    print(classification_report(fruit_true, fruit_pred, target_names=FRUIT_CLASSES))

    d = args.output_dir

    print("\n[Eval] Generating plots...")

    # 1. Confusion Matrix - Condition
    cm_cond = confusion_matrix(cond_true, cond_pred)
    save_confusion_matrix(cm_cond, COND_CLASSES,
                          "Condition Confusion Matrix (Test Set)",
                          os.path.join(d, "1_confusion_matrix_condition.png"), COND_COLORS)

    # 2. Confusion Matrix - Fruit
    cm_fruit = confusion_matrix(fruit_true, fruit_pred)
    save_confusion_matrix(cm_fruit, FRUIT_CLASSES,
                          "Fruit Type Confusion Matrix (Test Set)",
                          os.path.join(d, "2_confusion_matrix_fruit.png"), FRUIT_COLORS)

    # 3. Per-class metrics - Condition
    save_per_class_metrics(cond_true, cond_pred, COND_CLASSES,
                           "Per-Class Metrics – Condition Detection",
                           os.path.join(d, "3_per_class_metrics_condition.png"), COND_COLORS)

    # 4. Per-class metrics - Fruit
    save_per_class_metrics(fruit_true, fruit_pred, FRUIT_CLASSES,
                           "Per-Class Metrics – Fruit Classification",
                           os.path.join(d, "4_per_class_metrics_fruit.png"), FRUIT_COLORS)

    # 5. ROC Curves - Condition
    save_roc_curves(cond_true, cond_probs, COND_CLASSES,
                    os.path.join(d, "5_roc_curves_condition.png"))

    # 6. Confidence distribution
    save_confidence_distribution(cond_probs, cond_true,
                                 os.path.join(d, "6_confidence_distribution.png"))

    # 7. Summary panel
    save_summary_panel(fruit_true, fruit_pred, cond_true, cond_pred,
                       os.path.join(d, "7_summary_metrics_panel.png"))

    print(f"\n[Done] All 7 evaluation graphs saved to: {d}/")
    print("  1_confusion_matrix_condition.png")
    print("  2_confusion_matrix_fruit.png")
    print("  3_per_class_metrics_condition.png")
    print("  4_per_class_metrics_fruit.png")
    print("  5_roc_curves_condition.png")
    print("  6_confidence_distribution.png")
    print("  7_summary_metrics_panel.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, required=True,          help="Dataset root directory")
    parser.add_argument("--model_path",  type=str, default="models/best_model.pth")
    parser.add_argument("--output_dir",  type=str, default="models/eval_plots")
    main(parser.parse_args())
