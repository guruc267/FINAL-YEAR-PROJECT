"""
train.py – Full training loop for VeggieAgent adulteration detection model.

Features:
  • Intel Iris Xe support via torch-directml (DirectML device)
  • Two-phase training: warm-up (heads only) → fine-tune (whole network)
  • MixUp augmentation
  • AdamW + Cosine Annealing with Warm-Up
  • Early stopping on validation F1 score
  • Checkpointing best model

Usage:
  python veggie_training/train.py --data_dir ./veggie_dataset --epochs 60 --batch_size 16
"""
import argparse
import os
import sys
import time
import copy
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
)
from tqdm import tqdm

# ─── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from veggie_dataset import (
    build_dataloaders, mixup_data,
    VEGGIE_CLASSES, COND_CLASSES, COMBINED_CLASSES
)
from veggie_model import VeggieAgentModel, build_criterion

# ──────────────────────────────────────────────────────────────────────────────
# Device setup: CUDA then CPU
# ──────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[Device] Falling back to CPU")
    return torch.device("cpu")



# ──────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(all_preds, all_targets, class_names):
    acc  = accuracy_score(all_targets, all_preds)
    f1   = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    cm   = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    return {"acc": acc, "f1": f1, "precision": prec, "recall": rec, "cm": cm}


# ──────────────────────────────────────────────────────────────────────────────
# Training epoch
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, veggie_crit, cond_crit, device, use_mixup=True):
    model.train()
    running_loss   = 0.0
    veggie_preds, veggie_targets = [], []
    cond_preds,   cond_targets   = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        vl     = batch["veggie_label"].to(device)
        cl     = batch["cond_label"].to(device)

        # MixUp augmentation
        if use_mixup and np.random.rand() < 0.5:
            images, vl_a, vl_b, cl_a, cl_b, lam = mixup_data(images, vl, cl, alpha=0.4)
            veggie_logits, cond_logits = model(images)
            loss = (lam * veggie_crit(veggie_logits, vl_a) + (1 - lam) * veggie_crit(veggie_logits, vl_b)
                  + lam * cond_crit(cond_logits,    cl_a) + (1 - lam) * cond_crit(cond_logits,    cl_b))
            vl_used = vl_a
            cl_used = cl_a
        else:
            veggie_logits, cond_logits = model(images)
            loss = veggie_crit(veggie_logits, vl) + cond_crit(cond_logits, cl)
            vl_used = vl
            cl_used = cl

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        veggie_preds.extend(veggie_logits.argmax(1).cpu().numpy())
        veggie_targets.extend(vl_used.cpu().numpy())
        cond_preds.extend(cond_logits.argmax(1).cpu().numpy())
        cond_targets.extend(cl_used.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss   = running_loss / len(loader)
    veggie_met = compute_metrics(veggie_preds, veggie_targets, VEGGIE_CLASSES)
    cond_met   = compute_metrics(cond_preds,   cond_targets,   COND_CLASSES)
    return avg_loss, veggie_met, cond_met


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation epoch
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, veggie_crit, cond_crit, device, split="Val"):
    model.eval()
    running_loss   = 0.0
    veggie_preds, veggie_targets = [], []
    cond_preds,   cond_targets   = [], []

    pbar = tqdm(loader, desc=f"  {split:<5}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        vl     = batch["veggie_label"].to(device)
        cl     = batch["cond_label"].to(device)
        veggie_logits, cond_logits = model(images)
        loss = veggie_crit(veggie_logits, vl) + cond_crit(cond_logits, cl)
        running_loss += loss.item()
        veggie_preds.extend(veggie_logits.argmax(1).cpu().numpy())
        veggie_targets.extend(vl.cpu().numpy())
        cond_preds.extend(cond_logits.argmax(1).cpu().numpy())
        cond_targets.extend(cl.cpu().numpy())

    avg_loss   = running_loss / len(loader)
    veggie_met = compute_metrics(veggie_preds, veggie_targets, VEGGIE_CLASSES)
    cond_met   = compute_metrics(cond_preds,   cond_targets,   COND_CLASSES)
    return avg_loss, veggie_met, cond_met


# ──────────────────────────────────────────────────────────────────────────────
# Resume checkpoint
# ──────────────────────────────────────────────────────────────────────────────
def save_resume_checkpoint(path, epoch, model, optimizer, scheduler,
                           best_f1, patience_cnt, history, warmup_done):
    torch.save({
        "epoch":        epoch,
        "warmup_done":  warmup_done,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict() if scheduler else None,
        "best_f1":      best_f1,
        "patience_cnt": patience_cnt,
        "history":      history,
    }, path)
    print(f"  [SAVED] Resume checkpoint saved (epoch {epoch})")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    resume_path = os.path.join(args.output_dir, "veggie_resume_checkpoint.pth")

    # ─── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )

    # ─── Model ─────────────────────────────────────────────────────────────────
    model = VeggieAgentModel(pretrained=True).to(device)
    veggie_crit, cond_crit = build_criterion(
        cond_class_weights=class_weights, device=device
    )

    # ─── Setup Phase-2 optimizer & scheduler ───────────────────────────────────
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_f1       = 0.0
    patience_cnt  = 0
    history       = []
    start_epoch   = 1
    warmup_done   = False

    # ─── Try to resume from checkpoint ────────────────────────────────────────
    if os.path.exists(resume_path):
        print(f"\n[Resume] Found checkpoint at: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        best_f1       = ckpt["best_f1"]
        patience_cnt  = ckpt["patience_cnt"]
        history       = ckpt["history"]
        warmup_done   = ckpt["warmup_done"]
        start_epoch   = ckpt["epoch"] + 1
        print(f"[Resume] Resuming from epoch {start_epoch}/{args.epochs}  "
              f"Best F1 so far: {best_f1:.4f}")
    else:
        print("[Resume] No checkpoint found — starting fresh.")

    # ─── PHASE 1: Warm-up ─────────────────────────────────────────────────────
    if not warmup_done:
        print("\n[Phase 1] Warm-up: training heads only (backbone frozen) …")
        model.freeze_backbone()
        warmup_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 5, weight_decay=1e-4
        )
        warmup_epochs = min(5, args.epochs // 6)
        warmup_epochs = max(warmup_epochs, 1)
        for wu_epoch in range(1, warmup_epochs + 1):
            t_loss, t_vm, t_cm = train_one_epoch(
                model, train_loader, warmup_optimizer,
                veggie_crit, cond_crit, device, use_mixup=False
            )
            v_loss, v_vm, v_cm = evaluate(model, val_loader, veggie_crit, cond_crit, device)
            print(f"  Warmup [{wu_epoch:02d}/{warmup_epochs}]  "
                  f"T-Loss: {t_loss:.4f}  V-Loss: {v_loss:.4f}  "
                  f"V-Cond-F1: {v_cm['f1']:.4f}")
        model.unfreeze_backbone()
        warmup_done = True
        print("[Phase 1] Warm-up complete.")

    # ─── PHASE 2: Fine-tune entire network ────────────────────────────────────
    print(f"\n[Phase 2] Fine-tuning entire network "
          f"(epochs {start_epoch}–{args.epochs}) …")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        t_loss, t_vm, t_cm = train_one_epoch(
            model, train_loader, optimizer, veggie_crit, cond_crit, device, use_mixup=True
        )
        v_loss, v_vm, v_cm = evaluate(model, val_loader, veggie_crit, cond_crit, device)
        scheduler.step()

        elapsed  = time.time() - t0
        cond_f1  = v_cm["f1"]

        row = {
            "epoch":          epoch,
            "train_loss":     round(t_loss, 5), "val_loss":       round(v_loss, 5),
            "train_cond_f1":  round(t_cm["f1"], 5),
            "val_cond_f1":    round(cond_f1, 5),
            "train_veggie_f1":round(t_vm["f1"], 5),
            "val_veggie_f1":  round(v_vm["f1"], 5),
            "val_cond_acc":   round(v_cm["acc"], 5),
        }
        history.append(row)
        print(f"  Epoch [{epoch:03d}/{args.epochs}]  "
              f"T: {t_loss:.4f}  V: {v_loss:.4f}  "
              f"Cond-F1: {cond_f1:.4f}  Veggie-F1: {v_vm['f1']:.4f}  "
              f"({elapsed:.1f}s)")

        # ── Best model checkpoint ──────────────────────────────────────────────
        if cond_f1 > best_f1:
            best_f1 = cond_f1
            patience_cnt = 0
            torch.save(copy.deepcopy(model.state_dict()),
                       os.path.join(args.output_dir, "veggieagent_best.pth"))
            print(f"  [BEST] Best model saved (Cond F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n  [STOP] Early stopping at epoch {epoch} "
                      f"(patience={args.patience})")
                save_resume_checkpoint(
                    resume_path, epoch, model, optimizer, scheduler,
                    best_f1, patience_cnt, history, warmup_done
                )
                break

        save_resume_checkpoint(
            resume_path, epoch, model, optimizer, scheduler,
            best_f1, patience_cnt, history, warmup_done
        )

    # ─── Training complete ─────────────────────────────────────────────────────
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print("\n[Resume] Training complete — checkpoint file removed.")

    # ─── Test Evaluation ───────────────────────────────────────────────────────
    print("\n[Test] Loading best model …")
    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, "veggieagent_best.pth"), map_location=device
    ))
    te_loss, te_vm, te_cm = evaluate(
        model, test_loader, veggie_crit, cond_crit, device, split="Test"
    )

    print("\n" + "="*60)
    print("  VEGGIEAGENT — FINAL TEST RESULTS")
    print("="*60)
    print(f"  Condition  Accuracy : {te_cm['acc']*100:.2f}%")
    print(f"  Condition  F1-Score : {te_cm['f1']*100:.2f}%")
    print(f"  Condition  Precision: {te_cm['precision']*100:.2f}%")
    print(f"  Condition  Recall   : {te_cm['recall']*100:.2f}%")
    print(f"  Veggie     Accuracy : {te_vm['acc']*100:.2f}%")
    print("="*60)

    import numpy as npa
    npa.save(os.path.join(args.output_dir, "veggie_cond_cm.npy"),  te_cm["cm"])
    npa.save(os.path.join(args.output_dir, "veggie_item_cm.npy"),  te_vm["cm"])
    with open(os.path.join(args.output_dir, "veggie_train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] All VeggieAgent artifacts saved to: {args.output_dir}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeggieAgent Training Script")
    parser.add_argument("--data_dir",    type=str,   required=True,      help="Path to veggie dataset root directory")
    parser.add_argument("--output_dir",  type=str,   default="./models", help="Where to save checkpoints")
    parser.add_argument("--epochs",      type=int,   default=60,         help="Max training epochs")
    parser.add_argument("--batch_size",  type=int,   default=16,         help="Batch size (keep 8-16 for Iris Xe)")
    parser.add_argument("--lr",          type=float, default=1e-4,       help="Base learning rate for AdamW")
    parser.add_argument("--patience",    type=int,   default=15,         help="Early stopping patience (epochs)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()
    main(args)
