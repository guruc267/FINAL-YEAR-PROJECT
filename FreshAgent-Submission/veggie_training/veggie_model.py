"""
model.py – Multi-Task EfficientNetV2-B0 model for vegetable adulteration detection.

Architecture:
  EfficientNetV2-B0 backbone (ImageNet pretrained via timm)
    └─► Shared Feature Extractor
          ├─► Head 1: Veggie Classifier  (N classes, starts with 1: Ginger)
          └─► Head 2: Condition Classifier (3 classes: Adulterated / Fresh / Rotten)
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import VEGGIE_CLASSES, COND_CLASSES


class MultiTaskHead(nn.Module):
    """A two-layer dropout head used for both classification tasks."""
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class VeggieAgentModel(nn.Module):
    """
    Multi-Task EfficientNetV2-B0 for vegetables.
    Returns logits for (veggie_type, condition) separately.
    """
    def __init__(
        self,
        num_veggie_classes: int = len(VEGGIE_CLASSES),  # starts at 1 (Ginger)
        num_cond_classes:   int = len(COND_CLASSES),    # 3
        pretrained:         bool = True,
        dropout:            float = 0.3,
    ):
        super().__init__()
        # ─── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b0",
            pretrained=pretrained,
            num_classes=0,       # Remove original classifier
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1280 for EfficientNetV2-B0

        # ─── Multi-task Heads ──────────────────────────────────────────────────
        self.veggie_head = MultiTaskHead(in_features, num_veggie_classes, dropout)
        self.cond_head   = MultiTaskHead(in_features, num_cond_classes,   dropout)

    def forward(self, x):
        feats         = self.backbone(x)             # (B, 1280)
        veggie_logits = self.veggie_head(feats)      # (B, num_veggie_classes)
        cond_logits   = self.cond_head(feats)        # (B, 3)
        return veggie_logits, cond_logits

    def freeze_backbone(self):
        """Freeze backbone for initial warm-up phase."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze entire backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 3):
        """Partially unfreeze (last n blocks) for gradual unfreezing strategy."""
        blocks = list(self.backbone.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True


class FocalLoss(nn.Module):
    """
    Focal Loss with optional class weights and label smoothing.
    Useful when Adulterated class is under-represented.
    """
    def __init__(
        self,
        weight=None,
        gamma:  float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.weight          = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probs = torch.exp(-ce)
        focal = (1 - probs) ** self.gamma * ce
        return focal.mean()


def build_criterion(cond_class_weights=None, device="cpu"):
    """
    Build loss functions for both tasks.
    Condition head uses focal loss for imbalanced adulteration classes.
    Veggie head uses standard cross-entropy.
    """
    if cond_class_weights is not None:
        cond_class_weights = cond_class_weights.to(device)

    veggie_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    cond_criterion   = FocalLoss(weight=None, gamma=2.0, label_smoothing=0.1)
    return veggie_criterion, cond_criterion


if __name__ == "__main__":
    # Quick sanity check
    model = VeggieAgentModel(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    vl, cl = model(dummy)
    print(f"Veggie logits shape : {vl.shape}")   # Expected: (4, 1) for Ginger only
    print(f"Cond  logits shape  : {cl.shape}")   # Expected: (4, 3)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
