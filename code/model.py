"""
1D CNN for microlensing light-curve classification.

Input:  a fixed-length, multi-channel magnitude/brightness series
        (batch, in_channels, L) — in_channels=2 for the gap-aware pipeline
        (brightness + validity mask), see data.resample_curve_binned.
Output: 3 class logits (batch, 3) -> softmax for
        [CLASS_NO_EVENT, CLASS_EVENT, CLASS_AMBIGUOUS].

CLASS_AMBIGUOUS is the disagreement class: citizen-science volunteers who
can't reach consensus on an event are the ONLY source of training signal for
it (see code/retrain_from_votes.py) — the catalog-based train/val sets only
ever populate classes 0/1. A model checkpoint trained before this class
existed (2 output units, sigmoid) can be upgraded via
transplant_binary_checkpoint() below rather than retrained from scratch.

Deliberately small: this is a first-pass baseline, meant to be trained fast on
CPU and to give us an honest recall / FPR / AUC number to improve on.
"""
import torch
import torch.nn as nn

CLASS_NO_EVENT = 0
CLASS_EVENT = 1
CLASS_AMBIGUOUS = 2
NUM_CLASSES = 3


class MicrolensingCNN(nn.Module):
    def __init__(self, in_channels: int = 1, length: int = 200, dropout: float = 0.3,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),   # L   -> L/2
            block(32, 64),            # L/2 -> L/4
            block(64, 128),           # L/4 -> L/8
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (batch, 128, 1)
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),  # head.5: class logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        # num_classes=1 (the pre-disagreement-class baseline): squeeze to a
        # single logit per example, (batch,), for BCEWithLogitsLoss callers
        # (train_cnn.py, train_ogle_cnn.py). num_classes=3: leave as
        # (batch, 3) class logits for CrossEntropyLoss.
        return x.squeeze(-1) if self.num_classes == 1 else x


def transplant_binary_checkpoint(state_dict: dict) -> dict:
    """
    Upgrade a state_dict saved from an older 2-class-shaped model
    (in_channels=2, head.5 = Linear(64, 1), sigmoid) to this module's
    3-class shape (head.5 = Linear(64, 3), softmax).

    Every layer except head.5 (the final Linear) has an identical shape and
    is copied as-is — the model's learned feature extraction is preserved.
    head.5's single output row (the old "is event" logit) becomes the new
    CLASS_EVENT row; CLASS_NO_EVENT and CLASS_AMBIGUOUS have no prior
    weights to transplant (the old model never predicted either as a
    separate class) and are left at PyTorch's default Linear init —
    fine-tuning is expected to shape them, not a mirrored/negated init,
    which would just be a different arbitrary starting point.
    """
    old_w, old_b = state_dict["head.5.weight"], state_dict["head.5.bias"]
    if old_w.shape[0] == NUM_CLASSES:
        return state_dict  # already 3-class, nothing to transplant
    if old_w.shape[0] != 1:
        raise ValueError(f"Expected a 1-logit (binary) or {NUM_CLASSES}-class head.5, got shape {tuple(old_w.shape)}")

    new = MicrolensingCNN(in_channels=state_dict["features.0.0.weight"].shape[1], num_classes=NUM_CLASSES)
    new_state = new.state_dict()
    for key, val in state_dict.items():
        if key not in ("head.5.weight", "head.5.bias"):
            new_state[key] = val
    new_state["head.5.weight"][CLASS_EVENT] = old_w[0]
    new_state["head.5.bias"][CLASS_EVENT] = old_b[0]
    return new_state
