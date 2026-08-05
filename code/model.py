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


class _GradientReversalFunction(torch.autograd.Function):
    """Ganin & Lempitsky (2015) gradient-reversal layer: identity in the
    forward pass, multiplies the incoming gradient by -lambd in backward.
    This is the entire domain-adversarial mechanism -- the domain classifier
    trains normally (best-effort to tell OGLE from KMTNet), while the shared
    feature extractor gets pushed the opposite direction (to make that
    impossible), with no change to the forward-pass loss value itself."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambd: float = 0.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambd)


class DANNMicrolensingCNN(MicrolensingCNN):
    """MicrolensingCNN extended with a domain-classification head behind a
    GradientReversalLayer, for domain-adversarial training toward
    survey-invariant features (KARTIKFUTUREPLANNING.md objective 1: "close
    to same accuracy across surveys, not learning where the model comes
    from"). See code/train_ogle_dann.py for the training loop and the
    pre-registered success criteria this is meant to satisfy.

    `features`/`pool`/`head` are identical in name and shape to the base
    MicrolensingCNN(in_channels=2, num_classes=1) -- so a deployed OGLE
    checkpoint loads directly as a warm start (state_dict(), strict=False,
    since domain_head/grl have no prior weights), and `base_state_dict()`
    below strips the domain head back out, producing a checkpoint every
    existing eval script (the five cross-survey checks, precision_curve.py,
    the scorecard) can load completely unchanged -- the domain head is a
    training-time-only scaffold, never part of the deployed architecture.
    """

    def __init__(self, in_channels: int = 1, length: int = 200, dropout: float = 0.3,
                 num_classes: int = 1, domain_hidden: int = 64):
        super().__init__(in_channels=in_channels, length=length, dropout=dropout,
                          num_classes=num_classes)
        self.grl = GradientReversalLayer(lambd=0.0)
        self.domain_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, domain_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden, 1),  # domain logit: 0 = OGLE (source), 1 = KMTNet (target)
        )

    def set_lambda(self, lambd: float):
        self.grl.lambd = lambd

    def extract(self, x):
        """Pooled feature vector, (batch, 128, 1) -- shared input to both
        the class head and (via the GRL) the domain head."""
        return self.pool(self.features(x))

    def classify(self, feats):
        out = self.head(feats)
        return out.squeeze(-1) if self.num_classes == 1 else out

    def domain_logits(self, feats):
        return self.domain_head(self.grl(feats)).squeeze(-1)

    def forward(self, x, return_domain: bool = False):
        feats = self.extract(x)
        class_out = self.classify(feats)
        if not return_domain:
            return class_out
        return class_out, self.domain_logits(feats)

    def base_state_dict(self):
        """features/pool/head only, matching MicrolensingCNN's own
        checkpoint shape exactly -- drops domain_head/grl for saving a
        deployment-compatible checkpoint."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith(("domain_head.", "grl."))}


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
