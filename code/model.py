"""
1D CNN for microlensing light-curve classification.

Input:  a fixed-length, single-channel magnitude series  (batch, 1, L)
        (optionally 2 channels if you stack magerr — set in_channels=2)
Output: a single logit -> P(microlensing) via sigmoid.

Deliberately small: this is a first-pass baseline, meant to be trained fast on
CPU and to give us an honest recall / FPR / AUC number to improve on.
"""
import torch
import torch.nn as nn


class MicrolensingCNN(nn.Module):
    def __init__(self, in_channels: int = 1, length: int = 200, dropout: float = 0.3):
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),         # single logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)  # (batch,)
