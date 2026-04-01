import torch
import torch.nn as nn
import torch.nn as nn


class NeuroGuard(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes), 
        )

    def forward(self, x):
        return self.network(x)
