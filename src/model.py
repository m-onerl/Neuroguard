import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from train import train

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(name)s = %(levelname)s - %(message)s'
)


class NeuroGuard(nn.Module):
    def __init__(self, input_dim):
        super(NeuroGuard, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward (self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.sigmoid(self.out(x))
  
            
if __name__ == "__main__":
    train()