import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from torch.utils.data import DataLoader, TensorDataset

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
    
def train():
    
        
    x_raw = np.load('./data/processed/X_train.npy')
    y_raw = np.load('./data/processed/y_train.npy')


    X_tensor = torch.from_numpy(x_raw).float()
    y_tensor = torch.from_numpy(y_raw).float().reshape(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size = 64, shuffle = True)
    
    model = NeuroGuard(input_dim = X_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    epochs = 10
    
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            logger.info(f'Epoch {epoch+1}/{epochs} Average of loss {running_loss/len(train_loader)}')
            
if __name__ == "__main__":
    train()