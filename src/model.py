import pandas as pd
import numpy as np

x = np.load('./data/processed/X_train.npy')
y = np.load('./data/processed/y_train.npy')

print(x.shape)
print(y.shape)