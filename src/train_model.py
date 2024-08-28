from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F

if True:
    import sys
    system = platform.system()
    if system == "Windows":
        sys.path.insert(0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-4/src')
    elif system == "Darwin":
        sys.path.insert(0, '/Users/lorenzogurrola/workspace/github.com/LorenzoGurrola/Project-4/src')
    from data_loader import prepare_train, prepare_test

class NeuralNetwork(nn.Module):
    def __init__(self, n, h):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(n, h)
        self.l2 = nn.Linear(h, 1)
    
    def forward(self, X):
        A1 = F.relu(self.l1(X))
        A2 = torch.sigmoid(self.l2(A1))
        return A2

def load_data():
    data = pd.read_csv('../framingham.csv')
    data = data.dropna()
    train, test = train_test_split(data, train_size=0.85, random_state=10)
    X_train, y_train, scalers = prepare_train(train)
    X_test, y_test = prepare_test(test, scalers)
    return X_train, y_train, X_test, y_test

def initialize_params(n, h):
    W1 = np.random.randn(n, h) * 0.1
    b1 = np.zeros((1, h))
    w2 = np.random.randn(h, 1) * 0.1
    b2 = np.zeros((1, 1))
    params = {'W1':W1, 'b1':b1, 'w2':w2, 'b2':b2}
    param_count = n * h + 2 * h + 1
    print(f'initialized {param_count} total trainable params with {h} hidden units and {n} input features')
    return params

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def forward(X, params):
    W1 = params['W1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    z2 = A1 @ w2 + b2
    a2 = sigmoid(z2)

    return a2