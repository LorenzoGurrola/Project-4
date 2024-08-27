import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import platform

if True:
    import sys
    system = platform.system()
    if system == "Windows":
        sys.path.insert(0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-3/src')
    elif system == "Darwin":
        sys.path.insert(0, '/Users/lorenzogurrola/workspace/github.com/LorenzoGurrola/Project-3/src')
    from train_model import forward_prop, calculate_cost, back_prop


class test_forward_and_back_prop(unittest.TestCase):

    def test_basic(self):
        m = 100
        n = 100
        X = np.random.randn(m,n)
        w = np.random.rand(n, 1) * 0.1
        b = np.zeros((1,1))
        y = np.random.randint(0, 2, size=(m, 1))
        params = {'w':w,'b':b}
        yhat, inter_vals = forward_prop(X, params)
        cost = calculate_cost(yhat, y)
        grads = back_prop(y, yhat, inter_vals, X)
        dw_result = grads['dw']
        db_result = grads['db']

        X = torch.tensor(X)
        y = torch.tensor(y)
        w = torch.tensor(w, requires_grad=True)
        b = torch.tensor(b, requires_grad=True)
        z = X @ w + b
        yhat = 1/(1 + torch.exp(-z))
        losses = (y * torch.log(yhat)) + ((1 - y) * torch.log(1 - yhat))
        cost = -torch.sum(losses, dim=0, keepdims=True)/m
        cost.backward()
        dw_expected = w.grad.detach().numpy()
        db_expected = b.grad.detach().numpy()
        
        np.testing.assert_allclose(dw_result, dw_expected)
        np.testing.assert_allclose(db_result, db_expected)


unittest.main()
