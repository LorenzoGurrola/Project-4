import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

if True:
    import sys
    system = platform.system()
    if system == "Windows":
        sys.path.insert(
            0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-4/src')
    elif system == "Darwin":
        sys.path.insert(
            0, '/Users/lorenzogurrola/workspace/github.com/LorenzoGurrola/Project-4/src')
    from train_model import initialize_params, forward, NeuralNetwork, backward


class test_forward(unittest.TestCase):

    def test_basic(self):
        m = 50
        n = 18
        h = 3

        X = np.random.randn(m, n)
        params = initialize_params(n, h)
        expected, inter_vals = forward(X, params)

        model = NeuralNetwork(n, h)
        model.l1.weight = nn.Parameter(torch.tensor(params['W1'].T))
        model.l1.bias = nn.Parameter(torch.tensor(params['b1']))
        model.l2.weight = nn.Parameter(torch.tensor(params['w2'].T))
        model.l2.bias = nn.Parameter(torch.tensor(params['b2']))
        X = torch.tensor(X)
        result = model.forward(X).detach().numpy()

        np.testing.assert_allclose(result, expected)


class test_forward_backward(unittest.TestCase):

    def test_basic(self):
        m = 50
        n = 40
        h = 30

        X = np.random.randn(m, n)
        y = np.random.randint(0, 2, size=(m, 1))
        params = initialize_params(n, h)
        yhat, inter_vals = forward(X, params)
        grads = backward(y, yhat, inter_vals, X, params)
        dW1_result = grads['dW1']
        db1_result = grads['db1']
        dw2_result = grads['dw2']
        db2_result = grads['db2']

        model = NeuralNetwork(n, h)
        model.l1.weight = nn.Parameter(torch.tensor(params['W1'].T))
        model.l1.bias = nn.Parameter(torch.tensor(params['b1']))
        model.l2.weight = nn.Parameter(torch.tensor(params['w2'].T))
        model.l2.bias = nn.Parameter(torch.tensor(params['b2']))
        X = torch.tensor(X)
        y = torch.tensor(y).double()
        yhat = model.forward(X)
        loss = model.calculate_loss(yhat, y)
        model.backward(loss)
        dW1_expected = model.l1.weight.grad.detach().numpy()
        db1_expected = model.l1.bias.grad.detach().numpy()
        dw2_expected = model.l2.weight.grad.detach().numpy()
        db2_expected = model.l2.bias.grad.detach().numpy()

        np.testing.assert_allclose(dW1_result, dW1_expected.T)
        np.testing.assert_allclose(db1_result, db1_expected)
        np.testing.assert_allclose(dw2_result, dw2_expected.T)
        np.testing.assert_allclose(db2_result, db2_expected)


unittest.main()
