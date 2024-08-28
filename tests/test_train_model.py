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
        sys.path.insert(0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-4/src')
    elif system == "Darwin":
        sys.path.insert(0, '/Users/lorenzogurrola/workspace/github.com/LorenzoGurrola/Project-4/src')
    from train_model import initialize_params, forward_prop, NeuralNetwork



class test_forward_prop(unittest.TestCase):

    def test_basic(self):
        m = 50
        n = 18
        h = 3
        
        X = np.random.randn(m, n)
        params = initialize_params(n, h)
        expected = forward_prop(X, params)

        model = NeuralNetwork(n, h)
        model.l1.weight = nn.Parameter(torch.tensor(params['W1'].T))
        model.l1.bias = nn.Parameter(torch.tensor(params['b1']))
        model.l2.weight = nn.Parameter(torch.tensor(params['W2'].T))
        model.l2.bias = nn.Parameter(torch.tensor(params['b2']))
        X = torch.tensor(X)
        result = model.forward(X).detach().numpy()

        np.testing.assert_allclose(result, expected)

        


unittest.main()
