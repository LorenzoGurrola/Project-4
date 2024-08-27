from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def load_data():
    std_dev = 1
    mean = 5
    x1 = np.linspace(1, 9, 100)
    x2 = np.linspace(3, 11, 100)
    y = ((x1 > 6) & (x2 > 9)).astype(int)
    X = np.stack((x1, x2), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=10)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    return X_train, X_test, y_train, y_test

def initialize_params(X):
    n = X.shape[1]
    w = np.random.randn(n, 1) * 0.1
    b = np.zeros((1, 1))
    params = {'w':w, 'b':b}
    return params

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def forward_prop(X, params):
    w = params['w']
    b = params['b']
    z = X @ w + b
    inter_vals = {'z':z}
    yhat = sigmoid(z)
    return yhat, inter_vals

def calculate_cost(yhat, y):
    m = y.shape[0]
    losses = y * np.log(yhat) + (1 - y) * np.log(1 - yhat)
    cost = -np.sum(losses, axis=0, keepdims=True)/m
    return cost

def back_prop(y, yhat, inter_vals, X):
    m = y.shape[0]
    dc_dyhat = (-1/m) * ((y/yhat) - ((1-y)/(1-yhat)))
    dyhat_dz = yhat * (1 - yhat)
    dc_dz = dc_dyhat * dyhat_dz
    dc_dw = np.matmul(X.T, dc_dz)
    dc_db = np.sum(dc_dz, axis=0, keepdims=True)
    grads = {'dw':dc_dw, 'db':dc_db}
    return grads