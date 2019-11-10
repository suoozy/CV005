# -*- coding: utf-8 -*-
#------------------------------------
# @Time          : 2019.11.2
# @Author        : Su
#------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize(x):
    w = np.random.normal(loc=0, scale=0.5, size=(x.shape[0],1))
    b = 0

    assert w.shape == (x.shape[0], 1)
    assert isinstance(b, float) or isinstance(b, int)

    return w,b

def hypothesis(x, w, b):
    """
    hypothesis function
    x.shape = [n, m](n为特征数，m为样本数)
    w.shape = [n, 1]
    b.shape = [1, m]
    y.shape = [1, m]
    """
    return sigmoid(np.dot(w.T, x) + b)

def propagate(x, y, y_hat):
    cost = np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))/(-len(y))

    dz = sigmoid(y_hat) - y
    dw = np.dot(x, dz.T)/len(y)
    db = np.sum(dz)/len(y)

    grads = {"dw":dw,
             "db":db}

    return cost, grads

def train(x, y, alpha=0.01, iterations=1000):
    w, b = initialize(x)
    costs = []

    for i in range(iterations):
        y_hat = hypothesis(x, w, b)
        cost, grads = propagate(x, y, y_hat)

        dw = grads["dw"]
        db = grads["db"]

        w -= alpha*dw
        b -= alpha*db

        costs.append(cost)

    return w, b, costs


if __name__ == '__main__':
    num = 200
    x, y = make_classification(n_samples=num, n_features=2, n_redundant=0)
    x1 = x.T[0, :]
    x2 = x.T[1, :]

    plt.scatter(x1, x2, marker='*', c=y)

    w, b, costs = train(x.T, y.reshape(1, num))
    x2 = (-w[0, :] * x1 - b)/w[1, :]

    plt.plot(x1, x2)

    plt.show()



