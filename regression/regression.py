# -*- coding: utf-8 -*-
#------------------------------------
# @Time          : 2019.10.26
# @Author        : Su
#------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random

def hypothesis(x, w, b):
    """
    hypothesis function
    x.shape = [1, m](n为特征数，m为样本数)
    w.shape = [1, 1]
    b.shape = [1, m]
    y.shape = [1, m]
    """
    return w * x + b

def linear_propagate(x, y, y_pre):
    cost = np.sum((y_pre - y) ** 2) / (2 * len(x))

    dw = np.sum(np.dot(x, (y_pre-y).T))/len(x)
    db = np.sum(y_pre-y)/len(x)
    grads = {"dw":dw,
             "db":db}
    return cost, grads

def linear_train(x, y, alpha=0.001, iterations=1000):
    w = np.zeros([1, 1])
    b = 0
    costs = []

    for i in range(iterations):
        y_pre = hypothesis(x, w, b)

        cost, grads = linear_propagate(x, y, y_pre)
        dw = grads["dw"]
        db = grads["db"]

        w = w - alpha * dw
        b = b - alpha * db

        costs.append(cost)

    return w, b, costs

def data(num):
    X = []
    Y = []

    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()

    for i in range(num):
        x = random.randint(0, 100) * random.random()
        y = w* x + b + random.random() * random.randint(-1, 100)

        X.append(x)
        Y.append(y)

    return X, Y

if __name__ == '__main__':
    x, y = data(100)
    plt.scatter(x, y)

    w, b, costs = linear_train(x, y)
    y_hat = (w*x+b).reshape((len(x),))
    plt.plot(x, y_hat, color='g')
    plt.show()






