import numpy as np


def binary_cross_entropy(y_pred, y_true, eps=1e-12):
    
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    dA = (y_pred - y_true) / len(y_true)
    return loss, dA