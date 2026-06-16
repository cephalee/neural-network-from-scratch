import numpy as np

def categorical_cross_entropy(y_pred, y_true, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred))
    dA = -(y_true / y_pred)
    return loss, dA