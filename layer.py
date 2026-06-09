import numpy as np

class Layer:
    def __init__(self, n_in, n_out, activation):
        self.W = np.random.randn(n_out, n_in) * 0.1
        self.B = np.zeros(n_out)
        self.activation = activation

    def forward(self, x):
        self.input = x
        Z = self.W @ x + self.B
        return self.activation.forward(Z)
    
    def backward(self, lr, dA):
        dZ = self.activation.backward(dA)
        dW = np.outer(dZ, self.input)
        dB = np.sum(dZ ,axis=0)
        self.W -= lr * dW
        self.B -= lr * dB
        dOut = self.W.T @ dZ
        return dOut

