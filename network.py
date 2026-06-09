import numpy as np

class Network:
    def __init__(self, layers):
        self.network = layers

    def forward(self, x):
        for layer in self.network:
            x = layer.forward(x)
        return x

    def backward(self, lr, dA):
        for layer in reversed(self.network):
            dA = layer.backward(lr, dA)