import numpy as np

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(Z, 0)


    def backward(self, dA):
        return dA * (self.Z > 0)
        
class Sigmoid:
    def forward(self, Z):
        self.a = 1 / (1 + np.exp(-Z))
        return self.a


    def backward(self, dA):
        return dA * self.a * (1 - self.a)