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
    
class Softmax:
    def forward(self, Z):
        Z -= np.max(Z)
        self.S = np.exp(Z) / np.sum(np.exp(Z))
        return self.S
    
    def backward(self, dA):
        J = np.diag(self.S) - np.outer(self.S, self.S)
        return J @ dA 