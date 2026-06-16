
import numpy as np
from activation import ReLU, Softmax
from layer import Layer
from loss import categorical_cross_entropy
from network import Network
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

def prepare(X, y):
    X = X / 255
    y = np.eye(10)[y.astype(int)]
    model = Network([
        Layer(784, 128, ReLU()),
        Layer(128, 64, ReLU()),
        Layer(64, 10, Softmax())
    ])
    return X, y, model 

def train(model, X_train, Y_train, epochs=10, lr=0.01):
    n = len(X_train)
    idx = np.arange(n)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(idx)
        total_loss = 0.0

        for i in idx:
            pred = model.forward(X_train[i])
            loss, dA = categorical_cross_entropy(pred, Y_train[i])
            total_loss += loss
            model.backward(lr, dA)
        print(f"Epoch {epoch}/{epochs} - loss: {total_loss / n:.4f}")


X, y, model = prepare(X, y)
train(model, X, y, epochs=10, lr=0.01)

