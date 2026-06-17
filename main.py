
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
    return model

def predict(model, image):
    probs = model.forward(image)
    res = np.argmax(probs)
    print(f"Result : {res} - accuracy {probs[res]:.4f}")
    return probs[res]

def evaluate(model, X_test, y_test):
    correct = 0
    for i in range(len(X_test)):
        res = np.argmax(model.forward(X_test[i]))
        if res == np.argmax(y_test[i]):
            correct += 1
    print(f"Accuracy : {correct / len(X_test):.4f}")

X, y, model = prepare(X, y)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

model = train(model, X_train, y_train)
evaluate(model, X_test, y_test)
