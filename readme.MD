# Neural Network from Scratch

A minimal neural network implemented in Python using only NumPy. Built from the ground up to understand forward pass, backpropagation, and the role of activations like ReLU and Sigmoid.

## Files

- `activations.py` - ReLU and Sigmoid classes with forward and backward methods.
- `layer.py` - class that holds weights, biases, and applies backpropagation.
- `network.py` - Network class that chains layers and orchestrates training.

## Status

- Forward propagation: done
- Backpropagation (ReLU, Sigmoid): done
- Loss functions: planned
- Training on real datasets: planned

## Usage (later)

```python
from network import Network
from layer import Layer
from activations import ReLU, Sigmoid

net = Network([
    Layer(2, 4, ReLU()),
    Layer(4, 1, Sigmoid())
])
```
