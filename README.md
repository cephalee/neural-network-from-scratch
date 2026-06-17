# Neural Network from Scratch

A fully connected neural network implemented from scratch using only NumPy, trained on the MNIST dataset.

---

## Results

**97.45% accuracy** on the MNIST test set (10,000 images) after 10 epochs.

---

## Architecture

```
Input (784) → Dense (128, ReLU) → Dense (64, ReLU) → Output (10, Softmax)
```

## What's implemented

- Forward and backward pass without any ML framework
- ReLU, Sigmoid and Softmax activations with analytical gradients
- Categorical cross-entropy loss
- Stochastic gradient descent (SGD)
- Modular design — layers, activations and loss are fully decoupled

---

## Stack

- NumPy only
- scikit-learn for dataset loading only

---

## File structure

```
├── activation.py   # ReLU, Sigmoid, Softmax
├── layer.py        # Dense layer with forward/backward
├── loss.py         # Categorical cross-entropy
├── network.py      # Network class
└── main.py         # Training and evaluation
```

---

## Run

```bash
pip install numpy scikit-learn
python main.py
```
