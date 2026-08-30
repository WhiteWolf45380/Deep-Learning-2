import numpy as np
from numbers import Real

"""Partie A"""
x = 2.0
y_true = 10.0
w = 3.0
b = 0.0

y_pred = x * w + b
# y_pred = 2 * 3 + 0 = 6
print(f"y_pred: {y_pred}")

loss = (y_true - y_pred)**2
# loss = (10 - 6)**2 = 16
print(f"loss: {loss}")

gradient_w = 2 * (y_pred - y_true) * x
# gradient_w = 2 * (6 - 10) * 2 = -16
print(f"gradient_w: {gradient_w}")

gradient_b = 2 * (y_pred - y_true)
# gradient_b = 2 * (6 - 10) = -8
print(f"gradient_b: {gradient_b}")

"""Partie b"""
learning_rate = 0.1
w = w - learning_rate * gradient_w
b = b - learning_rate * gradient_b

y_pred = x * w + b
print(f"\ny_pred: {y_pred}")

loss = (y_true - y_pred)**2
print(f"loss: {loss}")

"""Partie C"""
x = 2.0
w = 3.0
b = 0.0
learning_rate = 0.01

print()
for i in range(20):
    y_pred = x * w + b
    loss = (y_true - y_pred)**2
    gradient_w = 2 * (y_pred - y_true) * x
    gradient_b = 2 * (y_pred - y_true)
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b
    print(loss)

"""Partie D"""
def train(x: Real, y_true: Real, w: float, b: float, learning_rate: float, epochs: int) -> None:
    print()
    for i in range(epochs):
        y_pred = x * w + b
        loss = (y_true - y_pred)**2
        gradient_w = 2 * (y_pred - y_true) * x
        gradient_b = 2 * (y_pred - y_true)
        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b
        print(i, loss)
    return w, b
        
x = 2.0
y_true = 10.0
w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 20
w, b = train(x, y_true, w, b, learning_rate, epochs)
print(f"\n(w, b): ({w}, {b})")