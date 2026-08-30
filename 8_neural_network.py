import numpy as np

"""Partie A"""
x = np.array([1., 2., 3., 4.])
y_true = np.array([2., 4., 6., 8.])
w = 0.0
b = 0.0
lr = 0.01

y_pred = x * w + b
# y_pred = [0, 0, 0, 0]
print(f"y_pred: {y_pred}")

loss = np.mean((y_true - y_pred)**2)
# loss = (4 + 16 + 36 + 64) / 4 = 30
print(f"loss: {loss}")

"""Partie B"""
error = y_pred - y_true

gradient_w = 2 * np.mean(error * x)
print(f"\ngradient_w: {gradient_w}")

gradient_b = 2 * np.mean(error)
print(f"gradient_b: {gradient_b}")

"""Partie C et D"""
def train(x: np.ndarray, y_true: np.ndarray, w: float, b: float, lr: float, epochs: int = 100) -> tuple[float, float]:
    print()
    for i in range(1, epochs + 1):
        y_pred = x * w + b
        loss = np.mean((y_true - y_pred)**2)
        error = y_pred - y_true
        gradient_w = 2 * np.mean(error * x)
        gradient_b = 2 * np.mean(error)
        w = w - gradient_w * lr
        b = b - gradient_b * lr

        if i % 10 == 0:
            print(i, loss)

    return w, b

w, b = train(x, y_true, w, b, lr, 100)
print(f"\n(w, b): ({w, b})")

"""Partie E"""
y_pred = x * w + b
print(f"\nexpected: {y_true}")
print(f"got: {y_pred}")