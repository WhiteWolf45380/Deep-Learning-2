import numpy as np

"""Partie A"""
X = np.array([
    [1., 2., 3.],
    [2., 1., 4.],
    [3., 5., 1.],
    [4., 2., 2.]
])

y_true = np.array([5., 3., 19., 8.])
w = np.zeros(3)
b = 0.0

y_pred = X @ w + b
# y_pred = [0, 0, 0, 0]
print(f"y_pred {y_pred.shape}: {y_pred}")

loss = np.mean((y_true - y_pred) ** 2)
# loss = 114.75
print(f"loss: {loss}")

"""Partie B"""
error = y_pred - y_true

gradient_w = 2 * X.T @ error / len(X)
gradient_b = 2 * np.mean(error)

# X.shape = (4, 3)
# w.shape = (3,)
# y_pred.shape = (4,)
# error.shape = (4,)
# gradient_w.shape = (3,)
# gradient_b.shape = ()

print(f"\nX.shape: {X.shape}"
      f"\nw.shape: {w.shape}"
      f"\ny_pred.shape: {y_pred.shape}"
      f"\nerror.shape: {error.shape}"
      f"\ngradient_w.shape: {gradient_w.shape}"
      f"\ngradient_b.shape: {gradient_b.shape}"
)

"""Partie C"""
def train(X: np.ndarray, y_true: np.ndarray, w: float, b: float, lr: float = 0.01, epochs: int = 1000) -> tuple[float, float]:
    print()
    for i in range(epochs):
        y_pred = X @ w + b

        loss = np.mean((y_true - y_pred)**2)
        error = y_pred - y_true

        gradient_w = 2 * X.T @ error / len(X)
        gradient_b = 2 * np.mean(error)

        w = w - gradient_w * lr
        b = b - gradient_b * lr

        if i % 100 == 0:
            print(loss)

    print(loss)

    return w, b

"""Partie D"""
w, b = train(X, y_true, w, b)
print(f"\n(w, b): {list(map(float, w)), float(b)}")