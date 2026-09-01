import numpy as np

X = np.array([
    [1., 2.],
    [2., 1.],
    [3., 5.],
    [4., 2.]
])

W = np.zeros((2, 2))
b = np.zeros(2)

lr = 0.01

"""Partie A"""

Y = np.zeros((4, 2))
Y[:, 0] = 2 * X[:, 0] + 3 * X[:, 1]
Y[:, 1] = -X[:, 0] + 4 * X[:, 1]

# Y_00 = 2 * X_00 + 3 * X_01 = 2 * 1 + 3 * 2 = 8
# <=> correct

print(f"Y {Y.shape}: \n{Y}")

"""Partie B"""
Z = X @ W + b

E = Z - Y
L = np.mean(E**2)

print(f"\nW(0): \n{W}")
print(f"b(0): \n{b}")
print(f"loss(0): {L}")

dZ = 2 * E / Z.size
dW = X.T @ dZ
db = np.sum(dZ, axis=0)

W = W - dW * lr
b = b - db * lr

Z = X @ W + b
E = Z - Y
L = np.mean(E**2)

print(f"\nW(1): \n{W}")
print(f"b(1): \n{b}")
print(f"loss(1): {L}")

"""Partie C"""
def train(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.ndarray, lr: float = 0.01, epochs: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    for i in range(epochs):
        Z = X @ W + b

        E = Z - Y
        L = np.mean(E**2)

        dZ = 2 * E / Z.size
        dW = X.T @ dZ
        db = np.sum(dZ, axis=0)

        W = W - dW * lr
        b = b - db * lr

        if i % 100 == 0:
            print(f"loss ({i}): {L}")

    print(f"loss ({epochs}): {L}")

    return W, b

print()
W, b = train(X, Y, W, b)
print(f"\nW(final): \n{W}")
print(f"b(final): \n{b}")