import numpy as np

"""Partie A"""
X = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

W = np.array([
    [0.1, 0.3],
    [0.2, 0.4]
])

b = np.array([0.5, 0.7])

Y = np.array([
    [1., 2.],
    [2., 3.],
    [3., 4.]
])

Z = X @ W + b
A = np.maximum(0, Z)

E = A - Y
L = np.mean(E**2)

print(f"loss: {L}")

"""Partie B"""
dA = 2 * E / A.size

"""Partie C"""
dZ = dA * (Z > 0)

"""Partie D"""
dW = X.T @ dZ
db = np.sum(dZ, axis=0)

"""Partie E"""
lr = 0.01

W = W - dW * lr
b = b - db * lr

Z = X @ W + b
A = np.maximum(0, Z)

L = np.mean((A - Y)**2)

print(f"new loss: {L}")