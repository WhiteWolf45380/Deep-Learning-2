import numpy as np
from numpy.typing import NDArray
import importlib

module = importlib.import_module("19_Dense")
Dense = module.Dense

module = importlib.import_module("20_ReLU")
ReLU = module.ReLU

"""Partie A"""

sequential = [
    Dense(2, 3),
    ReLU(),
    Dense(3, 1),
]

X = np.array([
    [1., 2.],
    [3., 4.]
])

def sequential_forward(sequential: list[Dense | ReLU], X: NDArray) -> NDArray:
    print()
    for layer in sequential:
        X = layer.forward(X)
        print(X.shape)
    return X

print()
print(sequential_forward(sequential, X))

"""Partie B"""
def sequential_backward(sequential: list[Dense | ReLU], dZ: NDArray) -> NDArray:
    print()
    for layer in reversed(sequential):
        dZ = layer.backward(dZ)
        print(dZ.shape)
    return dZ

dZ2 = np.array([
    [10.],
    [20.]
])

print()
print(sequential_backward(sequential, dZ2))
