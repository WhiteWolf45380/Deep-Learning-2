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
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0]
])

Z = X @ W + b
E = Z - Y
dZ = 2 * E / Z.size

# Z =