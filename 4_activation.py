import numpy as np

"""Partie A"""
x = np.array([-5, -2, -1, 0, 1, 3, 7])
x_relu = np.maximum(0, x)

print(f"ReLU(x): \n{x_relu}")

"""Partie B"""
z = np.array([-3, 2, -1, 4, -5, 6, -2, 0, 8]).reshape(3, 3)
z_relu = np.maximum(0, z)

print(f"\nReLU(z): \n{z_relu}")
print(f"\nz.shape == ReLU(z).shape: {z.shape == z_relu.shape}")

"""Partie C"""
X = np.array([
    [2, 3, 5],
    [1, 4, 2]
])

W = np.array([
    [0.1,  0.2],
    [0.5, -0.3],
    [-0.2, 0.4]
])

b = np.array([0.3, -0.1])

Z = X @ W + b
A = np.maximum(0, Z)

print(f"\nZ {Z.shape}: \n{Z}")
print(f"\nA {A.shape}: \n{A}")

# Prédiction de Z:
# [1.0  1.4]
# [2.0  -0.3]

"""Partie D"""
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

x = np.array([-10, 0, 10])
x_sigmoid = sigmoid(x)

print(f"\nSigmoid(x): \n{x_sigmoid}")