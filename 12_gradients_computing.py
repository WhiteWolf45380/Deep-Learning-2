import numpy as np

"""Partie A"""
X = np.array([
    [1., 2.],
    [3., 4.]
])

w = np.array([0.1, 0.2])
b = 0.5

Y = np.array([1., 2.])

Z = X @ w + b       # (2,)
# Z = [1.0  1.6]

E = Z - Y           # (2,)
# E = [0.0  -0.4]

L = np.mean(E**2)   # ()
# L = 0.08

dZ = 2 * E / len(Z) # (2,)
# dZ = [0.0, -0.4]

# dL / dw2 = (dL / dZ) * (dZ / dw2)
#          = (dL / dZ_1) * (dZ_1 / dw2) + (dL / dZ_2) * (dZ_2 / dw2)
#          = (2/n) * (Z_1 - Y_1) * X_12 + (2/n) * (Z_2 - Y_2) * X_22
#          = 1 * 0.0 * 2 + 1 * (-0.4) * 4
#          = -1.6

"""Partie C"""
gradient_w = np.array([-1.2, -1.6])
dw = X.T @ dZ

print(f"Is gradient_w correct: {np.allclose(gradient_w, dw)}")  # True

"""Partie D"""
# dL / db = (dL / dZ) * (dZ / db)
#         = (dL /dZ_1) * (dZ_1 / db) + (dL / dZ_2) * (dZ2 / db)
#         = (2/n) * (Z_1 - Y_1) * 1 + (2/n) * (Z_2 - Y_2) * 1
#         = 1 * 0.0 * 1 + 1 * (-0.4) * 1
#         = -0.4

gradient_b = -0.4
db = np.sum(dZ, axis=0)

print(f"Is gradient_b correct: {np.allclose(gradient_b, db)}")  # True

"""Partie E"""
lr = 0.001

w = w - gradient_w * lr
b = b - gradient_b * lr

Z = X @ w + b

E = Z - Y
L = np.mean(E**2)

print("new loss:", L)