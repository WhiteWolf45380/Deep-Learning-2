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
loss = np.mean(E ** 2)

print(
    f"Z {Z.shape}: \n{Z}\n\n"
    f"E {E.shape}: \n{E}\n\n"
    f"loss {loss.shape}: \n{loss}\n\n"
)

"""Partie B"""
dZ = 2 * E / Z.size
dW = X.T @ dZ
db = np.sum(dZ, axis=0)

print(
    f"Z.shape: {Z.shape}\n"
    f"E.shape: {E.shape}\n"
    f"dZ.shape: {dZ.shape}\n"
    f"W.shape: {W.shape}\n"
    f"dW.shape: {dW.shape}\n"
    f"b.shape: {b.shape}\n"
    f"db.shape: {db.shape}\n"
)

"""Partie C"""
dW_manual = np.zeros_like(W)

for i, row in enumerate(X.T):
    for j, col in enumerate(dZ.T):
        dW_manual[i][j] = np.dot(row, col)

print("Is dW_manual correct:", np.allclose(dW, dW_manual))

"""Partie D"""
lr = 0.1
W = W - lr * dW
b = b - lr * db

Z = X @ W + b
loss = np.mean((Z - Y)**2)
print("new loss:", loss)

"""Bonus"""
X = np.arange(1, 5).reshape(2, 2)
w = np.array([0.1, 0.2])
b = 0.5

Y = np.array([1, 2])

Z = X @ w + b
# Z = [1*0.1+2*0.2+0.5      3*0.1+4*0.2+0.5] = [1.0     1.6]

error = Z - Y
# error = [1.0-1.0    1.6-2.0] = [0.0  -0.4]

loss = np.mean(error**2)
# loss = (0.0 + 0.16) / 2 = 0.08

# dL / dw1 = (dL / dZ) * (dZ / dw1)
#          = (dL / dZ_1) * (dZ_1 / dw1) + (dL / dZ_2) * (dZ_2 / dw1)
#          = (2/n) * (Z_1 - Y_1) * X_11 + (2/n) * (Z_2 - Y_2) * X_21
#          = 0 - 0.4 * 3
#          = -1.2
