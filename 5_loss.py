import numpy as np

"""Partie A"""
y_true = np.array([10, 5, 2])
y_pred = np.array([8, 7, 3])

loss = np.mean((y_true - y_pred)**2)
print(f"loss: {loss}")

# Prédiction des erreurs
# (2, -2, -1)

# Prédiction des carrés
# (4, 4, 1)

# Prédiction de la moyenne
# 3

"""Partie B"""
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred)**2)

print(f"\nCohérence de la loss: {mse(y_true, y_pred) == loss}")

"""Partie C"""
y_true = np.array([10, 5, 2, 8])
pred_A = np.array([9, 6, 3, 7])
pred_B = np.array([4, 8, 1, 10])

print(f"\nMSE(A): {mse(y_true, pred_A)}")
print(f"MSE(B): {mse(y_true, pred_B)}")

# Prédiction des losses
# MSE(A) = 1.0
# MSE(B) = 12.5
# Meilleur modèle : A

"""Partie D"""
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

y_true = np.array([
    [1.5, 1.0],
    [2.0, 0.0]
])

Z = X @ W + b
A = np.maximum(0, Z)
loss = mse(y_true, A)

print(f"\nloss: {loss}")