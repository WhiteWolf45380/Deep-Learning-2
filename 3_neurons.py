import numpy as np

"""Partie A"""

x = np.array([2, 3, 5])
w = np.array([0.1, 0.5, -0.2])
b = 0.3

z = x @ w + b
print(f"z: \n{z}")

# Prédiction de z:
# z = 0.2 + 1.5 - 1 + 0.3 = 1.0

"""Partie B"""
x = np.array([2, 3, 5])
w = np.array([
    [0.1,   0.2],
    [0.5,   -0.3],
    [-0.2,  0.4],
])
b = [0.3, -0.1]

z = x @ w + b
print(f"\nz {z.shape}: \n{z}")

# Prédiction de z.shape:
# (2,)

"""Partie C"""
x = np.array([
    [2, 3,  5],
    [1, 4,  2],
])

z = x @ w + b
print(f"\nz {z.shape}: \n{z}")

# Prédiction de z.shape:
# (2, 2)

"""Partie D"""
# x : données d'entrée du réseau. Ici n vecteurs à 3 dimensions.
# w : ensemble des poids des neuronnes
# b : ensemble des biais des neuronnes
# z : ensemble des sorties de neuronnes