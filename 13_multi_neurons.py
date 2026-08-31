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

# X.shape = (3, 2)      3 observations * 2 features
# W.shape = (2, 2)      2 features * 2 neurones
# b.shape = (2,)        2 neurones
# Z.shape = (3, 2)      3 observations * 2 neurones

"""Partie B"""
# Z_11 = X_11 * W_11 + X_12 * W_21 + b_1 = 1 * 0.1 + 2 * 0.2 + 0.5 = 1.0
# Z_12 = X_11 * W_12 + X_12 * W_22 + b_2 = 1 * 0.3 + 2 * 0.4 + 0.7 = 1.8

"""Partie C"""
Z = X @ W + b
print(f"Z: \n{Z}")
# correct

"""Partie D"""
# 1 ligne de x: une observation
# 1 colonne de W: poids d'un neurone
# 1 ligne de Z: sorties des neurones pour 1 observations
# 1 colonne de Z: sorties d'un neurone pour toutes les observations