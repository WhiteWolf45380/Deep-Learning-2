import numpy as np

X = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

W = np.array([
    [1., -2.],
    [2.,  1.]
])

b = np.array([0.5, -1.])

"""Partie A"""
Z = X @ W + b   # (3, 2)

# Z =
# [5.5      -1]
# [11.5     -3]
# [17.5     -5]

"""Partie B"""
A = np.maximum(0, Z)

"""Partie C"""
# Z_ij : sortie brut du neurone j pour l'observation i
# A_ij : sortie passée par une fonction d'activation du neurone j pour l'observation i
# Z et A ont la même shape puisque Z ne fait que modifier les valeurs de sortie pour introduire une non linéarité, il n'en crée pas et n'en supprime pas.

"""Partie D"""
# Les valeurs de Z modifiées par ReLU sont : [-1, -3, -5]