import numpy as np

"""Partie A"""
# A: oui, (3, 4)
# B: non
# C: oui, (3, 4)
# D: non
# E: oui, (3, 4)
# F: non

"""Partie B"""
A_0 = np.zeros((3, 4))
A_1 = np.zeros(4)
A = A_0 + A_1

# Flemme de tous les faire c'est la meme chose...

"""Partie C"""
X = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

b = np.array([10., 20., 30.])

Y = X + b
# On a X.shape = (2, 3) et b.shape = (3,). Numpy applique donc le broadcasting suivant:
# [1+10     2+20        3+30]
# [4+10     5+20        6+30]

"""Partie D"""
X = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

W = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

b = np.array([10., 20.])

Z = X @ W + b
# X.shape = (2, 3)
# W.shape = (3, 2)
# (X @ W).shape = (2, 2)
# b.shape = (2,)
# Z.shape = (2, 2)

# Z =
# [32   48]
# [56   84]

"""Partie E"""
# numpy compare de droite à gauche pour vérifier que le broadcasting soit valide. Or, 3 != 4, il lévera donc une erreur.