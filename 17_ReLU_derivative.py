import numpy as np

Z = np.array([
    [-2.,  3.],
    [ 4., -1.],
    [-5.,  6.]
])

dA = np.array([
    [10., 20.],
    [30., 40.],
    [50., 60.]
])

"""Partie A"""
# ReLU'(Z) =
# [0    1]
# [1    0]
# [0    1]

"""Partie B"""
# dZ = dA * ReLU'(Z) =
# [0    20]
# [30   0]
# [0    60]

"""Partie C"""
dZ = dA * (Z > 0)

print(f"dZ {dZ.shape}: \n{dZ}")

"""Partie D"""
# Une valeur négative de Z bloque le gradient puisqu'il sera erroné en raison de l'action de ReLU.
# En revanche, lorsque la valeur de Z est positive, le calcul du gradient ne change pas, il peut donc être utilisé.