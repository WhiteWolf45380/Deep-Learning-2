import numpy as np

"""Partie A"""
x = np.arange(1, 5)

print(x.shape)

x_1_4 = np.array([
    [1, 2, 3, 4]
])

x_4_1 = np.array([
    [1],
    [2],
    [3],
    [4],
])

print(f"\nx {x.shape}: \n{x}")
print(f"\nx_4_1 {x_4_1.shape}: \n{x_4_1}")
print(f"\nx_1_4 {x_1_4.shape}: \n{x_1_4}")

"""Partie B"""
x = np.array([1, 2, 3])
W = np.arange(1, 7).reshape(3, 2)

y = x @ W

print(f"\ny {y.shape}: \n{y}")

# Shape attendue:
# (2,)
# (1, 2) était-il correct aussi ou pas ?

"""Partie C"""
X = np.arange(1, 10).reshape(3, 3)
W = np.arange(1, 7).reshape(3, 2)

Y = X @ W

print(f"\nY {Y.shape}: \n{Y}")

# Prédiction de X.shape:
# (3, 3)

# Prédiction de W.shape:
# (3, 2)

# Prédiction de (X @ W).shape:
# (3, 2)

"""Partie D"""
# A : oui
# B : non
# C : oui
# D : non