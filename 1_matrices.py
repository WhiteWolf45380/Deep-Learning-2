import numpy as np

A = np.arange(1, 5).reshape(2, 2)
B = np.arange(5, 9).reshape(2, 2)

print(f"\nA + B : \n{A + B}")
print(f"\nA * B : \n{A * B}")
print(f"\nA @ B : \n{A @ B}")
print(f"\nShape(A) : {A.shape}")
print(f"Shape(B) : {B.shape}")

# Prédiction de A @ B:
# [19   22]
# [43   50]

A = np.arange(1, 7).reshape(2, 3)
B = np.random.rand(2, 5)
print(f"\nA @ B : {A @ B}")

# Prédiction de (A@B).shape :
# ça va pas fonctionné puisqu'il faut que A ait autant de colonnes que B a de lignes