import numpy as np
from numbers import Real
from typing import Callable

"""Partie A"""
def f(x: Real):
    return x ** 2

def numerical_gradient(f: Callable, x: Real, epsilon: Real=0.0001):
    return (f(x + epsilon) - f(x)) / epsilon

print(f"numerical_gradient(f, 3, 0.0001) = {numerical_gradient(f, 3, 0.0001)}")

"""Partie B"""
def test_lr(x_0: Real, f: Callable, lr: Real = 0.1) -> None:
    x = x_0

    print(f"\ni_0: {x, f(x)}")
    for i in range(1, 11):
        x = x - numerical_gradient(f, x) * lr
        print(f"i_{i}: {x, f(x)}")

test_lr(3.0, f)

"""Partie C"""
test_lr(3.0, f, lr=0.01)
test_lr(3.0, f, lr=0.5)

# On constate que plus lr est important plus la vitesse de convergence est élevée. A l'inverse, plus lr est faible, plus la précision de convergence est importante.

"""Partie D"""
# 1) Si le gradient vaut 0, alors un minimum (local ou non) est atteint
# 2) On soustrait le gradient pour minimiser l'écart à la valeur cible, additionner aurait l'effet inverse
# 3) Le learning rate permet de régler le rapport entre vitesse et précision de la convergence