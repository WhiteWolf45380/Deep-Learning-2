import numpy as np
from .abc import Layer

class ReLU(Layer):
    """Couche d'activation
    
    Applique un filtre *max(0, X_ij)* sur les entrées
    """
    __slots__ = tuple()

    def __init__(self):
        super().__init__()

    def forward(self, X):
        self._X = X
        Z = np.maximum(0, self._X)
        return Z

    def backward(self, dZ):
        dX = dZ * (self._X > 0)
        return dX