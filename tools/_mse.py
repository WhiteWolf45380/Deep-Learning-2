from .abc import Loss
import numpy as np

class MSE(Loss):
    """Loss type Mean Squared Error (MSE)"""
    __slots__ = tuple()

    def __init__(self):
        super().__init__()

    def compute(self, Z, Y):
        self._E = Z - Y
        self._L = np.mean(self._E**2)
        return self._L

    def backward(self):
        dZ = 2 * self._E / self._E.size
        return dZ