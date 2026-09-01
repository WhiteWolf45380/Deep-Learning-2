import numpy as np
from numpy.typing import NDArray

class ReLU:
    def __init__(self):
        self.X: NDArray | None = None

    def forward(self, X: NDArray) -> NDArray:
        """Renvoie *A = ReLU(X)*

        Args:
            X: observations d'entrée
        """
        self.X = X
        A = np.maximum(0, X)
        return A

    def backward(self, dA: NDArray) -> NDArray:
        """Renvoie *dX = dA * ReLU'(X)*
        
        Args:
            dA: gradient de A
        """
        dX = dA * (self.X > 0)
        return dX