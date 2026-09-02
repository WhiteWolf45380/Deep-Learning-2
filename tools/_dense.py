import numpy as np
from numpy.typing import NDArray
from .abc import Layer

class Dense(Layer):
    """Couche neuronal dense

    Args:
        n: taille des entrées
        m: taille des sorties
    """
    __slots__ = (
        "_n", "_m",
        "_W", "_b",
        "_Z",
        "_dW", "_db", "_dX",
    )

    def __init__(self, n: int, m: int) -> None:
        super().__init__()

        # Taille de la couche
        self._n: int = n
        self._m: int = m

        # Paramètres de la couche
        self._W: NDArray = np.random.random((n, m))
        self._b: NDArray = np.random.rand(m)

        # Sauvegarde de la sortie courante
        self._Z: NDArray | None = None

        # Sauvegarde des gradients courants
        self._dW: NDArray | None = None
        self._db: NDArray | None = None
        self._dX: NDArray | None = None

    @property
    def n(self) -> int:
        """Nombre d'entrées"""
        return self._n

    @property
    def m(self) -> int:
        """Nombre de sorties"""
        return self._m

    def forward(self, X):
        self._X = X
        self._Z = self._X @ self._W + self._b
        return self._Z

    def backward(self, dZ):
        self._dW = self._X.T @ dZ
        self._db = np.sum(dZ, axis=0)
        self._dX = dZ @ self._W.T
        return self._dX

    def learn(self, lr) -> None:
        self._W = self._W - self._dW * lr
        self._b = self._b - self._db * lr