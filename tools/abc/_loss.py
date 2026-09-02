import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class Loss(ABC):
    def __init__(self) -> None:
        self._E: NDArray | None = None
        self._L: float | None = None

    @abstractmethod
    def compute(self, Z: NDArray, Y: NDArray) -> float:
        """Calcul le coût et le renvoie
        
        Args:
            Z: valeurs prédites
            Y: valeurs réelles
        """

    @abstractmethod
    def backward(self) -> NDArray:
        """Calcul le gradient de sortie *dZ* et le renvoie"""