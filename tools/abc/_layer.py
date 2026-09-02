from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import ClassVar

class Layer(ABC):
    """Classe abstraite des couches neuronales
    """
    __slots__ = ("_X",)

    ACT_FUNC: ClassVar[bool] = False

    def __init__(self) -> None:
        self._X: NDArray | None = None

    @abstractmethod
    def forward(self, X: NDArray) -> NDArray:
        """Renvoie *Z*

        Args:
            X: données d'entrée
        """

    @abstractmethod
    def backward(self, dZ: NDArray) -> NDArray:
        """Renvoie *dX*

        Args:
            dZ: gradient de sortie
        """

    def learn(self, lr: float = 1.0) -> None:
        """Fonction d'apprentissage

        Args:
            lr: taux d'apprentissage
        """
        pass