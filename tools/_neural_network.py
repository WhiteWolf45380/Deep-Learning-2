from .abc import Layer, Loss
from . import MSE

from numpy.typing import NDArray

class NeuralNetwork:
    """Réseau neuronal séquentiel
    
    Args:
        *layers: couches composant le réseau
    """
    __slots__ = (
        "_layers", "_loss",
        "_Z",
    )

    def __init__(self, *layers: Layer, loss: Loss = None):
        self._layers: tuple[Layer] = layers
        self._loss: Loss = loss or MSE()

    def forward(self, X: NDArray) -> NDArray:
        """Passe un jeu de données dans le réseau et renvoie *Z*
        
        Args:
            x: données d'entrée
        """
        self._Z = X
        for layer in self._layers:
            self._Z = layer.forward(self._Z)
        return self._Z

    def backward(self, Y: NDArray) -> NDArray:
        """Effectue la descente de gradients et renvoie le coût *L*"""
        L = self._loss.compute(self._Z, Y)
        dX = self._loss.backward()
        for layer in reversed(self._layers):
            dX = layer.backward(dX)
        return L

    def learn(self, lr: float = 1.0) -> None:
        """Applique l'apprentissage
        
        Appeler cette fonction après la descente de gradients.
        """
        for layer in self._layers:
            layer.learn(lr=lr)