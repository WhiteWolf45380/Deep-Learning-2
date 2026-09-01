import numpy as np
from numpy.typing import NDArray

class Dense:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size: int = input_size
        self.output_size: int = output_size

        self.W: NDArray = np.zeros((input_size, output_size))
        self.b: NDArray = np.zeros(self.output_size)

        self.X: NDArray | None = None
        self.Z: NDArray | None = None

        self.dW: NDArray | None = None
        self.db: NDArray | None = None
        self.dX: NDArray | None = None

    def forward(self, X: NDArray) -> NDArray:
        """Renvoie *Z = X @ W + b*

        Args:
            X: observations d'entrée
        """
        self.X = X
        self.Z = self.X @ self.W + self.b
        return self.Z

    def backward(self, dZ: NDArray) -> NDArray:
        """Renvoie *dX = dZ @ W.T*

        Args:
            dZ: gradient de Z
        """
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0)
        self.dX = dZ @ self.W.T
        return self.dX

if __name__ == "__main__":
    X = np.array([
        [1., 2.],
        [3., 4.]
    ])

    layer = Dense(2, 2)
    Z = layer.forward(X)
    print(Z)

    dZ = np.array([
        [1., 2.],
        [3., 4.]
    ])

    dX = layer.backward(dZ)
    print()
    print(dX)

    lr = 0.01
    layer.W = layer.W - lr * layer.dW
    layer.b = layer.b - lr * layer.db

    Z = layer.forward(X)
    print()
    print(Z)