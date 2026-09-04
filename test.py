from tools import NeuralNetwork, MSE, Dense, ReLU
from numpy.typing import NDArray

NN = NeuralNetwork(
    Dense(2, 8),
    ReLU(),
    Dense(8, 32),
    ReLU(),
    Dense(32, 8),
    ReLU(),
    Dense(8, 1),
    loss=MSE(),
)

import numpy as np

X = np.random.uniform(-5, 5, (1000, 2))

Y = X[:, 0] ** 2 + X[:, 1]**2
Y = Y.reshape(-1, 1)

Z = NN.forward(X)
print(f"Z {Z.shape}: \n{Z}\n")

L = NN.backward(Y)
print(f"Loss: {L}")

NN.learn(lr=0.001)

Z = NN.forward(X)
print(f"Z {Z.shape}: \n{Z}\n")

L = NN.backward(Y)
print(f"Loss: {L}")

def train(NN: NeuralNetwork, X: NDArray, Y: NDArray, lr: float = 0.001, epochs: int = 100000, print_rate: float = 0.01) -> None:
    print()

    for i in range(epochs):
        Z = NN.forward(X)
        L = NN.backward(Y)
        NN.learn(lr=lr)

        if print_rate > 0 and i % (1 / print_rate) == 0:
            print(f"L ({i}): {L}")

    if  print_rate > 0 or epochs % (1 / print_rate - 1) != 0:
        print(f"L ({epochs}): {L}")

train(NN, X, Y)