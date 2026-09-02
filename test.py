from tools import NeuralNetwork, MSE, Dense, ReLU
from numpy.typing import NDArray

NN = NeuralNetwork(
    Dense(2, 3),
    ReLU(),
    Dense(3, 3),
    ReLU(),
    Dense(3, 1),
    loss=MSE(),
)

import numpy as np

X = np.array([
    [1.,  2.],
    [2.,  1.],
    [3.,  4.],
    [4.,  2.],
    [5.,  3.],
    [6.,  1.],
    [7.,  5.],
    [8.,  2.],
])

Y = np.array([
    [0.],
    [5.],
    [2.],
    [9.],
    [10.],
    [17.],
    [12.],
    [21.],
])

Z = NN.forward(X)
print(f"Z {Z.shape}: \n{Z}\n")

L = NN.backward(Y)
print(f"Loss: {L}")

NN.learn(lr=0.001)

Z = NN.forward(X)
print(f"Z {Z.shape}: \n{Z}\n")

L = NN.backward(Y)
print(f"Loss: {L}")

def train(NN: NeuralNetwork, X: NDArray, Y: NDArray, lr: float = 0.001, epochs: int = 1000, print_rate: float = 0.01) -> None:
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