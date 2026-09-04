from torchvision import datasets

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
)