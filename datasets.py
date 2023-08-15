import torchvision.datasets as datasets
import numpy as np

from torchvision.transforms import ToTensor

class MNIST:
    def __init__(self, mode="conv") -> None:
        mnist_trainset = datasets.MNIST(
            root="./data", train=True, download=True, transform=ToTensor()
        )
        mnist_evalset = datasets.MNIST(
            root="./data", train=False, download=True, transform=ToTensor()
        )

        io_dim = (-1, 1, 28, 28) if mode == "conv" else (-1, 784)

        self.train_dataset = mnist_trainset.data.reshape(io_dim) / 255.0
        self.eval_dataset = mnist_evalset.data.reshape(io_dim) / 255.0
        self.io_dim = io_dim

    def get_datasets(self):
        return self.train_dataset, self.eval_dataset
