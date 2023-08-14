import torchvision.datasets as datasets

class MNIST:

    def __init__(self, mode='conv') -> None:
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

        io_dim = (-1, 1, 28, 28) if mode == 'conv' else (-1, 784)

        self.train_dataset = mnist_trainset.data[:-10000].reshape(io_dim) / 255.
        self.eval_dataset = mnist_trainset.data[-10000:].reshape(io_dim) / 255.
        self.io_dim = io_dim

    def get_train_eval(self):
        return self.train_dataset, self.eval_dataset