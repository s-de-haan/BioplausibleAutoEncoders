import torch
import torch.nn as nn

from datasets import MNIST

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    mnist = MNIST(mode="conv")


if __name__ == "__main__":
    main()
