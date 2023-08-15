import torch
import torch.nn as nn

from models import AE, MLP_Decoder, MLP_Encoder
from datasets import MNIST
from trainers import Trainer
from utils import dotdict

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Load data
    mnist = MNIST(mode="flat")
    train, eval = mnist.get_datasets()

    # Training configuration
    config = {
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 100,
        "num_workers": 4,
        "optimizer": "Adam",
        "device": device,
        "save_path": "./checkpoints",
        "output_dir": "./outputs",
        "seed": 1337,
    }
    config = dotdict(config)

    # Train model
    model = AE(encoder=MLP_Encoder([784, 512, 16]), decoder=MLP_Decoder([16, 512, 784]))
    trainer = Trainer(model, train, eval, config)
    trainer.train()

    # Evaluate model

if __name__ == "__main__":
    main()
