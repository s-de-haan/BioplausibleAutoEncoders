import torch
import torch.nn as nn

from models.ae import AE, MLP_Decoder, MLP_Encoder
from src.datasets import MNIST
from src.trainers import Trainer
from src.utils import dotdict

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Load data
    mnist = MNIST(mode="flat")
    train, eval = mnist.get_datasets()

    # Training configuration
    config = {
        "encoder_layers": [784, 256, 32],
        "decoder_layers": [32, 256, 784],
        "lr": 1e-4,
        "batch_size": 128,
        "epochs": 25,
        "num_workers": 4,
        "optimizer": "Adam",
        "scheduler": None, #"CosineAnnealingLR",
        "device": device,
        "output_dir": "./outputs",
        "seed": 1337,
    }
    config = dotdict(config)

    # Train model
    model = AE(encoder=MLP_Encoder(config.encoder_layers), decoder=MLP_Decoder(config.decoder_layers))
    trainer = Trainer(model, train, eval, config)
    trainer.train()

    # Evaluate model

if __name__ == "__main__":
    main()
