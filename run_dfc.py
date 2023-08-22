import torch
import torch.nn as nn

from models.coders import Structure
from models.dfc import DFC, DFC_layer
from src.datasets import MNIST
from src.trainers import Trainer
from src.utils import dotdict, set_device

device = set_device()

torch.set_grad_enabled(False)


def main():
    # Load data
    mnist = MNIST(mode="flat")
    train, eval = mnist.get_datasets()

    # Training configuration
    config = {
        "encoder_layers": [784, 512, 16],
        "decoder_layers": [16, 512, 784],
        "lr": 1e-4,
        "batch_size": 2048,
        "epochs": 200,
        "num_workers": 4,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": device,
        "output_dir": "./outputs",
        "seed": 1337,
        "target_lr": 1e-4,
        "alpha_di": 1e-4,
    }
    config = dotdict(config)

    # Train model
    structure = Structure(
        DFC_layer, config.encoder_layers, config.decoder_layers, nn.ReLU(), nn.Sigmoid()
    )
    model = DFC(
        encoder=structure.encoder,
        decoder=structure.decoder,
        config=config,
    )
    trainer = Trainer(model, train, eval, config)
    trainer.train()

    # Evaluate model


if __name__ == "__main__":
    main()
