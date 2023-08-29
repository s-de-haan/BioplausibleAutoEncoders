from models.coders import Structure
import torch.nn as nn

from models.ae import AE, MLP_layer
from src.datasets import MNIST
from src.trainers import Trainer
from src.utils import dotdict, set_device


def main():
    # Load data
    mnist = MNIST(mode="flat")
    train, eval = mnist.get_datasets()

    # Training configuration
    config = {
        "encoder_layers": [784, 512, 16],
        "decoder_layers": [16, 512, 784],
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 200,
        "runs": 10,
        "num_workers": 6,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cuda:0",
        "output_dir": "./outputs",
        "seed": 1337,
    }
    config = dotdict(config)

    # Train model
    for _ in range(config.runs):
        structure = Structure(
            MLP_layer, config.encoder_layers, config.decoder_layers, nn.ReLU(), nn.Sigmoid()
        )
        model = AE(
            encoder=structure.encoder,
            decoder=structure.decoder,
        )

        trainer = Trainer(model, train, eval, config)
        trainer.train()


if __name__ == "__main__":
    main()
