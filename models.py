import torch
import torch.nn as nn
from utils import ModelOutput


class AE(nn.Module):

    """Vanilla Autoencoder"""

    def __init__(self, encoder, decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)

        loss = self.loss_fn(y_hat, x)

        output = ModelOutput(loss=loss, y_hat=y_hat, z=z)

        return output

    def loss_fn(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)  # might be different for convolutions


class MLP_Encoder(nn.Module):

    """Multilayer perceptron"""

    def __init__(self, layers) -> None:
        assert (
            len(layers) > 2
        ), "layers must be a list of integers, starting with input_dim, ending with output_dim"

        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Sequential([nn.Linear(layers[i], layers[i + 1]), nn.ReLU()]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP_Decoder(nn.Module):

    """Multilayer perceptron"""

    def __init__(self, layers) -> None:
        assert (
            len(layers) > 2
        ), "layers must be a list of integers, starting with input_dim, ending with output_dim"

        self.layers = nn.ModuleList()

        for i in range(len(layers) - 2):
            self.layers.append(nn.Sequential([nn.Linear(layers[i], layers[i + 1]), nn.ReLU()]))
        self.layers.append(nn.Sequential([nn.Linear(layers[i], layers[i + 1]), nn.Sigmoid()]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x