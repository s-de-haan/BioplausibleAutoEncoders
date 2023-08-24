from src.utils import ModelOutput
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, encoder, decoder, name="Network") -> None:
        super(Network, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.name = name

    @property
    def layers(self):
        return self.encoder.layers + self.decoder.layers

    def forward(self, x):
        self.input = x
        z = self.encoder(x)
        y_hat = self.decoder(z)

        self.loss = self.loss_fn(y_hat, x)
        self.y = x
        self.y_hat = y_hat

        output = ModelOutput(loss=self.loss, y_hat=y_hat, z=z)

        return output

    def backward(self):
        pass

    def loss_fn(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)