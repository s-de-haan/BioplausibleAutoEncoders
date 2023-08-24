import torch
import torch.nn as nn
from models.network import Network
from src.utils import ModelOutput


class AE(Network):

    """Vanilla Autoencoder"""

    def __init__(self, encoder, decoder) -> None:
        super().__init__(encoder, decoder, name="AE")

    def backward(self):
        self.loss.backward()
    
class MLP_layer(nn.Module):
    def __init__(
        self, in_features, out_features, activation_fn=nn.ReLU(), name="DFC_layer"
    ) -> None:
        super(MLP_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.name = name

        self.feedforward = nn.Sequential(
            nn.Linear(self.in_features, self.out_features), self.activation_fn
        )
        nn.init.kaiming_normal_(self.feedforward[0].weight)
        self._weights = self.feedforward[0].weight
        self._bias = self.feedforward[0].bias

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def shape(self):
        return self._weights.shape

    def forward(self, x):
        return self.activation_fn(torch.matmul(x, self.weights.t()) + self.bias)