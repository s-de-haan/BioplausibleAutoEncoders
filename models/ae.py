import torch
import torch.nn as nn
from models.network import Network
from src.utils import ModelOutput


class AE(Network):

    """Vanilla Autoencoder"""

    def __init__(self, encoder, decoder) -> None:
        super().__init__(self, encoder, decoder, name="AE")

    def backward(self):
        self.loss.backward()