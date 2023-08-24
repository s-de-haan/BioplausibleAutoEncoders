from collections import OrderedDict
import multiprocessing
from typing import Any, Tuple

import torch
import torch.nn as nn


class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library

    taken from clementchadebec github"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    multiprocessing.set_start_method("fork")

    device = "cpu"

    torch.set_default_device(device)

    return device


def derivative_sigmoid(x):
    return torch.mul(torch.sigmoid(x), 1.0 - torch.sigmoid(x))


def derivative_relu(x):
    grad = torch.ones_like(x)
    grad[x < 0] = 0
    return grad


def get_derivative(activation_fn):
    if isinstance(activation_fn, torch.nn.Sigmoid):
        return derivative_sigmoid
    elif isinstance(activation_fn, torch.nn.ReLU):
        return derivative_relu
    else:
        raise ValueError(f"Activation function {activation_fn} not supported")

class DataParallelWrapper(nn.DataParallel):
    def __init__(self, module):
        super().__init__(module)


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)