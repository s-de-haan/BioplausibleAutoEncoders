import torch
import torch.nn as nn
import numpy as np

from models.network import Network
from src.utils import get_derivative


class DFC(Network):
    def __init__(self, encoder, decoder, config, name="DFC") -> None:
        super().__init__(self, encoder, decoder, name=name)

        # Last layer should have identity feedback
        self.layers[-1].feedback = torch.eye(self.layers[-1].shape[0])

        self._target_lr = config.target_lr
        self._alpha_di = config.alpha_di

    @property
    def layers(self):
        return self.encoder.layers + self.decoder.layers

    @property
    def layer_sizes(self):
        return [layer.out_features for layer in self.layers]

    @property
    def activations(self):
        return [layer.activations for layer in self.layers]

    @property
    def linear_activations(self):
        return [layer.linear_activations for layer in self.layers]

    def backward(self):
        self._set_targets()
        self._non_dynamical_inversion()

        for layer in self.layers:
            layer.backward()

    def _set_targets(self):
        """MSE loss solution"""
        self.targets = (
            1 - 2 * self._target_lr
        ) * self.y_hat + 2 * self._target_lr * self.y

    def _calculate_full_jacobian(self):
        Js = []

        activations_vec = [
            layer.activation_derivative(layer.linear_activations)
            for layer in self.layers
        ]

        bsz = self.layers[0].shape[0]
        output_sz = self.layers[-1].shape[1]

        # Last layer
        Js.append(
            activations_vec[-1].unsqueeze(-1) * torch.eye(output_sz).repeat(bsz, 1, 1)
        )
        # Rest of the layers
        for i in range(len(self.layers) - 1, -1, -1):
            J = activations_vec[i].unsqueeze(1) * torch.matmul(
                J, self.layers[i + 1].weights
            )
            Js.append(J)

        Js.reverse()

        return torch.stack(Js, dim=1)

    def _non_dynamical_inversion(self, targets):
        J = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.solve(
            error, torch.matmul(J, J_T) + self._alpha_di * torch.eye(J.shape[1])
        )[0].squeeze(-1)
        delta_v = torch.matmul(J_T, u.unsqueeze(2)).squeeze(-1)
        delta_vs = torch.split(delta_v, torch.cumsum(self.layer_sizes, dim=0), dim=1)

        vs = []
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            vs.append(
                delta_vs[i]
                + torch.matmul(rs[i], layer.weights.t())
                + layer.bias.unsqueeze(0)
            )
            rs.append(layer.activation_fn(vs[i]))
            layer.target = vs[i]
            layer.delta_v = delta_vs[i]
            layer.r_prev = rs[i]

        self.u = u


class DFC_layer(nn.Module):
    def __init__(
        self, in_features, out_features, activation_fn=nn.ReLU(), name="DFC_layer"
    ) -> None:
        super(DFC_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.name = name

        self.activation_derivative = get_derivative(self.activation_fn)

        self.feedforward = nn.Sequential(
            nn.Linear(self.in_features, self.out_features), self.activation
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

    def forward(self, x):
        a = torch.matmul(x, self.weights.t())
        a += self.bias.unsqueeze(0).expand_as(a)
        self.activations = self.activation_fn(a)
        self.linear_activations = a

        return self.activations

    def backward(self):
        v_ff = torch.matmul(self.r_prev, self.feedforward[0].weight.t())
        v_ff += self.feedforward[0].bias.unsqueeze(0).expand_as(v_ff)
        v = v_ff + self.delta_v

        teaching_signal = self.activation_fn(v) - self.activation_fn(v_ff)

        bsz = self.r_prev.shape[0]
        weights_grad = -2 * 1.0 / bsz * teaching_signal.t().mm(self.r_prev)
        bias_grad = -2 * teaching_signal.mean(dim=0)

        self._weights.grad = weights_grad
        self._bias.grad = bias_grad
