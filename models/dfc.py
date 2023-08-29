import time
import torch
import torch.nn as nn

from models.network import Network
from src.utils import get_derivative


class DFC(Network):
    def __init__(self, encoder, decoder, config) -> None:
        super().__init__(encoder, decoder, name="DFC")

        # Last layer should have identity feedback
        self.layers = self.encoder.layers + self.decoder.layers
        self.layers[-1].feedback = torch.eye(self.layers[-1].shape[0])

        self._target_lr = config.target_lr
        self._alpha_di = config.alpha_di
        
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
        
        activations_derivatives = [
            layer.activation_derivative(layer.linear_activations)
            for layer in self.layers
        ]
        bsz = self.layers[0].activations.shape[0]

        # TODO DFC misses bias
        # start = [
        #     layer.activation_derivative(layer.linear_activations).unsqueeze(2) * torch.eye(layer.out_features).repeat(bsz, 1, 1)
        #     for layer in self.layers
        # ]

        # biases = [layer.bias.unsqueeze(0).expand(bsz, layer.bias.shape[0]) for layer in self.layers[2:]]
        # start[:-1] = [layer.activation_fn(torch.matmul(A, layer.weights.t())) for A, layer in zip(start[:-1], self.layers[1:])]

        # for i, layer in enumerate(self.layers[2:]):
        #     start[:i+1] = [layer.activation_fn(torch.matmul(A, layer.weights.t()) + biases[i].unsqueeze(1).expand(-1,A.shape[1],-1)) for A in start[:i+1]]

        # J = torch.cat(start, dim=1)
        # J = J.transpose(1, 2)

        # return J


        output_sz = self.layers[-1].out_features
        
        # Last layer
        Js.append(
            activations_derivatives[-1].view(bsz, output_sz, 1)
            * torch.eye(output_sz).repeat(bsz, 1, 1)
        )
        # Rest of the layers
        for i in range(len(self.layers) - 2, -1, -1):
            J = activations_derivatives[i].unsqueeze(1) * torch.matmul(
                Js[-1], self.layers[i + 1].weights
            )
            Js.append(J)

        Js.reverse()

        return torch.cat(Js, dim=2)



    def _non_dynamical_inversion(self):
        J = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.matmul(J, J_T) + self._alpha_di * torch.eye(J.shape[1]), error
        )

        delta_v = torch.matmul(J_T, u).squeeze(-1)
        delta_vs = torch.tensor_split(
            delta_v, torch.cumsum(torch.tensor(self.layer_sizes[:-1]), dim=0).cpu(), dim=1
        )

        rs = [self.input]

        for i, layer in enumerate(self.layers):
            v_ff = torch.matmul(rs[i], layer.weights.t())
            v_ff += layer.bias.unsqueeze(0).expand_as(v_ff)
            v = v_ff + delta_vs[i]

            r_ff = layer.activation_fn(v_ff)
            r = layer.activation_fn(v)
            rs.append(r)

            layer.v_ff = v_ff
            layer.v = v
            layer.delta_v = delta_vs[i]
            
            layer.r = r
            layer.r_ff = r_ff
            layer.r_prev = rs[i]
            
        # self.u = u


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
        a = torch.matmul(x, self.weights.t())
        a += self.bias.unsqueeze(0).expand_as(a)
        self.activations = self.activation_fn(a)
        self.linear_activations = a

        return self.activations

    def backward(self):
        teaching_signal = self.r - self.r_ff

        bsz = self.r_prev.shape[0]
        weights_grad = -2 * 1.0 / bsz * teaching_signal.t().mm(self.r_prev)
        bias_grad = -2 * teaching_signal.mean(dim=0)

        self._weights.grad = weights_grad
        self._bias.grad = bias_grad
