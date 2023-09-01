import torch
import torch.nn as nn

from models.network import Network
from src.utils import get_derivative


class JacProp(Network):
    def __init__(self, encoder, decoder, config) -> None:
        super().__init__(encoder, decoder, name="JacProp")

        self._target_lr = (
            config.target_lr
        )  # TODO doesnt this multiply with the learning rate?

    def backward(self):
        """MSE loss solution"""
        error = -2 * self._target_lr * self.y_hat + 2 * self._target_lr * self.y
        error = error.unsqueeze(2)

        bsz = error.shape[0]
        output_sz = self.layers[-1].out_features

        act_fn_deriv = [
            layer.activation_derivative(layer.linear_activations)
            for layer in self.layers[::-1]
        ]
        Ws = [torch.eye(self.layers[-1].out_features).repeat(error.shape[0], 1, 1)] + [layer.weights for layer in self.layers[:0:-1]]
        # bs = [layer.bias.unsqueeze(0) for layer in self.layers[::-1]]

        J_vdv = torch.eye(self.layers[-1].out_features).repeat(error.shape[0], 1, 1)

        for l, layer in enumerate(self.layers[::-1]):
            if l == 0: # Last layer
                v = act_fn_deriv[l].view(error.shape[0], -1, 1)
            else: # All other layers
                v = act_fn_deriv[l].unsqueeze(1)

            # TODO Does the bias need to be included in the Jacobian_v?
            # v = a + bs[l].expand_as(a)
            if l == 0: # Last layer
                J_v = v * J_vdv
            else: # All other layers
                J_v = v * torch.matmul(J_vdv, Ws[l])

            dv = torch.matmul(J_v.transpose(1, 2), error).squeeze(2)
            if l == 0: # Last layer
                v_dv = layer.activation_derivative(layer.linear_activations + dv).view(error.shape[0], -1, 1)
                # v_dv = a_dv + bs[l].unsqueeze(2).expand_as(a_dv)
            else: # All other layers
                v_dv = layer.activation_derivative(layer.linear_activations + dv).unsqueeze(1)
                # v_dv = a_dv + bs[l].expand_as(a_dv)

            J_vdv = v_dv * torch.matmul(J_vdv, Ws[l])
            layer.r = layer.activation_fn(layer.linear_activations + dv)

        # Last layer
        # J = act_fn_deriv[-1].view(bsz, output_sz, 1) * torch.eye(output_sz).repeat(
        #     bsz, 1, 1
        # )
        # self.layers[-1].dv = torch.matmul(J.transpose(1, 2), error).squeeze(2)

        # # All other layers
        # for l in range(len(self.layers) - 2, -1, -1):
        #     J = act_fn_deriv[l].unsqueeze(1) * torch.matmul(
        #         J, self.layers[l + 1].weights
        #     )
        #     self.layers[l].dv = torch.matmul(J.transpose(1, 2), error).squeeze(2)

        # rs = [self.input]

        # for i, layer in enumerate(self.layers):
        #     v_ff = torch.matmul(rs[i], layer.weights.t())
        #     v_ff += layer.bias.unsqueeze(0).expand_as(v_ff)
        #     v = v_ff + layer.dv

        #     r_ff = layer.activation_fn(v_ff)
        #     r = layer.activation_fn(v)
        #     rs.append(r)

        #     layer.v_ff = v_ff
        #     layer.v = v

        #     layer.r = r
        #     layer.r_ff = r_ff
        #     layer.r_prev = rs[i]

        for layer in self.layers:
            layer.backward()


class JacProp_layer(nn.Module):
    def __init__(
        self, in_features, out_features, activation_fn=nn.ReLU(), name="JacProp_layer"
    ) -> None:
        super(JacProp_layer, self).__init__()
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
        self.r_prev = x

        return self.activations

    def backward(self):
        teaching_signal = self.r - self.activations

        bsz = self.r_prev.shape[0]
        weights_grad = -2 * 1.0 / bsz * teaching_signal.t().mm(self.r_prev)
        bias_grad = -2 * teaching_signal.mean(dim=0)

        self._weights.grad = weights_grad
        self._bias.grad = bias_grad
