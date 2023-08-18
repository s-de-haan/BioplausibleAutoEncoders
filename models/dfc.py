import torch
import torch.nn as nn
import numpy as np

class DFC(nn.Module):

    def __init__(self, encoder, decoder, name="DFC") -> None:
        super(DFC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.name = name

    def forward(self, x):
        pass

    def backward(self):
        pass

    def loss_fn(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)
    

class DFC_encoder(nn.Module):

    """Multilayer perceptron"""

    def __init__(self, layers) -> None:
        super(DFC_encoder, self).__init__()
        assert (
            len(layers) > 2
        ), "layers must be a list of integers, starting with input_dim, ending with output_dim"

        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(
                nn.Sequential(nn.Linear(layers[i], layers[i + 1]), nn.ReLU())
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class DFC_decoder(nn.Module):

    """Multilayer perceptron"""

    def __init__(self, layers) -> None:
        super(DFC_decoder, self).__init__()
        assert (
            len(layers) > 2
        ), "layers must be a list of integers, starting with input_dim, ending with output_dim"

        self.layers = nn.ModuleList()

        for i in range(len(layers) - 2):
            self.layers.append(DFC_layer(layers[i], layers[i + 1]))
        self.layers.append(DFC_layer(layers[-2], layers[-1], activation=nn.Sigmoid()))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class DFC_layer(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(), name="DFC_layer") -> None:
        super(DFC_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.name = name

        self.feedforward = nn.Sequential(nn.Linear(self.in_features, self.out_features), self.activation)
        nn.init.kaiming_normal_(self.feedforward[0].weight)
        self._weights = self.feedforward[0].weight
        self._bias = self.feedforward[0].bias

        self.feedback = torch.empty(self.out_features, self.in_features)
        nn.init.orthogonal_(self.feedback.weight, gain=np.sqrt(6. / (in_features + out_features)))

    @property
    def weights(self):
        return self._weights
    
    @property
    def bias(self):
        return self._bias

    @property
    def weights_backward(self):
        return self.feedback.weight

    def forward(self, x):
        return self.activation(torch.matmul(x, self.weights.t()) + self.bias)
    
    def compute_forward_gradients(self, u, r_prev):
        v_ff = torch.matmul(r_prev, self.feedforward[0].weight.t()) + self.feedforward[0].bias
        v = v_ff + u

        teaching_signal = self.activation(v) - self.activation(v_ff)

        bsz = r_prev.shape[0]
        weights_grad = -2 * 1./bsz * teaching_signal.t().mm(r_prev)
        bias_grad = -2 * teaching_signal.mean(dim=0)

        self._weights.grad = weights_grad
        self._bias.grad = bias_grad

