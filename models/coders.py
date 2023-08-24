import torch.nn as nn


class Structure(nn.Module):
    def __init__(
        self,
        layer_class,
        encoder_layers,
        decoder_layers,
        activation_fn,
        output_activation_fn,
    ):
        super(Structure, self).__init__()
        self.layer_class = layer_class
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        assert (
            len(self.encoder_layers) > 1
        ), "encoder_layers must be a list of integers, starting with input_dim, ending with output_dim"
        assert (
            len(self.decoder_layers) > 1
        ), "decoder_layers must be a list of integers, starting with input_dim, ending with output_dim"

        self._encoder = self.Encoder(layer_class, encoder_layers, activation_fn)
        self._decoder = self.Decoder(
            layer_class, decoder_layers, activation_fn, output_activation_fn
        )

    class Encoder(nn.Module):
        def __init__(self, layer_class, _layers, activation_fn) -> None:
            super().__init__()

            self.layers = nn.ModuleList()
            for i in range(len(_layers) - 1):
                self.layers.append(
                    layer_class(
                        _layers[i],
                        _layers[i + 1],
                        activation_fn=activation_fn,
                    )
                )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class Decoder(nn.Module):
        def __init__(
            self, layer_class, _layers, activation_fn, output_activation_fn
        ) -> None:
            super().__init__()

            self.layers = nn.ModuleList()
            for i in range(len(_layers) - 2):
                self.layers.append(
                    layer_class(
                        _layers[i],
                        _layers[i + 1],
                        activation_fn=activation_fn,
                    )
                )
            self.layers.append(
                layer_class(
                    _layers[-2],
                    _layers[-1],
                    activation_fn=output_activation_fn,
                )
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
