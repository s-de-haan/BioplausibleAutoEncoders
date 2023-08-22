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

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        layers = nn.ModuleList()
        for i in range(len(self.encoder_layers) - 1):
            layers.append(
                self.layer_class(
                    self.encoder_layers[i],
                    self.encoder_layers[i + 1],
                    activation_fn=self.activation_fn,
                )
            )
        return layers

    def build_decoder(self):
        layers = nn.ModuleList()
        for i in range(len(self.decoder_layers) - 2):
            layers.append(
                self.layer_class(
                    self.decoder_layers[i],
                    self.decoder_layers[i + 1],
                    activation_fn=self.activation_fn,
                )
            )
        layers.append(
            self.layer_class(
                self.decoder_layers[-2],
                self.decoder_layers[-1],
                activation_fn=self.output_activation_fn,
            )
        )
        return layers

    @property
    def encoder(self):
        return self.encoder
    
    @property
    def decoder(self):
        return self.decoder