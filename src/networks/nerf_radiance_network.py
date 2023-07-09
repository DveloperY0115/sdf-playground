"""
radiance_network.py

The class for networks parameterizing radiance fields.
"""

from dataclasses import dataclass, field
from typing import Type

from jaxtyping import Float, jaxtyped
import torch
from torch import Tensor, nn
from typeguard import typechecked

from src.encoders.positional_encoder import PositionalEncoderConfig
from src.networks.base_network import Network, NetworkConfig


@dataclass
class NeRFRadianceNetworkConfig(NetworkConfig):
    """The configuration of a radiance network"""

    _target: Type = field(default_factory=lambda: NeRFRadianceNetwork)

    coord_dim: int = 3
    """Dimensionality of coordinate vectors"""
    view_dim: int = 3
    """Dimensionality of view direction vectors"""
    coord_encoder_config: PositionalEncoderConfig = PositionalEncoderConfig()
    """The configuration of the positional encoder for 3D coordinates"""
    view_encoder_config: PositionalEncoderConfig = PositionalEncoderConfig()
    """The configuration of the positional encoder for view directions""" 
    radiance_dim: int = 3
    """Dimensionality of radiance vectors"""
    num_hidden_layers: int = 1
    """Number of hidden layers"""
    hidden_dim: int = 1
    """Dimensionality of hidden layers"""
    actvn_func: nn.Module = nn.ReLU()
    """Activation function for input and hidden layers"""
    out_actvn_func: nn.Module = None
    """Activation function for output layer. Set to None by default"""


class NeRFRadianceNetwork(Network):
    """A radiance network"""

    config: NeRFRadianceNetworkConfig

    def __init__(
        self,
        config: NetworkConfig,
    ) -> None:
        super().__init__(config)

        self.coord_dim = self.config.coord_dim
        self.view_dim = self.config.view_dim
        self.coord_encoder_config = self.config.coord_encoder_config
        self.view_encoder_config = self.config.view_encoder_config
        self.radiance_dim = self.config.radiance_dim
        self.num_hidden_layers = self.config.num_hidden_layers
        self.hidden_dim = self.config.hidden_dim
        self.actvn_func = self.config.actvn_func
        self.out_actvn_func = self.config.out_actvn_func

        self._build_network()

    # pylint: disable=attribute-defined-outside-init
    def _build_network(self) -> None:
        """Builds the radiance network backbone"""

        # initialize layers
        self.input_layer = nn.Linear(self.coord_dim + self.view_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.radiance_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim) \
                for _ in range(self.num_hidden_layers)
            ],
        )

        # initialize positional encoders
        self.coord_encoder = self.coord_encoder_config.setup()
        self.view_encoder = self.view_encoder_config.setup()

    @jaxtyped
    @typechecked
    def forward(
        self,
        coords: Float[Tensor, "*batch_size coord_dim"],
        view_directions: Float[Tensor, "*batch_size view_dim"],
    ) -> Float[Tensor, "*batch_size 3"]:
        """The forward pass of the network"""

        # encode network inputs
        coord_embedding = self.coord_encoder.encode(coords)
        view_embedding = self.view_encoder.encode(view_directions)

        # network forward prop
        network_input = torch.cat([coord_embedding, view_embedding], dim=-1)
        hidden = self.actvn_func(self.input_layer(network_input))
        for hidden_layer in self.hidden_layers:
            hidden = self.actvn_func(hidden_layer(hidden))
        output = self.output_layer(hidden)
        if self.out_actvn_func is not None:
            output = self.out_actvn_func(output)

        return output
