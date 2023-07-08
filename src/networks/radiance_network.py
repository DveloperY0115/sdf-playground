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

from src.networks.base_network import Network, NetworkConfig


@dataclass
class RadianceNetworkConfig(NetworkConfig):
    """The configuration of a radiance network"""

    _target: Type = field(default_factory=lambda: RadianceNetwork)

    coord_dim: int = 1
    """Dimensionality of coordinate vectors"""
    view_dim: int = 1
    """Dimensionality of view direction vectors"""
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


class RadianceNetwork(Network):
    """A radiance network"""

    config: RadianceNetworkConfig

    def __init__(
        self,
        config: NetworkConfig,
    ) -> None:
        super().__init__(config)

        self.coord_dim = self.config.coord_dim
        self.view_dim = self.config.view_dim
        self.radiance_dim = self.config.radiance_dim
        self.num_hidden_layers = self.config.num_hidden_layers
        self.hidden_dim = self.config.hidden_dim
        self.actvn_func = self.config.actvn_func
        self.out_actvn_func = self.config.out_actvn_func

        self._build_network()

    # pylint: disable=attribute-defined-outside-init
    def _build_network(self) -> None:
        """Builds the radiance network backbone"""

        self.input_layer = nn.Linear(self.coord_dim + self.view_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.radiance_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim) \
                for _ in range(self.num_hidden_layers)
            ],
        )

    @jaxtyped
    @typechecked
    def forward(
        self,
        coords: Float[Tensor, "*batch_size coord_dim"],
        view_directions: Float[Tensor, "*batch_size view_dim"],
    ) -> Float[Tensor, "*batch_size 3"]:
        """The forward pass of the network"""

        network_input = torch.cat([coords, view_directions], dim=-1)

        hidden = self.actvn_func(self.input_layer(network_input))

        for hidden_layer in self.hidden_layers:
            hidden = self.actvn_func(hidden_layer(hidden))

        output = self.output_layer(hidden)
        if self.out_actvn_func is not None:
            output = self.out_actvn_func(output)

        return output
