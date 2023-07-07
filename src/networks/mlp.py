"""
mlp.py

The class for multi-layer perceptrons.
"""

from dataclasses import dataclass, field
from typing import Type
from jaxtyping import Float, jaxtyped

from torch import Tensor, nn
from typeguard import typechecked

from src.networks.base_network import Network, NetworkConfig


@dataclass
class MLPConfig(NetworkConfig):
    """The configuration of a multi-layer perceptron"""

    _target: Type = field(default_factory=lambda: MLP)

    input_dim: int = 1
    """Dimensionality of network input"""
    output_dim: int = 1
    """Dimensionality of network output"""
    num_hidden_layers: int = 1
    """Number of hidden layers"""
    hidden_dim: int = 1
    """Dimensionality of hidden layers"""
    actvn_func: nn.Module = nn.ReLU()
    """Activation function for input and hidden layers"""
    out_actvn_func: nn.Module = None
    """Activation function for output layer"""


class MLP(Network):
    """A MLP"""

    config: MLPConfig

    def __init__(
        self,
        config: NetworkConfig
    ) -> None:
        super().__init__(config)

        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim
        self.num_hidden_layers = self.config.num_hidden_layers
        self.hidden_dim = self.config.hidden_dim
        self.actvn_func = self.config.actvn_func
        self.out_actvn_func = self.config.out_actvn_func

        self._build_backbone()

    def _build_backbone(self) -> None:  
        """Builds the MLP backbone"""

        # pylint: disable=attribute-defined-outside-init
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
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
        network_input: Float[Tensor, "*batch_size input_dim"],
    ) -> Float[Tensor, "*batch_size output_dim"]:
        """The forward pass of the network"""

        hidden = self.actvn_func(self.input_layer(network_input))

        for hidden_layer in self.hidden_layers:
            hidden = self.actvn_func(hidden_layer(hidden))

        output = self.output_layer(hidden)
        if self.out_actvn_func is not None:
            output = self.out_actvn_func(output)

        return output
