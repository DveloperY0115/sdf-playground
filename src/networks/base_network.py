"""
base_network.py

The base class for neural networks.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type

from torch import nn

from src.configs.base_config import InstantiateConfig


@dataclass
class NetworkConfig(InstantiateConfig):
    """The configuration of a network"""

    _target: Type = field(default_factory=lambda: Network)


class Network(nn.Module):
    """The base class of neural networks"""

    config: NetworkConfig

    backbone: nn.Module
    """"""

    def __init__(
        self,
        config: NetworkConfig,
    ) -> None:
        self.config = config

        super().__init__()

    @abstractmethod
    def _build_backbone(self) -> None:
        """Builds the backbone of the network"""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """The forward pass of the network"""
