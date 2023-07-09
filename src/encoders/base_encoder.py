"""
base_encoder.py

The base class for signal encoders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type

from jaxtyping import Shaped
from torch import Tensor

from src.configs.base_config import InstantiateConfig


@dataclass
class EncoderConfig(InstantiateConfig):
    """The configuration of a signal encoder"""

    _target: Type = field(default_factory=lambda: Encoder)


class Encoder(ABC):
    """The base class of signal encoders"""

    config: EncoderConfig

    def __init__(
        self,
        config: EncoderConfig,
    ) -> None:
        self.config = config

    @abstractmethod
    def encode(
        self,
        signal: Shaped[Tensor, "*batch_size signal_dim"],
    ) -> Shaped[Tensor, "*batch_size encoding_dim"]:
        """Encodes the signal using the encoder."""
