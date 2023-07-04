"""
base_primitive.py

The base class for SDF primitives.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type

from jaxtyping import Shaped
from torch import Tensor

from src.configs.base_config import InstantiateConfig


@dataclass
class PrimitiveConfig(InstantiateConfig):
    """The configuration of a SDF primitive"""

    _target: Type = field(default_factory=lambda: Primitive)


class Primitive(ABC):
    """The base class of SDF primitives"""

    config: PrimitiveConfig

    def __init__(
        self,
        config: PrimitiveConfig,
    ) -> None:
        self.config = config

    @abstractmethod
    def evaluate_density(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """Evaluates the density function parameterized by SDF at the given points."""

    @abstractmethod
    def evaluate_sdf(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """Evaluates the signed distance function at the given points."""

    @abstractmethod
    def evaluate_sdf_gradient(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point 3"]:
        """Evaluates the gradient of the signed distance function at the given points."""


if __name__ == "__main__":

    # create config
    config = PrimitiveConfig()

    # instantiate primitive
    primitive = config.setup()
