"""
base_field.py

The base class for field representations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type

from jaxtyping import Shaped
from torch import Tensor
from torch import nn

from src.configs.base_config import InstantiateConfig


@dataclass
class FieldConfig(InstantiateConfig):
    """The configuration of a field"""

    _target: Type = field(default_factory=lambda: Field)


class Field(nn.Module):
    """The base class of fields"""

    config: FieldConfig

    def __init__(
        self,
        config: FieldConfig,
    ) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def evaluate_radiance(
        self,
        coords: Shaped[Tensor, "num_point 3"],
        view_directions: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point 3"]:
        """Evaluates the radiance function at the given points."""

    @abstractmethod
    def evaluate_density(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """Evaluates the density function at the given points."""

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
