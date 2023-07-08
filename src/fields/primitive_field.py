"""
primitive_field.py

The class for fields parameterized by SDF primitives.
"""

from dataclasses import dataclass, field
from typing import Type
from jaxtyping import Shaped, jaxtyped

from torch import Tensor
from typeguard import typechecked

from src.fields.base_field import Field, FieldConfig
from src.networks.base_network import NetworkConfig
from src.primitives.base_primitive import Primitive, PrimitiveConfig

@dataclass
class PrimitiveFieldConfig(FieldConfig):
    """The configuration of a field parameterized by a SDF primitive"""

    _target: Type = field(default_factory=lambda: PrimitiveField)

    primitive_config: PrimitiveConfig = PrimitiveConfig()
    """The configuration of the SDF primitive"""
    radiance_network_config: NetworkConfig = NetworkConfig()
    """The configuration of the network parameterizing radiance field"""
    ####
    # 20230707 Seungwoo
    # temporarily disable learning SDF for now
    # sdf_network_config: NetworkConfig = NetworkConfig()
    # """The configuration of the network parameterizing signed distance field"""
    ####


class PrimitiveField(Field):
    """A field parameterized by a SDF primitive"""

    config: PrimitiveFieldConfig

    def __init__(
        self,
        config: PrimitiveFieldConfig,
    ) -> None:
        super().__init__(config)

        self.primitive: Primitive = self.config.primitive_config.setup()
        self.radiance_network = self.config.radiance_network_config.setup()
        ####
        # 20230707 Seungwoo
        # temporarily disable learning SDF for now
        # self.sdf_network = self.config.sdf_network_config.setup()
        ####

    @jaxtyped
    @typechecked
    def evaluate_radiance(
        self,
        coords: Tensor,
        view_directions: Tensor,
    ) -> Tensor:
        """Evaluates the radiance function at the given points."""
        raise NotImplementedError()

    @jaxtyped
    @typechecked
    def evaluate_density(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """Evaluates the density function parameterized by SDF at the given points."""

        if True:
            density = self.primitive.evaluate_density(coords)
        else:
            raise NotImplementedError()

        return density

    @jaxtyped
    @typechecked
    def evaluate_sdf(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """Evaluates the signed distance function at the given points."""

        if True:
            signed_distance = self.primitive.evaluate_sdf(coords)
        else:
            raise NotImplementedError()

        return signed_distance

    @jaxtyped
    @typechecked
    def evaluate_sdf_gradient(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point 3"]:
        """Evaluates the gradient of the signed distance function at the given points."""

        if True:
            gradients = self.primitive.evaluate_sdf_gradient(coords)
        else:
            raise NotImplementedError()

        return gradients
