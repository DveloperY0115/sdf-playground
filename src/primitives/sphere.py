"""
sphere.py

The sphere class.
"""

from dataclasses import dataclass, field
from typing import Type
from jaxtyping import Shaped, jaxtyped

import torch
from torch import Tensor
from typeguard import typechecked

from src.primitives.base_primitive import Primitive, PrimitiveConfig

@dataclass
class SphereConfig(PrimitiveConfig):
    """The configuration of a sphere"""

    _target: Type = field(default_factory=lambda: Sphere)

    center: Shaped[Tensor, "3"] = torch.zeros(3)
    """Center of sphere"""
    radius: float = 1.0
    """Radius of sphere"""


class Sphere(Primitive):
    """A 2-Sphere"""

    config: SphereConfig

    def __init__(
        self,
        config: SphereConfig,
    ) -> None:
        super().__init__(config)

        self.center: Shaped[Tensor, "3"] = self.config.center
        self.radius = self.config.radius

    @jaxtyped
    @typechecked
    def evaluate_density(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """
        Evaluates the density field derived from the 
        signed distance field defined by a sphere
        specified by its center and radius.
        """

        signed_distance = self.evaluate_sdf(coords)

        ####
        # implementation brought from
        # https://github.com/QianyiWu/objsdf/blob/main/code/model/density.py#L16
        # 
        # TODO: Replace the hard-coded number 'beta'
        # TODO: Make an individual module for this types of conversion
        beta = 1e-8
        alpha = 1 / beta
        density =  alpha * (0.5 + 0.5 * signed_distance.sign() * torch.expm1(-signed_distance.abs() / beta))
        ####

        return density

    @jaxtyped
    @typechecked
    def evaluate_sdf(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """
        Evaluates the signed distance field defined by
        a sphere specified by its center and radius.
        """

        center = self.center[None, :].to(coords)
        radius = self.radius

        distance = torch.sqrt(
            torch.sum((coords - center) ** 2, dim=1)
        )
        signed_distance = distance - radius

        return signed_distance

    @jaxtyped
    @typechecked
    @torch.enable_grad()
    def evaluate_sdf_gradient(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point 3"]:
        """
        Evaluates the gradient field of the signed distance field
        defined by a sphere specified by its center and radius.

        While the gradient of the signed distance field of a sphere
        has a closed form, we use automatic differentiation to compute it.
        """
        coords_ = coords.clone()
        coords_.requires_grad = True

        signed_distance = self.evaluate_sdf(coords_)

        grad_outputs = torch.ones_like(
            signed_distance,
            requires_grad=False,
            device=signed_distance.device,
        )
        gradients = torch.autograd.grad(
            outputs=signed_distance,
            inputs=coords_,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=False,
        )[0]

        return gradients
