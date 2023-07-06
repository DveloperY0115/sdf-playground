"""
superquadric.py

The superquadric class.
"""

from dataclasses import dataclass, field
from typing import Type
from jaxtyping import Shaped, jaxtyped

import torch
from torch import Tensor
from typeguard import typechecked

from src.primitives.base_primitive import Primitive, PrimitiveConfig


@dataclass
class SuperquadricConfig(PrimitiveConfig):
    """The configuration of a superquadric"""

    _target: Type = field(default_factory=lambda: Superquadric)

    center: Shaped[Tensor, "3"] = torch.zeros(3)
    """Center of superquadric"""
    orientation: Shaped[Tensor, "3 3"] = torch.eye(3)
    """Orientation of superquadric"""
    epsilons: Shaped[Tensor, "2"] = torch.ones(2)
    """Powers involved in the equation of superquadric"""
    scales: Shaped[Tensor, "3"] = torch.ones(3)
    """Scale of superquadric along local, canonical axes"""
    truncation: float = 0.0
    """The truncated signed distance of the superquadric"""


class Superquadric(Primitive):
    """A superquadric"""

    config: SuperquadricConfig

    def __init__(
        self,
        config: SuperquadricConfig,
    ) -> None:
        super().__init__(config)

        # register shape parameters
        self.center: Shaped[Tensor, "3"] = self.config.center
        self.orientation: Shaped[Tensor, "3 3"] = self.config.orientation
        self.epsilons: Shaped[Tensor, "2"] = self.config.epsilons
        self.scales: Shaped[Tensor, "3"] = self.config.scales
        self.truncation: float = self.config.truncation

        # check whether the given rotation matrix is valid
        determinant = torch.linalg.det(self.orientation)
        assert torch.isclose(determinant, torch.ones(1))

        # check whether scales are positive
        assert torch.all(self.scales > 0.0)


    @jaxtyped
    @typechecked
    def evaluate_density(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point"]:
        """
        Evaluates the density field derived from the 
        signed distance field defined by a superquadric
        specified by its shape parameters.
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
        a superquadric specified by its shape parameters.
        """

        center: Shaped[Tensor, "1 3"] = self.center[None, ...].to(coords)
        orientation: Shaped[Tensor, "3 3"] = self.orientation.to(coords)
        eps1, eps2 = self.epsilons[0].to(coords), self.epsilons[1].to(coords)
        scale_x, scale_y, scale_z = self.scales[0], self.scales[1], self.scales[2]

        # world -> local
        coords = (coords - center) @ orientation

        coords_norm = torch.sqrt(torch.sum(coords ** 2, dim=-1))

        scale = ((((coords[:, 0] / scale_x) ** 2) ** (1 / eps2) + \
        ((coords[:, 1] / scale_y) ** 2) ** (1 / eps2)) ** (eps2 / eps1) + \
        ((coords[:, 2] / scale_z) ** 2) ** (1 / eps1)) ** (-eps1 / 2)

        signed_distance = coords_norm * (1.0 - scale)

        # SDF -> TSDF
        if self.truncation != 0:
            signed_distance = torch.minimum(
                torch.maximum(
                    signed_distance,
                    -self.truncation * torch.ones_like(signed_distance),
                ),
                self.truncation * torch.ones_like(signed_distance),
            )

        return signed_distance

    @jaxtyped
    @typechecked
    def evaluate_sdf_gradient(
        self,
        coords: Shaped[Tensor, "num_point 3"],
    ) -> Shaped[Tensor, "num_point 3"]:
        """
        Evaluates the gradient field of the signed distance field
        defined by a superquadric specified by its shape parameters.

        We use automatic differentiation to compute it.
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
