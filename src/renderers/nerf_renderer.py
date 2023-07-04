"""
nerf_renderer.py

A renderer that implements an adaptation of volume rendering algorithm proposed in
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Mildenhall et al., ECCV 2020.
"""

from dataclasses import dataclass, field
from typing import Type

from jaxtyping import Shaped, jaxtyped
from torch import Tensor
from typeguard import typechecked

from src.cameras.rays import RayBundle
from src.renderers.base_renderer import Renderer, RendererConfig


@dataclass
class NeRFRendererConfig(RendererConfig):
    """The configuration of SDF renderer adapted from NeRF's rendering algorithm"""

    _target: Type = field(default_factory=lambda: NeRFRenderer)


class NeRFRenderer(Renderer):
    """The renderer class implementing NeRF's rendering algorithm"""

    config: NeRFRendererConfig

    def __init__(
        self,
        config: NeRFRendererConfig,
    ) -> None:
        super().__init__(config)

    @jaxtyped
    @typechecked
    def render(
        self,
        scene: Type,
        ray_bundle: RayBundle,
    ) -> Shaped[Tensor, "num_ray 3"]:
        pass

    @jaxtyped
    @typechecked
    def compute_density_from_sdf(
        self,
        sdf: Shaped[Tensor, "num_ray 1"],
    ) -> Shaped[Tensor, "num_ray 1"]:
        pass
