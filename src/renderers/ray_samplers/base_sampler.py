"""
base_sampler.py

The base class for ray samplers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, Union

from jaxtyping import Float, Shaped, jaxtyped
import torch
from typeguard import typechecked

from src.cameras.rays import RayBundle, RaySamples
from src.configs.base_config import InstantiateConfig


@dataclass
class RaySamplerConfig(InstantiateConfig):
    """The configuration of a ray sampler"""

    _target: Type = field(default_factory=lambda: RaySampler)


class RaySampler(ABC):
    """The base class of ray samplers"""

    config: RaySamplerConfig

    def __init__(
        self,
        config: RaySamplerConfig,
    ) -> None:
        self.config = config

    @abstractmethod
    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        **kwargs,
    ) -> RaySamples:
        """Samples points along the given rays."""

    @jaxtyped
    @typechecked
    def map_t_to_euclidean(
        self,
        t_values: Shaped[torch.Tensor, "..."],
        near: float,
        far: float,
    ) -> Shaped[torch.Tensor, "..."]:
        """
        Maps values in the parametric space [0, 1] to Euclidean space.
        """
        return near * (1.0 - t_values) + far * t_values

    @jaxtyped
    @typechecked
    def create_t_bins(
        self,
        num_bin: int,
        device: Union[int, torch.device],
    ) -> Float[torch.Tensor, "num_bin"]:
        """
        Generates samples of t's by subdividing the interval [0.0, 1.0] inclusively.
        """
        assert isinstance(num_bin, int), (
            f"Expected an integer for parameter 'num_samples'. Got a value of type {type(num_bin)}."
        )
        t_bins = torch.linspace(0.0, 1.0, num_bin, device=device)

        return t_bins
