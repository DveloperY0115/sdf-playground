"""
uniform_sampler.py
"""

from dataclasses import dataclass, field
from typing import Type

from jaxtyping import jaxtyped
import torch
from typeguard import typechecked

from src.cameras.rays import RayBundle, RaySamples
from src.renderers.ray_samplers.base_sampler import RaySampler, RaySamplerConfig


@dataclass
class UniformSamplerConfig(RaySamplerConfig):
    """The configuration of an uniform sampler"""

    _target: Type = field(default_factory=lambda: UniformSampler)

    num_sample: int = 64
    """Number of samples to draw along each ray"""


class UniformSampler(RaySampler):
    """
    An uniform sampler class.
    """

    config: UniformSamplerConfig

    def __init__(self, config: UniformSamplerConfig) -> None:
        super().__init__(config)

        self.num_sample = self.config.num_sample

    @jaxtyped
    @typechecked
    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        **kwargs,
    ) -> RaySamples:
        """
        Samples points along rays.
        """

        # parse ray bundle
        ray_origins = ray_bundle.origins
        num_ray = ray_origins.shape[0]
        device = ray_origins.device
        near = ray_bundle.nears[0].item()
        far = ray_bundle.fars[0].item()

        # generate samples in the parametric domain [0.0, 1.0]
        t_bins = self.create_t_bins(self.num_sample, device)

        # map the samples to the scene domain [t_near, t_far]
        t_bins = self.map_t_to_euclidean(t_bins, near, far)
        t_bins = t_bins.expand(num_ray, -1)

        # uniformly sample points within each interval
        t_mid = 0.5 * (t_bins[..., 1:] + t_bins[..., :-1])
        t_upper = torch.cat([t_mid, t_bins[..., -1:]], -1)
        t_lower = torch.cat([t_bins[..., :1], t_mid], -1)
        noise = torch.rand_like(t_lower)
        t_samples = t_lower + (t_upper - t_lower) * noise

        ray_samples = RaySamples(ray_bundle, t_samples)

        return ray_samples
