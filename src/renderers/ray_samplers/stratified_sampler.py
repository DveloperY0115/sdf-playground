"""
stratified_sampler.py

Implementation of the stratified sampling algorithm proposed in
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Mildenhall et al., ECCV 2020.
"""

from dataclasses import dataclass, field
from typing import Optional, Type
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from typeguard import typechecked

from src.cameras.rays import RayBundle, RaySamples
from src.renderers.ray_samplers.base_sampler import RaySampler, RaySamplerConfig
from src.renderers.ray_samplers.uniform_sampler import UniformSamplerConfig
from src.renderers.ray_samplers.pdf_sampler import PDFSamplerConfig

@dataclass
class StratifiedSamplerConfig(RaySamplerConfig):
    """The configuration of a stratified sampler"""

    _target: Type = field(default_factory=lambda: StratifiedSampler)

    num_sample_coarse: int = 64
    """Number of samples to generate in the coarse sampling stage"""
    num_sample_fine: int = 128
    """Number of sampels to generate in the fine sampling stage"""


class StratifiedSampler(RaySampler):
    """
    A stratified sampler class.
    """

    config: StratifiedSamplerConfig

    def __init__(self, config: StratifiedSamplerConfig) -> None:
        super().__init__(config)

        self.num_sample_coarse = self.config.num_sample_coarse
        self.num_sample_fine = self.config.num_sample_fine

        # configure samplers
        self.uniform_sampler = UniformSamplerConfig(
            num_sample=self.num_sample_coarse,
        ).setup()
        self.pdf_sampler = PDFSamplerConfig(
            num_sample=self.num_sample_fine,
        ).setup()

    @jaxtyped
    @typechecked
    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        weights: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
        t_samples: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
    ) -> RaySamples:
        """
        Samples points along rays.
        """
        if not weights is None:
            assert not t_samples is None, (
                "Previous samples must be provided."
            )
            ray_samples = self.pdf_sampler.sample_along_rays(
                weights,
                t_samples,
            )
        else:
            ray_samples = self.uniform_sampler.sample_along_rays(ray_bundle)

        return ray_samples
