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

    @jaxtyped
    @typechecked
    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        importance_weights: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
        importance_t_samples: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
    ) -> RaySamples:
        """
        Samples points along rays.
        """
        if not importance_weights is None:
            assert not importance_t_samples is None, (
                "Previous samples must be provided."
            )
            t_samples = self.sample_along_rays_importance(
                importance_weights,
                importance_t_samples,
                self.num_sample_fine,
            )
        else:
            t_samples = self.sample_along_rays_uniform(
                ray_bundle,
                self.num_sample_coarse,
            )

        ray_samples = RaySamples(ray_bundle, t_samples)
        return ray_samples

    @jaxtyped
    @typechecked
    def sample_along_rays_uniform(
        self,
        ray_bundle: RayBundle,
        num_sample: int,
    ) -> Float[torch.Tensor, "num_ray num_sample"]:
        """
        Performs uniform sampling of points along rays.
        """

        # parse ray bundle
        ray_origins = ray_bundle.origins
        num_ray = ray_origins.shape[0]
        device = ray_origins.device
        near = ray_bundle.nears[0].item()
        far = ray_bundle.fars[0].item()

        # generate samples in the parametric domain [0.0, 1.0]
        t_bins = self.create_t_bins(num_sample, device)

        # map the samples to the scene domain [t_near, t_far]
        t_bins = self.map_t_to_euclidean(t_bins, near, far)
        t_bins = t_bins.expand(num_ray, -1)

        # uniformly sample points within each interval
        t_mid = 0.5 * (t_bins[..., 1:] + t_bins[..., :-1])
        t_upper = torch.cat([t_mid, t_bins[..., -1:]], -1)
        t_lower = torch.cat([t_bins[..., :1], t_mid], -1)
        noise = torch.rand_like(t_lower)
        t_samples = t_lower + (t_upper - t_lower) * noise

        return t_samples

    @jaxtyped
    @typechecked
    def sample_along_rays_importance(
        self,
        weights: Float[torch.Tensor, "num_ray num_sample"],
        t_samples: Float[torch.Tensor, "num_ray num_sample"],
        num_sample: int,
    ) -> Float[torch.Tensor, "num_ray new_num_sample"]:
        """
        Performs the inverse CDF sampling of points along rays given weights
        indicating the 'importance' of each given sample.
        """

        # NOTE: The elements of 't_samples' are assumed to be ordered.
        t_mid = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        weights_mid = weights[..., 1:-1]
        new_t_samples = sample_pdf(t_mid, weights_mid, num_sample)

        # combine the new samples with the previous ones.
        # NOTE: The elements of 't_samples' must be sorted.
        t_samples = torch.cat([t_samples, new_t_samples], -1)
        t_samples, _ = torch.sort(t_samples, dim=-1)

        return t_samples


@jaxtyped
@typechecked
def sample_pdf(
    bins: Float[torch.Tensor, "num_ray num_bin"],
    weights: Float[torch.Tensor, "num_ray num_weight"],
    num_sample: int,
) -> Float[torch.Tensor, "num_ray num_sample"]:
    """
    Draws samples from the probability density represented by given weights.
    """
    assert bins.shape[1] == weights.shape[1] + 1, f"{bins.shape[1]}, {weights.shape[1]}"

    # construct the PDF
    weights += 1e-5
    normalizer = torch.sum(weights, dim=-1, keepdim=True)
    pdf = weights / normalizer

    # compute the CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # sample from the uniform distribution: U[0, 1)
    cdf_ys = torch.rand(list(cdf.shape[:-1]) + [num_sample], device=cdf.device)
    cdf_ys = cdf_ys.contiguous()

    # inverse CDF sampling
    indices = torch.searchsorted(cdf, cdf_ys, right=True)
    lower = torch.max(torch.zeros_like(indices - 1), indices - 1)
    upper = torch.min(cdf.shape[-1] - 1 * torch.ones_like(indices), indices)

    cdf_lower = torch.gather(cdf, 1, lower)
    cdf_upper = torch.gather(cdf, 1, upper)
    bins_lower = torch.gather(bins, 1, lower)
    bins_upper = torch.gather(bins, 1, upper)

    # approximate the CDF to a linear function within each interval
    denom = cdf_upper - cdf_lower
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (cdf_ys - cdf_lower) / denom  # i.e., cdf_y = cdf_lower + t * (cdf_upper - cdf_lower)
    t_samples = bins_lower + t * (bins_upper - bins_lower)

    assert torch.isnan(t_samples).sum() == 0

    return t_samples
