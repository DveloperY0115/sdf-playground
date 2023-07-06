"""
pdf_sampler.py
"""

from dataclasses import dataclass, field
from typing import Optional, Type
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from typeguard import typechecked

from src.cameras.rays import RaySamples
from src.renderers.ray_samplers.base_sampler import RaySampler, RaySamplerConfig


@dataclass
class PDFSamplerConfig(RaySamplerConfig):
    """The configuration of a PDF sampler"""

    _target: Type = field(default_factory=lambda: PDFSampler)

    num_sample: int = 64
    """Number of samples to draw along each ray"""


class PDFSampler(RaySampler):
    """
    A PDF sampler class.
    """

    config: PDFSamplerConfig

    def __init__(self, config: PDFSamplerConfig) -> None:
        super().__init__(config)

        self.num_sample = self.config.num_sample

    @jaxtyped
    @typechecked
    def sample_along_rays(
        self,
        ray_samples: RaySamples,
        weights: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
        t_samples: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
    ) -> RaySamples:
        """
        Samples points along rays.
        """
        raise NotImplementedError(
            "Complete this function to return a RaySample instance"
        )

        # NOTE: The elements of 't_samples' are assumed to be ordered.
        t_mid = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        weights_mid = weights[..., 1:-1]
        new_t_samples = self.sample_pdf(t_mid, weights_mid, self.num_sample)

        # combine the new samples with the previous ones.
        # NOTE: The elements of 't_samples' must be sorted.
        t_samples = torch.cat([t_samples, new_t_samples], -1)
        t_samples, _ = torch.sort(t_samples, dim=-1)

        return t_samples

    @jaxtyped
    @typechecked
    def sample_pdf(
        self,
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
