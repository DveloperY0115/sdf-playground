"""
volsdf_renderer.py

A renderer that implements the volume rendering algorithm proposed in
Volume Rendering of Neural Implicit Surfaces, Yariv et al., NeurIPS 2021.
"""

from dataclasses import dataclass, field
from typing import Tuple, Type

from jaxtyping import Float, Int, Shaped, jaxtyped
import torch
from torch import Tensor
from typeguard import typechecked

from src.cameras.perspective_camera import PerspectiveCamera
from src.renderers.ray_samplers.stratified_sampler import StratifiedSamplerConfig
from src.renderers.base_renderer import Renderer, RendererConfig


@dataclass
class VolSDFRendererConfig(RendererConfig):
    """The configuration of a VolSDF renderer"""

    _target: Type = field(default_factory=lambda: VolSDFRenderer)

    ray_sampler_config: StratifiedSamplerConfig = StratifiedSamplerConfig()
    """The configuration of the ray sampler used in the renderer"""


class VolSDFRenderer(Renderer):
    """The renderer class implementing VolSDF algorithm"""

    config: VolSDFRendererConfig

    def __init__(
        self,
        config: VolSDFRendererConfig,
    ) -> None:
        super().__init__(config)

        self.ray_sampler = config.ray_sampler_config.setup()

    @jaxtyped
    @typechecked
    def render(
        self,
        scene,
        camera: PerspectiveCamera,
        pixel_indices: Int[Tensor, "num_ray"],
    ) -> Tuple[
        Shaped[Tensor, "num_ray"],
        Shaped[Tensor, "num_ray 3"]
    ]:
        """
        Renders the scene using the VolSDF renderer.

        Args:
            scene: The scene to render.
            ray_bundle: The ray bundle used to specify ray origin, direction, and bounds.

        Returns:
            depth_map: The depth map of the scene.
            normal_map: The normal map of the scene.
        """

        # generate rays
        ray_bundle = camera.generate_rays()
        ray_bundle.origins = ray_bundle.origins[pixel_indices, :]
        ray_bundle.directions = ray_bundle.directions[pixel_indices, :]
        ray_bundle.nears = ray_bundle.nears[pixel_indices]
        ray_bundle.fars = ray_bundle.fars[pixel_indices]

        # sample points along rays
        ray_samples = self.ray_sampler.sample_along_rays(ray_bundle)
        sample_points = ray_samples.compute_sample_coordinates()
        num_ray, num_sample, _ = sample_points.shape

        # query densities at sample points
        densities = scene.evaluate_density(sample_points.reshape(-1, 3))
        densities = densities.reshape(num_ray, num_sample)

        # query normal maps at sample points
        normals = scene.evaluate_sdf_gradient(sample_points.reshape(-1, 3))
        normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
        normals = normals.reshape(num_ray, num_sample, 3)

        # compute distance between adjacent samples
        deltas = ray_samples.compute_deltas()

        # evaluate weights for the volume rendering integral
        weights, alphas, transmittances = self.compute_weights_from_densities(
            densities,
            deltas,
        )

        # render depth map, normal map
        t_samples = ray_samples.t_samples
        depth_map = torch.sum(weights * t_samples, dim=1)
        normal_map = torch.sum(weights[..., None] * normals, dim=1)

        return depth_map, normal_map

    @jaxtyped
    @typechecked
    def compute_weights_from_densities(
        self,
        densities: Float[Tensor, "num_ray num_sample"],
        deltas: Float[Tensor, "num_ray num_sample"],
    ) -> Tuple[
        Float[Tensor, "num_ray num_sample"],
        Float[Tensor, "num_ray num_sample"],
        Float[Tensor, "num_ray num_sample"],
    ]:
        """
        Computes the contribution of each sample on a ray.

        Args:
            densities: Densities at sample points on rays.

        Returns:
            weights: The weights of each sample on a ray.
            alphas: The opacity values of each sample on a ray.
            transmittances: The transmittance values of each sample on a ray.
        """

        num_ray = densities.shape[0]
        device = densities.device

        density_delta = densities * deltas

        # compute transmittance T_i
        transmittances = torch.exp(
            -torch.cumsum(
                torch.cat(
                    [torch.zeros((num_ray, 1), device=device), density_delta],
                    dim=1,
                ),
                dim=1,
            )[..., :-1]
        )

        # compute opacity alpha_i
        alphas = 1.0 - torch.exp(-density_delta)

        # compute weight w_i = T_i * alpha_i
        weights = transmittances * alphas

        return weights, alphas, transmittances
