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
    ray_batch_size: int = 4096
    """Number of rays included in a single ray batch during training"""


class VolSDFRenderer(Renderer):
    """The renderer class implementing VolSDF algorithm"""

    config: VolSDFRendererConfig

    def __init__(
        self,
        config: VolSDFRendererConfig,
    ) -> None:
        super().__init__(config)

        self.ray_sampler = config.ray_sampler_config.setup()
        self.ray_batch_size = config.ray_batch_size

    @jaxtyped
    @typechecked
    def render(
        self,
        scene,
        camera: PerspectiveCamera,
        pixel_indices: Int[Tensor, "num_ray"],
    ) -> Tuple[
        Shaped[Tensor, "num_ray 3"],
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
        rgb_image, depth_map, normal_map, weights = self.render_ray_batches(
            scene,
            camera,
            ray_samples,
        )

        return rgb_image, depth_map, normal_map

    @jaxtyped
    @typechecked
    def render_ray_batches(
        self,
        scene,
        camera,
        ray_samples,
    ) -> Tuple[
        Shaped[Tensor, "num_ray 3"],
        Shaped[Tensor, "num_ray"],
        Shaped[Tensor, "num_ray 3"],
        Shaped[Tensor, "num_ray num_sample"],
    ]:
        """
        Renders an image by dividing its pixels into small batches.
        """
        rgb_image = []
        depth_map = []
        normal_map = []
        weights = []

        sample_points = ray_samples.compute_sample_coordinates()
        ray_dir = ray_samples.ray_bundle.directions
        t_samples = ray_samples.t_samples
        delta_t = ray_samples.compute_deltas()

        point_chunks = torch.split(sample_points, self.ray_batch_size, dim=0)
        ray_dir_chunks = torch.split(ray_dir, self.ray_batch_size, dim=0)
        t_sample_chunks = torch.split(t_samples, self.ray_batch_size, dim=0)
        delta_chunks = torch.split(delta_t, self.ray_batch_size, dim=0)

        assert len(point_chunks) == len(ray_dir_chunks) == len(delta_chunks), (
            f"{len(point_chunks)} {len(ray_dir_chunks)} {len(delta_chunks)}"
        )

        for point_chunk, ray_dir_chunk, t_sample_chunk, delta_chunk in zip(
            point_chunks, ray_dir_chunks, t_sample_chunks, delta_chunks
        ):

            chunk_size, num_sample, _ = point_chunk.shape

            # query radiances at sample points
            ray_dir_chunk = ray_dir_chunk.unsqueeze(1).repeat(1, num_sample, 1)
            radiance_chunk = scene.evaluate_radiance(
                point_chunk.reshape(-1, 3),
                ray_dir_chunk.reshape(-1, 3),
            )
            radiance_chunk = radiance_chunk.reshape(chunk_size, num_sample, 3)

            # query densities at sample points
            density_chunk = scene.evaluate_density(point_chunk.reshape(-1, 3))
            density_chunk = density_chunk.reshape(chunk_size, num_sample)

            # query normal maps at sample points
            normal_chunk = scene.evaluate_sdf_gradient(point_chunk.reshape(-1, 3))
            normal_chunk = normal_chunk / (
                torch.linalg.norm(normal_chunk, dim=1, keepdim=True) + 1e-8
            )
            normal_chunk = normal_chunk.reshape(chunk_size, num_sample, 3)
            camera_to_world = camera.camera_to_world
            normal_chunk = normal_chunk @ camera_to_world[:3, :3].T

            # evaluate weights for the volume rendering integral
            weights_chunk, alphas_chunk, transmittances_chunk = self.compute_weights_from_densities(
                density_chunk,
                delta_chunk,
            )

            # render
            rgb_chunk = torch.sum(weights_chunk[..., None] * radiance_chunk, dim=1)
            depth_chunk = torch.sum(weights_chunk * t_sample_chunk, dim=1)
            normal_chunk = torch.sum(weights_chunk[..., None] * normal_chunk, dim=1)

            # collect
            rgb_image.append(rgb_chunk)
            depth_map.append(depth_chunk)
            normal_map.append(normal_chunk)
            weights.append(weights_chunk)

        rgb_image = torch.cat(rgb_image, dim=0)
        depth_map = torch.cat(depth_map, dim=0)
        normal_map = torch.cat(normal_map, dim=0)
        weights = torch.cat(weights, dim=0)

        return rgb_image, depth_map, normal_map, weights

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
