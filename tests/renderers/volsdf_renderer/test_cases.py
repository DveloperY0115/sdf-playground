"""
test_cases.py

A set of test cases for VolSDFRenderer.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from src.primitives.sphere import SphereConfig
from src.primitives.superquadric import SuperquadricConfig
from src.renderers.ray_samplers.stratified_sampler import (
    StratifiedSampler,
    StratifiedSamplerConfig,
)
from src.renderers.volsdf_renderer import VolSDFRendererConfig
from tests.renderers.volsdf_renderer.utils import (
    create_camera_for_test,
    render_primitive_from_spherical_poses,
)
from src.utils.images import create_video_from_images


def test_create_renderer_stratified_sampler(out_dir: Path, device: torch.device):
    """Creates a renderer and checks its default parameters"""

    config = VolSDFRendererConfig(
        ray_sampler_config=StratifiedSamplerConfig(
            num_sample_coarse=64,
            num_sample_fine=128,
        ),
    )
    renderer = config.setup()

    assert isinstance(renderer.ray_sampler, StratifiedSampler)
    assert renderer.ray_sampler.num_sample_coarse == 64
    assert renderer.ray_sampler.num_sample_fine == 128

def test_quadrature_evaluation(out_dir: Path, device: torch.device):
    """Tests the quadrature evaluation of VolSDFRenderer"""

    # create renderer
    renderer_config = VolSDFRendererConfig(
        ray_sampler_config=StratifiedSamplerConfig(
            num_sample_coarse=64,
            num_sample_fine=128,
        ),
    )
    renderer = renderer_config.setup()

    # create a sphere
    sphere_config = SphereConfig(
        center=torch.tensor([0.0, 0.0, 0.0]),
        radius=1.0,
    )
    sphere = sphere_config.setup()

    # create camera
    camera = create_camera_for_test(
        camera_position=torch.tensor([0.0, 0.0, 5.0]),
        origin=torch.tensor([0.0, 0.0, 0.0]),
        image_height=400,
        image_width=400,
        device=device,
    )
    pixel_indices = torch.arange(
        0,
        camera.image_height * camera.image_width,
        dtype=torch.long,
    )

    # generate rays
    ray_bundle = camera.generate_rays()
    ray_bundle.origins = ray_bundle.origins[pixel_indices, :]
    ray_bundle.directions = ray_bundle.directions[pixel_indices, :]
    ray_bundle.nears = ray_bundle.nears[pixel_indices]
    ray_bundle.fars = ray_bundle.fars[pixel_indices]

    # sample points along rays
    ray_samples = renderer.ray_sampler.sample_along_rays(ray_bundle)
    sample_points = ray_samples.compute_sample_coordinates()
    t_samples = ray_samples.t_samples
    num_ray, num_sample, _ = sample_points.shape

    # query densities at sample points
    densities = sphere.evaluate_density(sample_points.reshape(-1, 3))
    densities = densities.reshape(num_ray, num_sample)


    def _generate_results_ours():

        # compute distance between adjacent samples
        deltas = ray_samples.compute_deltas()

        weights, alphas, transmittances = renderer.compute_weights_from_densities(
            densities,
            deltas,
        )

        return weights, alphas, transmittances

    weights_ours, alphas_ours, transmittances_ours = _generate_results_ours()

    def _generate_results_reference():

        dists = t_samples[:, 1:] - t_samples[:, :-1]
        dists = torch.cat(
            [
                dists,
                torch.tensor([1e10]).expand(dists.shape[0], 1).to(dists)
            ],
            -1,
        )

        # LOG SPACE
        free_energy = dists * densities
        shifted_free_energy = torch.cat(
            [
                torch.zeros(dists.shape[0], 1, device=dists.device),
                free_energy[:, :-1],
            ],
            dim=-1,
        )
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance

        return weights, alpha, transmittance

    weights_ref, alphas_ref, transmittances_ref = _generate_results_reference()

    # test values
    assert torch.allclose(weights_ours, weights_ref)
    assert torch.allclose(alphas_ours, alphas_ref)
    assert torch.allclose(transmittances_ours, transmittances_ref)

def test_render_sphere(out_dir: Path, device: torch.device):
    """Tests the rendering method of VolSDFRenderer"""

    # create renderer
    renderer_config = VolSDFRendererConfig(
        ray_sampler_config=StratifiedSamplerConfig(
            num_sample_coarse=64,
            num_sample_fine=128,
        ),
    )
    renderer = renderer_config.setup()

    # create a sphere
    sphere_config = SphereConfig(
        center=torch.tensor([0.0, 0.0, 0.0]),
        radius=1.0,
    )
    sphere = sphere_config.setup()

    # render primitive
    depth_maps, normal_maps = render_primitive_from_spherical_poses(
        sphere,
        renderer,
        camera_radius=7.0,
        camera_elevation=math.pi / 4,
        num_azimuth_step=36,
        image_height=400,
        image_width=400,
        device=device,
    )

    # save rendering outputs
    for camera_index, (depth_map, normal_map) in enumerate(
        zip(depth_maps, normal_maps)
    ):
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
        depth_map = np.clip(depth_map, 0.0, 1.0)
        depth_map = (depth_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_depth_map.png"),
            depth_map,
        )

        normal_map = (normal_map + 1.0) * 0.5
        normal_map = (normal_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_normal_map.png"),
            normal_map,
        )

def test_render_superquadric1(out_dir: Path, device: torch.device):
    """Tests the rendering method of VolSDFRenderer"""

    # create renderer
    renderer_config = VolSDFRendererConfig(
        ray_sampler_config=StratifiedSamplerConfig(
            num_sample_coarse=64,
            num_sample_fine=128,
        ),
    )
    renderer = renderer_config.setup()

    # create a superquadric
    superquadric_config = SuperquadricConfig(
        scales=torch.tensor([1.5, 1.0, 1.25]),
    )
    superqudric = superquadric_config.setup()

    # render primitive
    depth_maps, normal_maps = render_primitive_from_spherical_poses(
        superqudric,
        renderer,
        camera_radius=7.0,
        camera_elevation=math.pi / 4,
        num_azimuth_step=36,
        image_height=400,
        image_width=400,
        device=device,
    )

    # save rendering outputs
    for camera_index, (depth_map, normal_map) in enumerate(
        zip(depth_maps, normal_maps)
    ):
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
        depth_map = np.clip(depth_map, 0.0, 1.0)
        depth_map = (depth_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_depth_map.png"),
            depth_map,
        )

        normal_map = (normal_map + 1.0) * 0.5
        normal_map = (normal_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_normal_map.png"),
            normal_map,
        )

def test_render_superquadric2(out_dir: Path, device: torch.device):
    """Tests the rendering method of VolSDFRenderer"""

    # create renderer
    renderer_config = VolSDFRendererConfig(
        ray_sampler_config=StratifiedSamplerConfig(
            num_sample_coarse=64,
            num_sample_fine=128,
        ),
    )
    renderer = renderer_config.setup()

    # create a superquadric
    superquadric_config = SuperquadricConfig(
        epsilons=torch.tensor([0.1, 0.1]),
        scales=torch.tensor([1.0, 2.0, 1.0]),
    )
    superqudric = superquadric_config.setup()

    # render primitive
    depth_maps, normal_maps = render_primitive_from_spherical_poses(
        superqudric,
        renderer,
        camera_radius=7.0,
        camera_elevation=math.pi / 4,
        num_azimuth_step=36,
        image_height=400,
        image_width=400,
        device=device,
    )

    # save rendering outputs
    depth_processed = []
    normal_processed = []
    for camera_index, (depth_map, normal_map) in enumerate(
        zip(depth_maps, normal_maps)
    ):
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
        depth_map = np.clip(depth_map, 0.0, 1.0)
        depth_map = (depth_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_depth_map.png"),
            depth_map,
        )
        depth_processed.append(depth_map)

        normal_map = (normal_map + 1.0) * 0.5
        normal_map = (normal_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_normal_map.png"),
            normal_map,
        )
        normal_processed.append(normal_map)

    depth_processed = np.stack(depth_processed, axis=0)[..., None]
    normal_processed = np.stack(normal_processed, axis=0)

    create_video_from_images(
        depth_processed,
        out_dir / "depth_sequence.mp4",
    )

def test_render_superquadric3(out_dir: Path, device: torch.device):
    """Tests the rendering method of VolSDFRenderer"""

    # create renderer
    renderer_config = VolSDFRendererConfig(
        ray_sampler_config=StratifiedSamplerConfig(
            num_sample_coarse=64,
            num_sample_fine=128,
        ),
    )
    renderer = renderer_config.setup()

    # create a superquadric
    superquadric_config = SuperquadricConfig(
        epsilons=torch.tensor([2.5, 0.1]),
        scales=torch.tensor([1.0, 1.0, 2.0]),
    )
    superqudric = superquadric_config.setup()

    # render primitive
    depth_maps, normal_maps = render_primitive_from_spherical_poses(
        superqudric,
        renderer,
        camera_radius=7.0,
        camera_elevation=math.pi / 4,
        num_azimuth_step=36,
        image_height=400,
        image_width=400,
        device=device,
    )

    # save rendering outputs
    depth_processed = []
    normal_processed = []
    for camera_index, (depth_map, normal_map) in enumerate(
        zip(depth_maps, normal_maps)
    ):
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
        depth_map = np.clip(depth_map, 0.0, 1.0)
        depth_map = (depth_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_depth_map.png"),
            depth_map,
        )
        depth_processed.append(depth_map)

        normal_map = (normal_map + 1.0) * 0.5
        normal_map = (normal_map * 255.0).astype(int)
        cv2.imwrite(
            str(out_dir / f"{camera_index:03d}_normal_map.png"),
            normal_map,
        )
        normal_processed.append(normal_map)

    depth_processed = np.stack(depth_processed, axis=0)[..., None]
    normal_processed = np.stack(normal_processed, axis=0)

    create_video_from_images(
        depth_processed,
        out_dir / "depth_sequence.mp4",
    )
