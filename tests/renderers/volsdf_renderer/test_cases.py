"""
test_cases.py

A set of test cases for VolSDFRenderer.
"""

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from src.cameras.utils import compute_lookat_matrix
from src.cameras.perspective_camera import PerspectiveCamera
from src.primitives.sphere import SphereConfig
from src.primitives.superquadric import SuperquadricConfig
from src.renderers.ray_samplers.stratified_sampler import (
    StratifiedSampler,
    StratifiedSamplerConfig,
)
from src.renderers.volsdf_renderer import VolSDFRendererConfig


def create_camera_for_test(camera_position, origin):

    camera_to_world = compute_lookat_matrix(
        camera_position,
        origin,
    )
    focal_x = 505.0
    focal_y = 505.0
    center_x = 200.0
    center_y = 200.0
    near = 1.0
    far = 10.0
    image_height = 400
    image_width = 400
    device = torch.device("cpu")

    camera = PerspectiveCamera(
        camera_to_world=camera_to_world,
        focal_x=focal_x,
        focal_y=focal_y,
        center_x=center_x,
        center_y=center_y,
        near=near,
        far=far,
        image_height=image_height,
        image_width=image_width,
        device=device,
    )

    return camera

def test_create_renderer_stratified_sampler(out_dir: Path):
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

def test_quadrature_evaluation(out_dir: Path):
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
                torch.tensor([1e10]).unsqueeze(0).repeat(dists.shape[0], 1)
            ],
            -1,
        )

        # LOG SPACE
        free_energy = dists * densities
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights, alpha, transmittance

    weights_ref, alphas_ref, transmittances_ref = _generate_results_reference()

    # test values
    assert torch.allclose(weights_ours, weights_ref)
    assert torch.allclose(alphas_ours, alphas_ref)
    assert torch.allclose(transmittances_ours, transmittances_ref)

def test_render_sphere(out_dir: Path):
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

    # create a camera
    camera = create_camera_for_test(
        camera_position=torch.tensor([3.0, 3.0, 3.0]),
        origin=torch.tensor([0.0, 0.0, 0.0]),
    )
    pixel_indices = torch.arange(
        0,
        camera.image_height * camera.image_width,
        dtype=torch.long,
    )

    # render depth map, normal map
    depth_map, normal_map = renderer.render(
        sphere,
        camera,
        pixel_indices,
    )
    depth_map = depth_map.reshape(
        camera.image_height,
        camera.image_width,
    )
    normal_map = normal_map.reshape(
        camera.image_height,
        camera.image_width,
        3
    ).detach()

    # inspect the distribution of depth values
    plt.hist(depth_map.flatten().numpy(), bins=100)
    plt.savefig(out_dir / "depth_values.png")

    # visualize depth map
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
    depth_map = torch.clamp(depth_map, 0.0, 1.0)
    depth_map = depth_map.numpy()
    depth_map = (depth_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "depth_map.png"),
        depth_map,
    )

    # visualize normal map
    normal_map = (normal_map + 1.0) * 0.5
    normal_map = normal_map.numpy()
    normal_map = (normal_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "normal_map.png"),
        normal_map,
    )

def test_render_superquadric1(out_dir: Path):
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

    # create a camera
    camera = create_camera_for_test(
        camera_position=torch.tensor([3.0, 3.0, 3.0]),
        origin=torch.tensor([0.0, 0.0, 0.0]),
    )
    pixel_indices = torch.arange(
        0,
        camera.image_height * camera.image_width,
        dtype=torch.long,
    )

    # render depth map, normal map
    depth_map, normal_map = renderer.render(
        superqudric,
        camera,
        pixel_indices,
    )
    depth_map = depth_map.reshape(
        camera.image_height,
        camera.image_width,
    )
    normal_map = normal_map.reshape(
        camera.image_height,
        camera.image_width,
        3
    ).detach()

    # inspect the distribution of depth values
    plt.hist(depth_map.flatten().numpy(), bins=100)
    plt.savefig(out_dir / "depth_values.png")

    # visualize depth map
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
    depth_map = torch.clamp(depth_map, 0.0, 1.0)
    depth_map = depth_map.numpy()
    depth_map = (depth_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "depth_map.png"),
        depth_map,
    )

    # visualize normal map
    normal_map = (normal_map + 1.0) * 0.5
    normal_map = normal_map.numpy()
    normal_map = (normal_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "normal_map.png"),
        normal_map,
    )

def test_render_superquadric2(out_dir: Path):
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
        scales=torch.tensor([1.0, 1.0, 2.0]),
    )
    superqudric = superquadric_config.setup()

    # create a camera
    camera = create_camera_for_test(
        camera_position=torch.tensor([4.0, 4.0, 4.0]),
        origin=torch.tensor([0.0, 0.0, 0.0]),
    )
    pixel_indices = torch.arange(
        0,
        camera.image_height * camera.image_width,
        dtype=torch.long,
    )

    # render depth map, normal map
    depth_map, normal_map = renderer.render(
        superqudric,
        camera,
        pixel_indices,
    )
    depth_map = depth_map.reshape(
        camera.image_height,
        camera.image_width,
    )
    normal_map = normal_map.reshape(
        camera.image_height,
        camera.image_width,
        3
    ).detach()

    # inspect the distribution of depth values
    plt.hist(depth_map.flatten().numpy(), bins=100)
    plt.savefig(out_dir / "depth_values.png")

    # visualize depth map
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
    depth_map = torch.clamp(depth_map, 0.0, 1.0)
    depth_map = depth_map.numpy()
    depth_map = (depth_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "depth_map.png"),
        depth_map,
    )

    # visualize normal map
    normal_map = (normal_map + 1.0) * 0.5
    normal_map = normal_map.numpy()
    normal_map = (normal_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "normal_map.png"),
        normal_map,
    )

def test_render_superquadric3(out_dir: Path):
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
        epsilons=torch.tensor([3.0, 1.0]),
        scales=torch.tensor([1.0, 1.0, 2.0]),
    )
    superqudric = superquadric_config.setup()

    # create a camera
    camera = create_camera_for_test(
        camera_position=torch.tensor([4.0, 4.0, 4.0]),
        origin=torch.tensor([0.0, 0.0, 0.0]),
    )
    pixel_indices = torch.arange(
        0,
        camera.image_height * camera.image_width,
        dtype=torch.long,
    )

    # render depth map, normal map
    depth_map, normal_map = renderer.render(
        superqudric,
        camera,
        pixel_indices,
    )
    depth_map = depth_map.reshape(
        camera.image_height,
        camera.image_width,
    )
    normal_map = normal_map.reshape(
        camera.image_height,
        camera.image_width,
        3
    ).detach()

    # inspect the distribution of depth values
    plt.hist(depth_map.flatten().numpy(), bins=100)
    plt.savefig(out_dir / "depth_values.png")

    # visualize depth map
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min) + 1e-6
    depth_map = torch.clamp(depth_map, 0.0, 1.0)
    depth_map = depth_map.numpy()
    depth_map = (depth_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "depth_map.png"),
        depth_map,
    )

    # visualize normal map
    normal_map = (normal_map + 1.0) * 0.5
    normal_map = normal_map.numpy()
    normal_map = (normal_map * 255.0).astype(int)
    cv2.imwrite(
        str(out_dir / "normal_map.png"),
        normal_map,
    )
