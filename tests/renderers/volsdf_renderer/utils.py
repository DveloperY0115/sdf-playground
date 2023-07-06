"""
utils.py

A collection of utilities for testing VolSDFRenderer.
"""

from typing import Tuple

from jaxtyping import Float, jaxtyped
from numpy import ndarray
import torch
from torch import Tensor
from typeguard import typechecked

from src.cameras.perspective_camera import PerspectiveCamera
from src.cameras.utils import compute_lookat_matrix, sample_trajectory_along_upper_hemisphere


@jaxtyped
@typechecked
def create_camera_for_test(
    camera_position,
    origin,
    image_height,
    image_width,
    device,
):
    """
    Creates a camera object for testing.
    """

    # compute camera pose
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

@jaxtyped
@typechecked
def render_primitive_from_spherical_poses(
    primitive,
    renderer,
    camera_radius: float,
    camera_elevation: float,
    num_azimuth_step: int,
    image_height: int,
    image_width: int,
    device: torch.device,
) -> Tuple[
    Float[ndarray, "num_azimuth_step image_height image_width"],
    Float[ndarray, "num_azimuth_step image_height image_width 3"]
]:
    """
    Renders a primitive, assumed to be centered at the origin,
    from a set of spherical poses.
    """

    # sample camera positions
    camera_positions = sample_trajectory_along_upper_hemisphere(
        radius=camera_radius,
        elevation=camera_elevation,
        num_step=num_azimuth_step,
    )

    # iterate over the viewpoints
    # and render depth map, normal map
    depth_maps = torch.zeros(num_azimuth_step, image_height, image_width)
    normal_maps = torch.zeros(num_azimuth_step, image_height, image_width, 3)
    for camera_index, camera_position in enumerate(camera_positions):

        # create a camera
        camera = create_camera_for_test(
            camera_position=camera_position,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            image_height=image_height,
            image_width=image_width,
            device=device,
        )
        pixel_indices = torch.arange(
            0,
            camera.image_height * camera.image_width,
            dtype=torch.long,
        )

        # render depth map, normal map
        depth_map, normal_map = renderer.render(
            primitive,
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
            3,
        ).detach()

        # collect depth map, normal map
        depth_maps[camera_index] = depth_map.cpu()
        normal_maps[camera_index] = normal_map.cpu()

    # torch -> numpy
    depth_maps = depth_maps.numpy()
    normal_maps = normal_maps.numpy()

    return depth_maps, normal_maps
