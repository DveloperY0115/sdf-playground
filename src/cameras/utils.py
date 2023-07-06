"""
utils.py

A set of utilities related to cameras.
"""

import numpy as np
import torch

from jaxtyping import Float, jaxtyped
import torch
from torch import Tensor
from typeguard import typechecked


@jaxtyped
@typechecked
def compute_lookat_matrix(
    camera_position: Float[Tensor, "3"],
    origin: Float[Tensor, "3"],
    up_vector: Float[Tensor, "3"] = torch.tensor([0.0, 1.0, 0.0]),
) -> Float[Tensor, "3 4"]:
    """
    Computes a camera pose matrix given the camera position and the origin.

    The coordinate frame is defined in a way that the look-at vector,
    which is the vector from the camera position to the origin, is aligned
    with the negative z-axis of the camera coordinate frame.
    """

    # compute z-axis from the inverted "look-at" vector
    z_axis = camera_position - origin
    z_axis = z_axis / torch.sqrt(torch.sum(z_axis ** 2))

    # compute x-axis by finding the vector orthogonal to up and z-axis
    x_axis = torch.cross(up_vector, z_axis)
    x_axis = x_axis / torch.sqrt(torch.sum(x_axis ** 2))

    # compute y-axis by finding the vector orthogonal to z-axis and x-axis
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.sqrt(torch.sum(y_axis ** 2))

    # construct the camera pose matrix
    camera_to_world = torch.stack([x_axis, y_axis, z_axis], dim=1)
    camera_to_world = torch.cat([camera_to_world, camera_position[:, None]], dim=1)

    return camera_to_world

@jaxtyped
@typechecked
def convert_spherical_to_cartesian(
    radius: float,
    azimuth: float,
    elevation: float,
) -> Float[Tensor, "3"]:
    """
    Converts a point in spherical coordinates to cartesian coordinates.

    Args:
        radius: The radius of the sphere.
        azimuth: The azimuth angle in radian.
        elevation: The elevation angle in radian.
    
    Returns:
        The Cartesian coordinate (x, y, z) of the point specified by
        the given spherical coordinates.
    """

    # check arguments
    assert radius > 0.0, f"{radius:.3f}"
    assert 0 <= azimuth <= 2 * np.pi, f"{azimuth:.3f}"
    assert -np.pi / 2 <= elevation <= np.pi / 2, f"{elevation:.3f}"

    theta: float = np.pi / 2 - elevation
    phi: float = azimuth

    x = radius * np.sin(theta) * np.sin(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.cos(phi)
    position = torch.tensor([x, y, z])

    return position

@jaxtyped
@typechecked
def sample_trajectory_along_upper_hemisphere(
    radius: float,
    elevation: float,
    num_step: int,
) -> Float[Tensor, "num_step 3"]:
    """
    Samples camera positions along the upper hemisphere of a sphere.

    Args:
        radius: The radius of the sphere.
        elevation: The elevation angle in radian.
        num_step: The number of azimuth steps to sample.
    """

    # sample azimuth values
    azimuths = torch.linspace(0, 2 * np.pi, num_step + 1)[:-1]

    # compute Cartesian coordinates at each sample point
    positions = torch.zeros((num_step, 3))
    for index, azimuth in enumerate(azimuths):
        position = convert_spherical_to_cartesian(radius, azimuth.item(), elevation)
        positions[index] = position

    return positions
