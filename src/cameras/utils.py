"""
utils.py

A set of utilities related to cameras.
"""

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
