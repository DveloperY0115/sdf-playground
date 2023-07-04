"""
test_cases.py

A set of test cases for perspective camera.
"""

import math
from pathlib import Path

import torch

from src.cameras.perspective_camera import PerspectiveCamera


def test_create_camera(out_dir: Path):
    """Creates a perspective camera and checks its parameters"""

    # test arguments
    camera_to_world = torch.tensor(
        [
            [
                -0.9999999403953552,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                -0.7341099977493286,
                0.6790305972099304,
                2.737260103225708
            ],
            [
                0.0,
                0.6790306568145752,
                0.7341098785400391,
                2.959291696548462
            ],
        ],
    )
    focal_x = 505.0
    focal_y = 505.0
    center_x = 200.0
    center_y = 200.0
    near = 2.0
    far = 6.0
    image_height = 400
    image_width = 400
    device = torch.device("cpu")

    # create camera
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

    # test values
    assert torch.allclose(camera.camera_to_world, camera_to_world)
    assert camera.focal_x == focal_x
    assert camera.focal_y == focal_y
    assert camera.center_x == center_x
    assert camera.center_y == center_y
    assert camera.near == near
    assert camera.far == far
    assert camera.image_height == image_height
    assert camera.image_width == image_width
    assert camera.device == device
