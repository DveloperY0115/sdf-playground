"""
test_cases.py

A set of test cases for sphere primitive.
"""

import math
from pathlib import Path

import torch

from src.primitives.sphere import SphereConfig


def test_create_sphere(out_dir: Path):
    """Creates a sphere and checks its default parameters"""

    config = SphereConfig()
    sphere = config.setup()

    default_radius = sphere.radius
    default_center = sphere.center

    # test values
    assert default_radius == 1.0
    assert torch.all(default_center == torch.zeros(3))

def test_sdf_inner_1(out_dir: Path):
    """
    Test SDF evaluated inside a sphere.

    The sphere is placed at the origin and has radius 1.
    """

    config = SphereConfig()
    sphere = config.setup()

    sd_computed = sphere.evaluate_sdf(0.5 * torch.eye(3))
    sd_gt = torch.tensor([[-0.5], [-0.5], [-0.5]])

    # test values
    assert torch.allclose(sd_computed, sd_gt)

def test_sdf_inner_2(out_dir: Path):
    """
    Test SDF evaluated inside a sphere.

    The sphere is placed at the point (1, 1, 1) and has radius 2.
    """

    config = SphereConfig(
        center=torch.tensor([1.0, 1.0, 1.0]),
        radius=2.0,
    )
    sphere = config.setup()

    coords = torch.tensor(
        [
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [1.0, 1.5, 1.5],
        ],
    )

    sd_computed = sphere.evaluate_sdf(coords)
    sd_gt = torch.tensor(
        [
            [math.sqrt(0.75) - 2.0],
            [math.sqrt(3.0) - 2.0],
            [math.sqrt(0.5) - 2.0],
        ],
    )

    # test values
    assert torch.allclose(sd_computed, sd_gt)

def test_sdf_outer_1(out_dir: Path):
    """
    Test SDF evaluated outside a sphere.

    The sphere is placed at the origin and has radius 1.        
    """

    config = SphereConfig()
    sphere = config.setup()

    sd_computed = sphere.evaluate_sdf(2.0 * torch.eye(3))
    sd_gt = torch.tensor([[1.0], [1.0], [1.0]])

    # test values
    assert torch.allclose(sd_computed, sd_gt)

def test_sdf_outer_2(out_dir: Path):
    """
    Test SDF evaluated outside a sphere.

    The sphere is placed at the point (1, 1, 1) and has radius 0.5.
    """

    config = SphereConfig(
        center=torch.tensor([1.0, 1.0, 1.0]),
        radius=0.5,
    )
    sphere = config.setup()

    coords = torch.tensor(
        [
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [2.0, 1.5, 1.5],
        ],
    )

    sd_computed = sphere.evaluate_sdf(coords)
    sd_gt = torch.tensor(
        [
            [math.sqrt(0.75) - 0.5],
            [math.sqrt(3.0) - 0.5],
            [math.sqrt(1.5) - 0.5],
        ],
    )

    # test values
    assert torch.allclose(sd_computed, sd_gt)

def test_sdf_gradient(out_dir: Path):
    """
    Test the differentiation of SDF defined by a sphere.

    The sphere is placed at the origin and has radius 2.0.
    """

    config = SphereConfig()
    sphere = config.setup()

    coords = torch.tensor(
        [
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [2.0, 1.5, 1.5],
        ],
    )
    coords_norm = torch.linalg.norm(coords, dim=1, keepdim=True)

    sd_gradient_computed = sphere.evaluate_sdf_gradient(coords)
    sd_gradient_gt = coords / coords_norm

    # test values
    assert torch.allclose(sd_gradient_computed, sd_gradient_gt)
