"""
test_cases.py

A set of test cases for superquadric primitive.
"""

import math
from pathlib import Path

import torch

from src.primitives.superquadric import SuperquadricConfig


def test_create_superquadric(out_dir: Path):
    """Creates a superquadric and checks its default parameters"""

    config = SuperquadricConfig()
    superquadric = config.setup()

    # test values
    assert torch.allclose(superquadric.center, torch.zeros(3))
    assert torch.allclose(superquadric.orientation, torch.eye(3))
    assert torch.allclose(superquadric.epsilons, torch.ones(2))
    assert torch.allclose(superquadric.scales, torch.ones(3))
