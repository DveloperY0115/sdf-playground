"""
main.py

A script for running unit tests for sphere primitive.
"""

from dataclasses import dataclass, field
from typing import Type

from tests.base_test import TestConfig, Test
from tests.primitives.sphere.test_cases import (
    test_create_sphere,
    test_sdf_inner_1,
    test_sdf_inner_2,
    test_sdf_outer_1,
    test_sdf_outer_2,
    test_sdf_gradient,
)


@dataclass
class SphereTestConfig(TestConfig):
    """The configuration of a sphere test"""

    _target: Type = field(default_factory=lambda: SphereTest)


class SphereTest(Test):
    """The sphere test class"""

    config: SphereTestConfig


def main():
    """The entry point of the script"""
    test_config = SphereTestConfig(
        test_cases=[
            test_create_sphere,
            test_sdf_inner_1,
            test_sdf_inner_2,
            test_sdf_outer_1,
            test_sdf_outer_2,
            test_sdf_gradient,
        ],
    )
    test = test_config.setup()
    test.run()


if __name__ == "__main__":
    main()
