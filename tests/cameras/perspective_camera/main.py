"""
main.py

A script for running unit tests for perspective camera.
"""

from dataclasses import dataclass, field
from typing import Type

from tests.base_test import TestConfig, Test
from tests.cameras.perspective_camera.test_cases import (
    test_create_camera,
)


@dataclass
class PerspectiveCameraTestConfig(TestConfig):
    """The configuration of a perspective camera test"""

    _target: Type = field(default_factory=lambda: PerspectiveCameraTest)


class PerspectiveCameraTest(Test):
    """The perspective camera test class"""

    config: PerspectiveCameraTestConfig


def main():
    """The entry point of the script"""
    test_config = PerspectiveCameraTestConfig(
        test_cases=[
            test_create_camera,
        ],
    )
    test = test_config.setup()
    test.run()


if __name__ == "__main__":
    main()
