"""
main.py

A script for running unit tests for superquadric primitive.
"""

from dataclasses import dataclass, field
from typing import Type

from tests.base_test import TestConfig, Test
from tests.primitives.superquadric.test_cases import (
    test_create_superquadric,
)


@dataclass
class SuperquadricTestConfig(TestConfig):
    """The configuration of a superquadric test"""

    _target: Type = field(default_factory=lambda: SuperquadricTest)


class SuperquadricTest(Test):
    """The superquadric test class"""

    config: SuperquadricTestConfig


def main():
    """The entry point of the script"""
    test_config = SuperquadricTestConfig(
        test_cases=[
            test_create_superquadric,
        ],
    )
    test = test_config.setup()
    test.run()


if __name__ == "__main__":
    main()
