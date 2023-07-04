"""
main.py

A script for running unit tests for VolSDFRenderer.
"""

from dataclasses import dataclass, field
from typing import Type

from tests.base_test import TestConfig, Test
from tests.renderers.volsdf_renderer.test_cases import (
    test_create_renderer_stratified_sampler,
    test_quadrature_evaluation,
    test_render_sphere,
    test_render_superquadric1,
    test_render_superquadric2,
    test_render_superquadric3,
)


@dataclass
class VolSDFRendererTestConfig(TestConfig):
    """The configuration of a VolSDFRenderer test"""

    _target: Type = field(default_factory=lambda: VolSDFRendererTest)
    test_name: str = "volsdf_renderer"
    """The name of the test"""


class VolSDFRendererTest(Test):
    """The VolSDFRenderer test class"""

    config: VolSDFRendererTestConfig


def main():
    """The entry point of the script"""
    test_config = VolSDFRendererTestConfig(
        test_cases=[
            test_create_renderer_stratified_sampler,
            test_quadrature_evaluation,
            test_render_sphere,
            test_render_superquadric1,
            test_render_superquadric2,
            test_render_superquadric3,
        ]
    )
    test = test_config.setup()
    test.run()


if __name__ == "__main__":
    main()
