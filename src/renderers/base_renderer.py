"""
base_renderer.py

The base class for SDF renderers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Type

from jaxtyping import Shaped
from torch import Tensor

from src.cameras.base_camera import Camera
from src.configs.base_config import InstantiateConfig


@dataclass
class RendererConfig(InstantiateConfig):
    """The configuration of a SDF renderer"""

    _target: Type = field(default_factory=lambda: Renderer)


class Renderer(ABC):
    """The base class of SDF renderers"""

    config: RendererConfig

    def __init__(
        self,
        config: RendererConfig,
    ) -> None:
        self.config = config

    @abstractmethod
    def render(
        self,
        scene: Type,
        camera: Camera,
        **kwargs: Dict[str, Type],
    ) -> Shaped[Tensor, "num_ray 3"]:
        """Renders the scene using the renderer."""
