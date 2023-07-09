"""
positional_encoder.py


"""

from dataclasses import dataclass, field
from typing import Type

from jaxtyping import Shaped, jaxtyped
import torch
from torch import Tensor
from typeguard import typechecked

from src.encoders.base_encoder import Encoder, EncoderConfig


@dataclass
class PositionalEncoderConfig(EncoderConfig):
    """The configuration of a positional encoder"""

    _target: Type = field(default_factory=lambda: PositionalEncoder)

    signal_dim: int = 3
    """Dimensionality of the signal to encode"""
    embed_level: int = 10
    """Number of frequency bands to use in the positional encoding"""
    include_input: bool = True
    """A flag that determines whether to include raw input in the encoding"""


class PositionalEncoder(Encoder):
    """The encoder class implementing positional encoding"""

    config: PositionalEncoderConfig

    def __init__(
        self,
        config: PositionalEncoderConfig,
    ) -> None:
        super().__init__(config)

        self.signal_dim = self.config.signal_dim
        self.embed_level = self.config.embed_level
        self.include_input = self.config.include_input
        self.encoding_dim = 2 * self.embed_level * self.signal_dim
        if self.include_input:
            self.encoding_dim += self.signal_dim

        # creating embedding function
        self.embed_fns = self._create_embedding_fn()

    @jaxtyped
    @typechecked
    def encode(
        self,
        signal: Shaped[Tensor, "*batch_size signal_dim"],
    ) -> Shaped[Tensor, "*batch_size encoding_dim"]:
        """
        Encodes the signal using the positional encoder.
        """
        return torch.cat([fn(signal) for fn in self.embed_fns], -1)

    def _create_embedding_fn(self):
        """
        Creates embedding function from given
            (1) number of frequency bands;
            (2) dimension of data being encoded;

        The positional encoding is defined as:
        f(p) = [
                sin(2^0 * pi * p), cos(2^0 * pi * p),
                                ...,
                sin(2^{L-1} * pi * p), cos(2^{L-1} * pi * p)
            ],
        and is computed for all components of the input vector.

        Thus, the form of resulting tensor is:
        f(pos) = [
                sin(2^0 * pi * x), sin(2^0 * pi * y), sin(2^0 * pi * z),
                cos(2^0 * pi * x), cos(2^0 * pi * y), cos(2^0 * pi * z),
                    ...
            ],
        where pos = (x, y, z).

        NOTE: Following the official implementation, this code implements
        a slightly different encoding scheme:
            (1) the constant 'pi' in sinusoidals is dropped;
            (2) the encoding includes the original value 'x' as well;
        For details, please refer to https://github.com/bmild/nerf/issues/12.
        """
        embed_fns = []

        max_freq_level = self.embed_level

        freq_bands = 2 ** torch.arange(0.0, max_freq_level, dtype=torch.float32)

        if self.include_input:
            embed_fns.append(lambda x: x)

        for freq in freq_bands:
            embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
            embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))

        return embed_fns
