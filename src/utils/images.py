"""
images.py

A collection of utilities for image processing.
"""

from pathlib import Path

from jaxtyping import Int, jaxtyped
import imageio
import numpy as np
from numpy import ndarray
from typeguard import typechecked


@jaxtyped
@typechecked
def create_video_from_images(
    images: Int[ndarray, "num_image height width channels"],
    out_file: Path,
) -> None:
    """Creates a video from a sequence of images"""

    images = images.astype(np.uint8)

    writer = imageio.get_writer(
        str(out_file),
        format="FFMPEG",
        mode="I",
        fps=24,
        macro_block_size=1,
    )

    for image in images:
        writer.append_data(image)
    writer.close()
