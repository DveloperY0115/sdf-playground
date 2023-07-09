"""
train_shapenet.py

A script for a toy example that fits neural implicit surfaces to ShapeNet renderings.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import imageio
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import tyro


from src.cameras.perspective_camera import PerspectiveCamera
from encoders.positional_encoder import PositionalEncoderConfig
from src.fields.primitive_field import PrimitiveFieldConfig
from src.networks.mlp import MLPConfig
from src.networks.radiance_network import RadianceNetworkConfig
from src.primitives.sphere import SphereConfig
from src.renderers.ray_samplers.stratified_sampler import UniformSamplerConfig
from src.renderers.volsdf_renderer import VolSDFRendererConfig


@dataclass
class TrainConfig:
    """The configuration of a training process"""

    output_root: Path = Path("output/train_shapenet")
    """The root directory of the outputs"""
    data_dir: Path = Path("/home/dreamy1534/Projects/cvpr2024/code/shapenet_renderer/outputs/model_normalized/2023-07-08/18-30-37")
    """The directory where the training data are located"""
    num_iter: int = 1000000
    """Number of training iterations"""
    val_frequency: int = 1000
    """Number of iterations between validation"""

    ray_batch_size: int = 4096
    """Number of rays included in a single ray batch during training"""

def load_data(data_dir: Path) -> None:
    """
    Loads data for training.
    """

    assert data_dir.exists()

    image_dir = data_dir / "image"
    assert image_dir.exists()
    image_files = sorted(image_dir.glob("*.png"))
    images = []

    for file in image_files:
        image = np.asarray(imageio.imread(str(file)))
        images.append(image)
    images = np.stack(images, axis=0)
    images = images / 255.0
    images = images.astype(float)

    # load camera parameters
    with open(data_dir / "camera_parameters.json", "r") as f:
        camera_params = json.load(f)

    # load bounding box parameters
    bbox_params = np.load(data_dir / "bbox_params.npy")

    # assertions
    assert len(images) == len(camera_params), (
        f"{len(images)} != {len(camera_params)}"
    )

    # train / val split
    ####
    # val_indices = [5, 10, 15]  # TODO: Replace hard-coded sample indices

    # train_images = images[[i for i in range(len(images)) if i not in val_indices]]
    # val_images = images[val_indices]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    train_images = images
    val_images = images

    # train_camera_params = [camera_params[i] for i in range(len(camera_params)) if i not in val_indices]
    # val_camera_params = [camera_params[i] for i in val_indices]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    train_camera_params = camera_params
    val_camera_params = camera_params
    ####

    return (
        train_images,
        val_images,
        train_camera_params,
        val_camera_params,
        bbox_params,
    )

def train_one_iter(
    iter_index,
    field,
    renderer,
    optimizer,
    images,
    camera_params,
    num_pixel,
) -> float:
    """Routine for one training iteration"""

    # sample random training data
    ####
    # sample_index = np.random.randint(0, len(images))
    sample_index = 0
    ####

    image = torch.tensor(images[sample_index], device="cuda")
    camera_to_world = torch.tensor(
        camera_params[sample_index]["extrinsic"],
        device="cuda",
    )
    intrinsics = camera_params[sample_index]["intrinsic"]
    focal_x = intrinsics["focal_x"]
    focal_y = intrinsics["focal_y"]
    center_x = intrinsics["center_x"]
    center_y = intrinsics["center_y"]
    near = intrinsics["near"]
    far = intrinsics["far"]

    # compute sample points on rays
    camera = PerspectiveCamera(
        camera_to_world=camera_to_world,
        focal_x=focal_x,
        focal_y=focal_y,
        center_x=center_x,
        center_y=center_y,
        near=near,
        far=far,
        image_height=int(center_y * 2.0),
        image_width=int(center_x * 2.0),
        device="cuda",
    )

    # randomly sample pixels from which rays will be casted
    pixel_indices = torch.tensor(
        np.random.choice(
            camera.image_height.item() * camera.image_width.item(),
            size=[num_pixel],
            replace=False,
        ),
        device=camera.device,
    )

    if iter_index < 500:  # warm-up during early stage of training

        # sample pixels around the center
        center_i = (camera.image_height.item() - 1) // 2
        center_j = (camera.image_width.item() - 1) // 2

        center_is = torch.arange(
            center_i - center_i // 2,
            center_i + center_i // 2,
            device=camera.device,
        )
        center_js = torch.arange(
            center_j - center_j // 2,
            center_j + center_j // 2,
            device=camera.device,
        )

        center_indices = torch.cartesian_prod(center_is, center_js)
        center_indices = center_indices[:, 0] * camera.image_width + center_indices[:, 1]
        pixel_indices = center_indices[
            torch.randperm(len(center_indices))[: num_pixel]
        ]

    # evaluate field
    image_pred, depth_pred, normal_pred = renderer.render(
        field,
        camera,
        pixel_indices,
    )

    # reset gradients
    optimizer.zero_grad()

    # evaluate loss
    loss = torch.sum(
        (image.reshape(-1, 3)[pixel_indices, ...] - image_pred) ** 2.0
    )

    # back propagation
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def validate(
    field,
    renderer,
    images,
    camera_params,
):
    """Routine for validation"""

    gt_rgb_images = []
    rgb_images = []
    depth_maps = []
    normal_maps = []
    val_loss = 0.0

    for index, image in enumerate(images):

        image = torch.tensor(image, device="cuda")
        camera_to_world = torch.tensor(
            camera_params[index]["extrinsic"],
            device="cuda",
        )
        intrinsics = camera_params[index]["intrinsic"]
        focal_x = intrinsics["focal_x"]
        focal_y = intrinsics["focal_y"]
        center_x = intrinsics["center_x"]
        center_y = intrinsics["center_y"]
        near = intrinsics["near"]
        far = intrinsics["far"]

        # compute sample points on rays
        camera = PerspectiveCamera(
            camera_to_world=camera_to_world,
            focal_x=focal_x,
            focal_y=focal_y,
            center_x=center_x,
            center_y=center_y,
            near=near,
            far=far,
            image_height=int(center_y * 2.0),
            image_width=int(center_x * 2.0),
            device="cuda",
        )

        pixel_indices = torch.arange(
            0, camera.image_height * camera.image_width, device=camera.device,
        )

        image_pred, depth_pred, normal_pred = renderer.render(
            field,
            camera,
            pixel_indices,
        )

        loss = torch.sum(
            (image.reshape(-1, 3)[pixel_indices, ...] - image_pred) ** 2.0
        )

        # collect outputs
        val_loss += loss.item()

        gt_rgb_image = torch.clamp(image, 0.0, 1.0)
        gt_rgb_image = (gt_rgb_image * 255.0).type(torch.uint8)

        rgb_image = torch.clamp(image_pred, 0.0, 1.0)
        rgb_image = (rgb_image * 255.0).type(torch.uint8)

        depth_min, depth_max = depth_pred.min(), depth_pred.max()
        depth_map = (depth_pred - depth_min) / (depth_max - depth_min) + 1e-6
        depth_map = torch.clamp(depth_map, 0.0, 1.0)
        depth_map = (depth_map * 255.0).type(torch.uint8)

        normal_map = (normal_pred + 1.0) * 0.5
        normal_map = (normal_map * 255.0).type(torch.uint8)

        gt_rgb_images.append(gt_rgb_image.reshape(camera.image_height, camera.image_width, 3))
        rgb_images.append(rgb_image.reshape(camera.image_height, camera.image_width, 3))
        depth_maps.append(depth_map.reshape(camera.image_height, camera.image_width))
        normal_maps.append(normal_map.reshape(camera.image_height, camera.image_width, 3))

    val_loss /= len(images)
    gt_rgb_images = torch.stack(gt_rgb_images).detach().cpu()
    rgb_images = torch.stack(rgb_images).detach().cpu()
    depth_maps = torch.stack(depth_maps).detach().cpu()
    normal_maps = torch.stack(normal_maps).detach().cpu()

    return val_loss, gt_rgb_images, rgb_images, depth_maps, normal_maps


def main(config: TrainConfig) -> None:
    """The main function"""

    # ==========================================================================================
    # create experiment directory
    output_root = config.output_root
    experiment_dir = output_root / time.strftime("%Y-%m-%d/%H-%M-%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    # ==========================================================================================

    # ==========================================================================================
    # prepare train / val dataset
    (
        train_images,
        val_images,
        train_camera_params,
        val_camera_params,
        bbox_params,  # TODO: Use this when defining superquadrics
    ) = load_data(config.data_dir)
    # ==========================================================================================

    # ==========================================================================================
    # initialize model
    field = PrimitiveFieldConfig(
        primitive_config=SphereConfig(),
        radiance_network_config=RadianceNetworkConfig(
            coord_dim=63,
            view_dim=27,
            hidden_dim=256,
            num_hidden_layers=3,
            actvn_func=nn.ReLU(),
            out_actvn_func=nn.Sigmoid(),
        ),
        sdf_network_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dim=256,
            num_hidden_layers=3,
            actvn_func=nn.ReLU(),
            out_actvn_func=nn.ReLU(),
        ),
        coord_encoder_config=PositionalEncoderConfig(),
        view_direction_encoder_config=PositionalEncoderConfig(
            signal_dim=3,
            embed_level=4,
            include_input=True,
        ),
    ####
    # ).setup()
    ).setup().cuda()
    ####
    # ==========================================================================================

    # ==========================================================================================
    # initialize optimizer
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-4)
    # ==========================================================================================

    # ==========================================================================================
    # initialize renderer
    renderer_config = VolSDFRendererConfig(
        ray_sampler_config=UniformSamplerConfig(
            num_sample=128,
        ),
    )
    renderer = renderer_config.setup()

    # ==========================================================================================

    # ==========================================================================================
    # training loop
    for iter_index in tqdm(range(config.num_iter)):

        # train
        train_loss = train_one_iter(
            iter_index,
            field,
            renderer,
            optimizer,
            train_images,
            train_camera_params,
            config.ray_batch_size,
        )
        print(f"Train Loss [{iter_index} / {config.num_iter}]: {train_loss}")

        # validate
        if (iter_index + 1) % config.val_frequency == 0:
            val_loss, gt_rgb_images, rgb_images, depth_maps, normal_maps = validate(
                field,
                renderer,
                val_images,
                val_camera_params,
            )
            print(f"Validation Loss [{iter_index} / {config.num_iter}]: {val_loss}")

            # log outputs if necessary
            rgb_dir = experiment_dir / "rgb" / f"{iter_index:04d}"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir = experiment_dir / "depth" / f"{iter_index:04d}"
            depth_dir.mkdir(parents=True, exist_ok=True)
            normal_dir = experiment_dir / "normal" / f"{iter_index:04d}"
            normal_dir.mkdir(parents=True, exist_ok=True)

            for index, (gt_rgb_image, rgb_image, depth_map, normal_map) in enumerate(
                zip(gt_rgb_images, rgb_images, depth_maps, normal_maps)
            ):
                imageio.imwrite(
                    str(rgb_dir / f"{index:04d}.png"),
                    np.concatenate([rgb_image.numpy(), gt_rgb_image.numpy()], axis=1),
                )
                imageio.imwrite(
                    str(depth_dir / f"{index:04d}.png"),
                    depth_map.numpy(),
                )
                imageio.imwrite(
                    str(normal_dir / f"{index:04d}.png"),
                    normal_map.numpy(),
                )
    # ==========================================================================================


if __name__ == "__main__":
    main(
        tyro.cli(TrainConfig)
    )
