# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math

import torch
from torch import nn

from kornia.filters import filter2d, filter3d
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d


def distance_transform(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35) -> torch.Tensor:
    r"""Approximates the Euclidean distance transform of images using cascaded convolution operations.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.

    Args:
        image: Image with shape :math:`(B,C,H,W)` or :math:`(B,C,D,H,W)`.
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    Returns:
        tensor with shape :math:`(B,C,H,W)` or :math:`(B,C,D,H,W)`.

    Example:
        >>> tensor = torch.zeros(1, 1, 5, 5)
        >>> tensor[:,:, 1, 2] = 1
        >>> dt = kornia.contrib.distance_transform(tensor)

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) not in [4, 5]:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW or BxCxDxHxW. Got: {image.shape}")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    is_3d = len(image.shape) == 5
    device = image.device
    dtype = image.dtype

    if is_3d:
        # 3D Kernel Construction
        # n_iters is set based on the largest dimension (Depth, Height, or Width)
        n_iters: int = math.ceil(max(image.shape[2], image.shape[3], image.shape[4]) / math.floor(kernel_size / 2))

        # Create 3D grid for kernel
        grid = create_meshgrid3d(
            kernel_size, kernel_size, kernel_size, normalized_coordinates=False, device=device, dtype=dtype
        )
        grid -= math.floor(kernel_size / 2)

        # Calculate 3D Euclidean distance kernel: sqrt(x^2 + y^2 + z^2)
        kernel = torch.sqrt(grid[0, :, :, :, 0] ** 2 + grid[0, :, :, :, 1] ** 2 + grid[0, :, :, :, 2] ** 2)
        kernel = torch.exp(kernel / -h).unsqueeze(0)  # (1, kD, kH, kW)

    else:
        # 2D Kernel Construction
        n_iters = math.ceil(max(image.shape[2], image.shape[3]) / math.floor(kernel_size / 2))

        grid = create_meshgrid(
            kernel_size, kernel_size, normalized_coordinates=False, device=device, dtype=dtype
        )
        grid -= math.floor(kernel_size / 2)

        # Calculate 2D Euclidean distance kernel: hypot(x, y)
        kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
        kernel = torch.exp(kernel / -h).unsqueeze(0)  # (1, kH, kW)

    out = torch.zeros_like(image)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        if is_3d:
            cdt = filter3d(boundary, kernel, border_type="replicate")
        else:
            cdt = filter2d(boundary, kernel, border_type="replicate")

        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = torch.where(cdt > 0, 1.0, 0.0)
        if mask.sum() == 0:
            break

        offset: int = i * (kernel_size // 2)
        out += (offset + cdt) * mask
        boundary = torch.where(mask == 1, signal_ones, boundary)

    return out


class DistanceTransform(nn.Module):
    r"""Module that approximates the Euclidean distance transform of images using convolutions.

    Args:
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    """

    def __init__(self, kernel_size: int = 3, h: float = 0.35) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.h = h

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # If images have multiple channels, view the channels in the batch dimension to match kernel shape.
        if image.shape[1] > 1:
            # Flatten channels into batch dimension, preserving spatial dims (H,W or D,H,W)
            # image.shape[2:] captures all spatial dimensions regardless of 2D or 3D
            image_in = image.view(-1, 1, *image.shape[2:])
        else:
            image_in = image

        return distance_transform(image_in, self.kernel_size, self.h).view_as(image)