"""From-scratch residual CNN and shared full-image preprocessing."""

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class ResidualMooncakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 192, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2),
        )
        self.head = nn.Linear(192, 1)

    def forward(self, x):
        return self.head(self.features(x))


def preprocess_image(path, size=192):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        return torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1).float() / 255
