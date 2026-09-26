"""Inference-only model definition for the included one-class checkpoint."""

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class OneClassCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.3)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, image):
        features = self.blocks(image)
        return self.decoder(features), self.embedding(features)


def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def anomaly_score(model, image, center):
    reconstruction, embedding = model(image)
    reconstruction_error = ((reconstruction - image) ** 2).mean((1, 2, 3))
    feature_distance = .02 * ((embedding - center) ** 2).mean(1)
    return reconstruction_error + feature_distance
