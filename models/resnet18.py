import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18_CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 / CIFAR-100.
    - No pretrained weights
    - Modified first conv layer
    - No maxpool
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Load standard ResNet-18 architecture (no pretrained weights)
        self.model = resnet18(weights=None)

        # Modify first convolution for CIFAR (32x32 images)
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove maxpool layer (too aggressive for small images)
        self.model.maxpool = nn.Identity()

        # Modify final fully connected layer
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
