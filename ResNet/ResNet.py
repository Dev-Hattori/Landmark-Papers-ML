###################################
"""
author: Devansh
"""
###################################

from typing import List, Optional
import torch
from torch import nn


class LinearProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # Convolution layer for linear projection W_s.x
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride)
        # New dimension = (m-k)/s +1 = (m-1)/s +1
        # Paper suggests adding batch normalization after each convolution operation
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        # New Dim = (m+2-3)/s +1 = (m-1)/s +1 = m'
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()
        # same effect (on dimensions) as k=1 and p=0, ie Linear projection
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # New Dim = (m' +2 -3)/1 +1 = m' = (m-1)/s +1
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Check for if the dimensions are matching else need to project
        if stride != 1 or in_channels != out_channels:
            self.shortcut = LinearProjection(in_channels, out_channels, stride)
            # (m -1)/s +1 = m'
        else:
            self.shortcut = nn.Identity()

        self.activation2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        skip_con = self.shortcut(x)

        x = self.activation1(self.bn1(self.conv1(x)))

        x = self.bn2(self.conv2(x)) + skip_con

        return self.activation2(x)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, stride=1)
        # m' = m
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=bottleneck_channels,
                               out_channels=bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        # m'' = (m-1)/s +1
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels,
                               out_channels=out_channels, kernel_size=1, stride=1)
        # m' = m
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip = LinearProjection(
                in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            self.skip = nn.Identity()

        self.activation3 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        skip = self.skip(x)

        x = self.activation1(self.bn1(self.conv1(x)))

        x = self.activation2(self.bn2(self.conv2(x)))

        x = self.activation3(self.bn3(self.conv3(x)) + skip)

        return x


class ResNet(nn.Module):

    def __init__(self, n_blocks: List[int] = [3, 3],
                 n_channels:  List[int] = [16, 32],
                 # It is optional to have bottleneck blocks
                 bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3,  # By default 3
                 first_kernel_size: int = 7  # As used in the paper
                 ):
        super().__init__()

        # Number of blocks and number of channels for each feature map size
        assert len(n_blocks) == len(n_channels)

        # If bottleneck residual blocks are used, the number of channels in bottlenecks should be provided for each feature map size
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        self.conv1st = nn.Conv2d(
            img_channels, n_channels[0], kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])

        blocks = []

        prev_channels = n_channels[0]

        for i, channels in enumerate(n_channels):
            # The first block for the new feature map size, will have a stride length of 2 except for the very first block
            stride = 1 if len(blocks) == 0 else 2

            if bottlenecks is None:
                blocks.append(ResidualBlock(
                    in_channels=prev_channels, out_channels=channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(in_channels=prev_channels,
                              bottleneck_channels=bottlenecks[i], out_channels=channels, stride=stride))

            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(
                        in_channels=channels, out_channels=channels, stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(
                        in_channels=channels, bottleneck_channels=bottlenecks[i], out_channels=channels, stride=1))

            prev_channels = channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):

        x = self.bn(self.conv1st(x))

        x = self.blocks(x)

        return x


class classifier(nn.Module):
    def __init__(self, base: nn.Module, input_shape: tuple, n_classes: int):
        super().__init__()

        self.base = base
        # Figuring out the size of the outputs
        input = torch.randn(input_shape)
        base_out = base(input.unsqueeze(0))
        in_linear = nn.Flatten()(base_out)
        in_linear = in_linear.shape[1]

        self.bn = nn.BatchNorm2d(base_out.shape[1])
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_linear,
                             out_features=n_classes)
        self.drop = nn.Dropout()
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.base(x)
        x = self.bn(x)
        x = self.flat(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc1(x)

        return x
