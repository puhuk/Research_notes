from torch import nn

import torch.nn.functional as F
import torch

from utils_external.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from utils_external.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class UpBlock3D(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class SameBlock3D(nn.Module):
    """
    Simple block with group convolution.
    """

    def __init__(self, in_features, out_features, groups=None, kernel_size=3, padding=1):
        super(SameBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, dimension=2):
        super(Encoder, self).__init__()
        down_blocks = []

        if dimension==2:
            for i in range(num_blocks):
                down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                            min(max_features, block_expansion * (2 ** (i + 1))),
                                            kernel_size=3, padding=1))
            self.down_blocks = nn.ModuleList(down_blocks)

        elif dimension==3:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
            for i in range(num_blocks):
                down_blocks.append(DownBlock3D(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                            min(max_features, block_expansion * (2 ** (i + 1))),
                                            kernel_size=kernel_size, padding=padding))
            self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, out_features=10, num_blocks=3, max_features=256, dimension=2):
        super(Decoder, self).__init__()

        up_blocks = []

        if dimension==2:

            for i in range(num_blocks)[::-1]:
                in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
                out_filters = min(max_features, block_expansion * (2 ** i))
                up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

            # up_blocks.append(nn.Conv2d())
            self.up_blocks = nn.ModuleList(up_blocks)
            self.conv = nn.Conv2d(in_channels=block_expansion + in_features, out_channels=out_features, kernel_size=(3,3), padding=(1,1))

        elif dimension==3:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)

            up_blocks = []

            for i in range(num_blocks)[::-1]:
                up_blocks.append(UpBlock3D((1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (
                    2 ** (i + 1))),
                                        min(max_features, block_expansion * (2 ** i)),
                                        kernel_size=kernel_size, padding=padding))

            self.up_blocks = nn.ModuleList(up_blocks)
            self.conv = nn.Conv3d(in_channels=block_expansion + in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            # print("skip", out.shape, skip.shape, len(skip.T))
            # skip = skip.T[:len(skip.T)-1].T
            out = torch.cat([out, skip], dim=1)

        if self.conv is not None:
            return self.conv(out)
        return out


class Hourglass(nn.Module):
    def __init__(self, in_features=3, block_expansion = 32, dimension=2, out_features=10, max_features=1024):
        super(Hourglass, self).__init__()
        self.in_features = in_features
        self.dimension = dimension
        self.block_expansion = block_expansion
        self.out_features = out_features
        self.max_features = max_features
        self.encoder = Encoder(in_features=self.in_features, max_features=self.max_features, block_expansion=self.block_expansion,dimension=self.dimension)
        self.decoder = Decoder(in_features=self.in_features, max_features=self.max_features, block_expansion=self.block_expansion, dimension=self.dimension, out_features=self.out_features)

    def forward(self, x):
        return self.decoder(self.encoder(x))