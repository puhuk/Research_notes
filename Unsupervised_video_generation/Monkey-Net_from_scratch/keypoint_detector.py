from torch import nn
import torch
import torch.nn.functional as F
# from modules.util import Hourglass, make_coordinate_grid, matrix_inverse, smallest_singular
from utils import Encoder, Decoder

def gaussian2kp(heatmap):
    """
    Extract the mean and from a heatmap
    """
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
    value = (heatmap * grid).sum(dim=(2, 3))
    kp = {'value': value}

    return kp

class Hourglass(nn.Module):
    def __init__(self, dimension=2, num_kp=10):
        super(Hourglass, self).__init__()

        self.encoder = Encoder(in_features=3, max_features=1024, block_expansion=32,dimension=2)
        self.decoder = Decoder(in_features=3, max_features=1024, block_expansion=32, dimension=2, out_features=num_kp)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and variance.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature,
                 kp_variance, scale_factor=1, clip_variance=None, dimension=2):
        super(KPDetector, self).__init__()
        self.dimension = dimension
        self.num_kp = num_kp
        self.predictor = Hourglass(dimension=self.dimension, num_kp=self.num_kp)
        self.temperature = temperature
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor
        self.clip_variance = clip_variance
        self.dimension = dimension



    def forward(self, x):
        # if self.scale_factor != 1:
        #     x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))

        heatmap = self.predictor(x)
        final_shape = heatmap.shape

        if self.dimension==2:
            heatmap = heatmap.view(final_shape[0], final_shape[1], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=2)
            heatmap = heatmap.view(*final_shape)

            out = gaussian2kp(heatmap)

        elif self.dimension==3:
            heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=3)
            heatmap = heatmap.view(*final_shape)

            out = gaussian2kp(heatmap, self.kp_variance, self.clip_variance)

        return out
