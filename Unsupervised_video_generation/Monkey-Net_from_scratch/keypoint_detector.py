from torch import nn
import torch
import torch.nn.functional as F
# from modules.util import Hourglass, make_coordinate_grid, matrix_inverse, smallest_singular
from utils import Encoder, Decoder
from utils import Hourglass

def kp2gaussian_2d(kp, spatial_size, kp_variance=0.001):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def kp2gaussian_3d(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())

    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape

    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    if kp_variance == 'matrix':
        var = kp['var']
        inv_var = matrix_inverse(var)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)
        out = torch.exp(-0.5 * under_exp)
    elif kp_variance == 'single':
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp['var'])
    else:
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def gaussian2kp_2d(heatmap):
    """
    Extract the mean and from a heatmap
    """
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
    value = (heatmap * grid).sum(dim=(2, 3))
    kp = {'value': value}

    return kp

def gaussian2kp_3d(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape
    #adding small eps to avoid 'nan' in variance
    heatmap = heatmap.unsqueeze(-1) + 1e-7
    grid = make_coordinate_grid(shape[3:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)

    mean = (heatmap * grid).sum(dim=(3, 4))

    kp = {'mean': mean.permute(0, 2, 1, 3)}

    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        var = var * heatmap.unsqueeze(-1)
        var = var.sum(dim=(3, 4))
        var = var.permute(0, 2, 1, 3, 4)
        if clip_variance:
            min_norm = torch.tensor(clip_variance).type(var.type())
            sg = smallest_singular(var).unsqueeze(-1)
            var = torch.max(min_norm, sg) * var / sg
        kp['var'] = var

    elif kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=(3, 4))
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var

    return kp

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and variance.
    """

    def __init__(self, block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5, temperature=0.1,
                 kp_variance='matrix', scale_factor=1, clip_variance=None, dimension=2):
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
        out= {}

        if self.dimension==2:
            heatmap = heatmap.view(final_shape[0], final_shape[1], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=2)
            heatmap = heatmap.view(*final_shape)

            # out['heatmap'] = heatmap

            out = gaussian2kp_2d(heatmap)

        elif self.dimension==3:
            heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=3)
            heatmap = heatmap.view(*final_shape)

            out = gaussian2kp_3d(heatmap, self.kp_variance, self.clip_variance)

            # out['heatmap'] = heatmap

        return out
