from torch import nn

import torch.nn.functional as F
import torch

from utils_external.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

import cv2
# from utils_external.augmentation import AllAugmentationTransform, VideoToTensor

import glob

from utils import UpBlock2d, DownBlock2d, UpBlock3D, DownBlock3D, Encoder

source_img = torch.tensor(cv2.resize(cv2.imread('../../../../../../../dataset/celebA/img_align_celeba/000020.jpg'),(256,256))).float()
source_img = source_img.permute(2,0,1)
source_img = torch.unsqueeze(source_img, 0)
