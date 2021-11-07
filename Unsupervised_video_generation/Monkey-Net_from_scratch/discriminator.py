from tqdm import trange

import torch
from torch.utils.data import DataLoader

from logger import Logger
from losses import generator_loss, discriminator_loss, generator_loss_names, discriminator_loss_names

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from PIL import ImageFile

class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

    def forward(self, x, kp_joined, generated):
        kp_dict = split_kp(kp_joined, self.train_params['detach_kp_discriminator'])
        discriminator_maps_generated = self.discriminator(generated['video_prediction'].detach(), **kp_dict)
        discriminator_maps_real = self.discriminator(x['video'], **kp_dict)
        loss = discriminator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                  discriminator_maps_real=discriminator_maps_real,
                                  loss_weights=self.train_params['loss_weights'])
        return loss