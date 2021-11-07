from tqdm import trange

import torch
from torch.utils.data import DataLoader

# from logger import Logger

from torch.optim.lr_scheduler import MultiStepLR

from utils_external.sync_batchnorm import DataParallelWithCallback

from utils import Encoder, Decoder
from movement import MovementEmbeddingModule
from dense_motion import DenseMotionModule
from PIL import ImageFile
from torch import nn

class MotionTransferGenerator(nn.Module):
    """
    Motion transfer generator. That Given a keypoints and an appearance trying to reconstruct the target frame.
    Produce 2 versions of target frame, one warped with predicted optical flow and other refined.
    """

    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks, num_refinement_blocks,
                 dense_motion_params=None, kp_embedding_params=None, interpolation_mode='nearest'):
        super(MotionTransferGenerator, self).__init__()

        self.appearance_encoder = Encoder(block_expansion, in_features=num_channels, max_features=max_features,
                                          num_blocks=num_blocks)

        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                               num_channels=num_channels, **kp_embedding_params)
            embedding_features = self.kp_embedding_module.out_channels
        else:
            self.kp_embedding_module = None
            embedding_features = 0

        if dense_motion_params is not None:
            self.dense_motion_module = DenseMotionModule(num_kp=num_kp, kp_variance=kp_variance,
                                                         num_channels=num_channels,
                                                         **dense_motion_params)
        else:
            self.dense_motion_module = IdentityDeformation()

        self.video_decoder = Decoder(block_expansion=block_expansion, in_features=num_channels,
                                     out_features=num_channels, max_features=max_features, num_blocks=num_blocks,
                                     additional_features_for_block=embedding_features,
                                     use_last_conv=False)

        self.refinement_module = torch.nn.Sequential()
        in_features = block_expansion + num_channels + embedding_features
        for i in range(num_refinement_blocks):
            self.refinement_module.add_module('r' + str(i),
                                              ResBlock3D(in_features, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        self.refinement_module.add_module('conv-last', nn.Conv3d(in_features, num_channels, kernel_size=1, padding=0))
        self.interpolation_mode = interpolation_mode

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w), mode=self.interpolation_mode)
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def forward(self, source_image, kp_driving, kp_source):
        appearance_skips = self.appearance_encoder(source_image)

        deformations_absolute = self.dense_motion_module(source_image=source_image, kp_driving=kp_driving,
                                                         kp_source=kp_source)

        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.kp_embedding_module is not None:
            d = kp_driving['mean'].shape[1]
            movement_embedding = self.kp_embedding_module(source_image=source_image, kp_driving=kp_driving,
                                                          kp_source=kp_source)
            kp_skips = [F.interpolate(movement_embedding, size=(d,) + skip.shape[3:], mode=self.interpolation_mode) for skip in appearance_skips]
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        video_deformed = self.deform_input(source_image, deformations_absolute)
        video_prediction = self.video_decoder(skips)
        video_prediction = self.refinement_module(video_prediction)
        video_prediction = torch.sigmoid(video_prediction)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

    def forward(self, x):
        kp_joined = self.kp_extractor(torch.cat([x['source'], x['video']], dim=2))
        generated = self.generator(x['source'],
                                   **split_kp(kp_joined, self.train_params['detach_kp_generator']))
        video_prediction = generated['video_prediction']
        video_deformed = generated['video_deformed']

        kp_dict = split_kp(kp_joined, False)
        discriminator_maps_generated = self.discriminator(video_prediction, **kp_dict)
        discriminator_maps_real = self.discriminator(x['video'], **kp_dict)
        generated.update(kp_dict)

        losses = generator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                discriminator_maps_real=discriminator_maps_real,
                                video_deformed=video_deformed,
                                loss_weights=self.train_params['loss_weights'])

        return tuple(losses) + (generated, kp_joined)