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
################################################
# Every dataset should be parsed in png format #
# Vox                                          #
#   - video_name_1                             #
#           - 1.png                            #
#           - 2.png                            #
#           - 3.png                            #
#                                              #
# For test:                                    #
#   ../../../../../../../dataset/vox/          #
################################################

class video_dataset(Dataset):

    def __init__(self, root_dir, frames_per_video=1, consequent=False, is_train=True, image_shape =(256, 256)):
        self.root_dir = root_dir
        self.videos = os.listdir(self.root_dir)
        self.frames_per_video = frames_per_video
        self.consequent = consequent
        self.image_shape = (256, 256)

        if is_train:
            test_videos = self.videos
        else:
            train_videos, test_videos = train_test_split(self.videos, test_size=0.2)



    def __len__(self):
        print(self.videos[0])
        return len(self.videos)

    def __getitem__(self, idx):
        out = {}
        images = glob.glob(os.path.join(self.root_dir, self.videos[idx])+'/*.png')
        print(images)
        num_frames = len(images)

        print(num_frames, self.frames_per_video)

        source_idx, target_idx = np.sort(np.random.choice(num_frames-self.frames_per_video, replace=False, size=2))
        source_video, target_video = [], []
        for i in range(self.frames_per_video):
            source_video.append(img_as_float32(cv2.resize(io.imread(images[source_idx+i]), self.image_shape)))
            target_video.append(img_as_float32(cv2.resize(io.imread(images[target_idx+i]), self.image_shape)))

        source_video = np.array(source_video)
        target_video = np.array(target_video)
        print(source_video.shape, target_video.shape)

        out['source'] = source_video.transpose((3, 0, 1, 2))
        out['target'] = target_video.transpose((3, 0, 1, 2))
        out['name'] = self.videos[idx]

        return out
            


if __name__ == "__main__":
    # dataset = video_dataset(is_train=True, image_shape=(256, 256, 3), root_dir='../../../../../../../dataset/vox-video-png/')
    dataset = video_dataset(root_dir='../../../../../../../dataset/vox-video-png/')
    print(dataset.__len__())
    print(dataset.__getitem__(0)['source'].shape, dataset.__getitem__(0)['target'].shape)

    # dataset = FramesDataset(root_dir='../../../../../../../dataset/vox-png/')
    # print(dataset.__getitem__(0)['source'].shape, dataset.__getitem__(0)['video'].shape)

    
