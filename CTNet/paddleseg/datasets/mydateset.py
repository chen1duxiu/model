

import os

from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose



@manager.DATASETS.add_component
class MyDataset(Dataset):

    NUM_CLASSES = 2

    def __init__(self, transforms, dataset_root=None, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'trainval', 'trainaug', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'trainval', 'trainaug', 'val') in PascalVOC dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        image_set_dir = os.path.join(self.dataset_root, 'ImageSets',
                                     'Segmentation')
        if mode == 'train':
            file_path = os.path.join(image_set_dir, 'train.txt')
        elif mode == 'val':
            file_path = os.path.join(image_set_dir, 'val.txt')

        img_dir = os.path.join(self.dataset_root, 'JPEGImages')
        label_dir = os.path.join(self.dataset_root, 'SegmentationClass')

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                label_path = os.path.join(label_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, label_path])

