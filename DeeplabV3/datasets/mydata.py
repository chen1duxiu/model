import os
import random
import sys
import tarfile
import collections

import cv2
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


def mydata_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class Mydata(data.Dataset):

    cmap = mydata_cmap()

    def __init__(self,
                 root,
                 dataname,
                 image_set='train',
                 transform=None):
        self.dataname =dataname
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set

        mydata_root = os.path.join(self.root, dataname)

        if not os.path.isdir(mydata_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        image_dir = os.path.join(mydata_root, 'image')
        label_dir = os.path.join(mydata_root, 'label')

        # for img_file in os.listdir(image_dir):
        #     # 读取图片
        #     img_path = os.path.join(image_dir, img_file)
        #     img = Image.open(img_path)
        #     images.append(img)
        #
        #     # 读取标签
        #     label_file = img_file.split('.')[0] + '.png'
        #     label_path = os.path.join(label_dir, label_file)
        #     label = Image.open(label_path)
        #
        #     labels.append(label)

        file_name = []
        for filename in os.listdir(label_dir):
            if filename.endswith(".png"):
                file_name.append(os.path.splitext(filename)[0])
        # 设置随机数种子
        random.seed(1234)
        # 随机打乱数据集
        random.shuffle(file_name)
        # 划分数据集
        train_ratio = 0.7  # 训练集比例
        val_ratio = 0.2  # 验证集比例
        test_ratio = 0.1  # 测试集比例
        num_data = len(file_name)

        # 计算数据集划分的大小
        num_train = int(num_data * train_ratio)
        num_val = int(num_data * val_ratio)
        num_test = num_data - num_train - num_val

        # 划分数据集
        train_data = file_name[:num_train]
        val_data = file_name[num_train:num_train + num_val]
        test_data = file_name[num_train + num_val:]

        def save_imageset(imageset_dir, samples, type):
            if not os.path.exists(imageset_dir):
                os.makedirs(imageset_dir)
            with open(os.path.join(imageset_dir, type + ".txt"), "w") as f:
                for sample in samples:
                    f.write(sample + "\n")

        save_imageset(os.path.join(mydata_root, "ImageSets/Segmentation"), train_data, 'train')
        save_imageset(os.path.join(mydata_root, "ImageSets/Segmentation"), val_data, 'val')
        save_imageset(os.path.join(mydata_root, "ImageSets/Segmentation"), test_data, 'test')
        if image_set == 'train':
            split_f = os.path.join(mydata_root, 'ImageSets/Segmentation/train.txt')
        elif image_set == 'val':
            split_f = os.path.join(mydata_root, 'ImageSets/Segmentation/val.txt')
        else:
            split_f = os.path.join(mydata_root, 'ImageSets/Segmentation/test.txt')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.labels = [os.path.join(label_dir, x + ".png") for x in file_names]

        assert (len(self.images) == len(self.labels))

        # splits_dir = os.path.join(mydata_root, 'ImageSets/Segmentation')
        # split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        # self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        # split_f_train = []
        # split_f_val = []
        # split_f_test = []
        # sge_path = os.mkdir(os.path.join(mydata_root, 'Segmentation'))
        # for file in train_data:
        #     train_file = file
        #     split_f_train.append(train_file)
        # for file in val_data:
        #     val_file = os.path.basename(file)
        #     split_f_val.append(val_file)
        # for file in test_data:
        #     test_file = os.path.basename(file)
        #     split_f_test.append(test_file)
        #
        # with open(os.path.join(sge_path, "train.txt"), 'w') as f:
        #     for i in split_f_train:
        #         f.write(i)
        #
        #
        #
        # splits_dir = os.path.join(mydata_root, 'ImageSets/Segmentation')
        # split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        #
        # if not os.path.exists(split_f):
        #     raise ValueError(
        #         'Wrong image_set entered! Please use image_set="train" '
        #         'or image_set="trainval" or image_set="val"')
        #


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.labels[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

