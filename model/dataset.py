import os
import sys
import re
import six
import math
import lmdb
import torch
from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
try:
    from torch._utils import _accumulate
except ImportError:
    from itertools import accumulate as _accumulate
import torchvision.transforms as transforms


class LmdbDataset(Dataset):

    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))

        self.filtered_index_list = list(range(1, self.nSamples + 1))

    def __len__(self):
        return len(self.filtered_index_list)

    def __getitem__(self, index):
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label = txn.get(f'label-{index:09d}'.encode()).decode('utf-8')
            imgbuf = txn.get(f'image-{index:09d}'.encode())

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('L')

        label = label.lower()
        out_of_char = f'[^{re.escape(self.opt.character)}]'
        label = re.sub(out_of_char, '', label)
        label = re.sub(r'\s+', ' ', label).strip()

        return img, label


class ResizeNormalize(object):
    def __init__(self, size):
        self.size = size
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, Image.BICUBIC)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class AlignCollate(object):
    def __init__(self, imgH=32, imgW=256):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        images, labels = zip(*batch)
        transform = ResizeNormalize((self.imgW, self.imgH))
        images = [transform(image) for image in images]
        images = torch.stack(images, 0)
        return images, labels
