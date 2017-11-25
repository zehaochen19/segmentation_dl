import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

import cfg
import transform
import numpy as np

from matplotlib import pyplot as plt

to_pil = transforms.ToPILImage()


def ade_lbl_trans(label):
    label = torch.from_numpy(label).long()
    return label


class ADEDataset(Dataset):
    def __init__(self, root, split, transform):
        self.root = root
        self.split = split
        self.transform = transform

        self.ids = []

        self._img = os.path.join(cfg.ade_root, 'images', '{}.jpg')
        self._lbl = os.path.join(cfg.ade_root, 'annotations_sceneparsing',
                                 '{}.png')

        anno_path = os.path.join(cfg.ade_root, 'images', split + '.txt')
        for line in open(anno_path):
            self.ids.append(os.path.join(line.strip()[:-4]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, lbl = self._img.format(self.ids[idx]), self._lbl.format(
            self.ids[idx])
        img, lbl = Image.open(img), Image.open(lbl)

        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.imshow(img)
        fig.add_subplot(2, 2, 2)
        plt.imshow(lbl)

        img, lbl = self.transform(img, lbl)

        img_, label_ = to_pil(img), Image.fromarray(lbl.numpy().astype(
            np.uint8))
        fig.add_subplot(2, 2, 3)
        plt.imshow(img_)
        fig.add_subplot(2, 2, 4)
        plt.imshow(label_)
        plt.show()

        return img, lbl


def ade_test():
    dataset = ADEDataset(cfg.ade_root, 'validation', transform.augmentation)
    dataloader = DataLoader(dataset, 4, True)
    for data in dataloader:
        print(data)
        break


if __name__ == '__main__':
    ade_test()
