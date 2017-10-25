import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import cfg
import random
import numpy as np

from matplotlib import pyplot as plt

img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])


def ade_lbl_trans(label):
    label = torch.from_numpy(label).long()
    return label


class ADEDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split

        self.ids = []

        self._img = os.path.join(cfg.ade_root, 'images', '{}.jpg')
        self._lbl = os.path.join(cfg.ade_root, 'annotations_sceneparsing', '{}.png')

        anno_path = os.path.join(cfg.ade_root, 'images', split + '.txt')
        for line in open(anno_path):
            self.ids.append(os.path.join(line.strip()[:-4]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, lbl = self._img.format(self.ids[idx]), self._lbl.format(self.ids[idx])
        img, lbl = Image.open(img), Image.open(lbl)
        img, lbl = img.resize((cfg.size, cfg.size)), lbl.resize((cfg.size, cfg.size))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        img_small = img.resize((30, 30))
        lbl_small = lbl.resize((30, 30))
        lbl_np_small = np.array(lbl_small).astype(np.uint8)
        print(lbl_np_small)
        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.imshow(img)
        fig.add_subplot(2, 2, 2)
        plt.imshow(lbl)
        fig.add_subplot(2, 2, 3)
        plt.imshow(img_small)
        fig.add_subplot(2, 2, 4)
        plt.imshow(lbl_small)
        plt.show()

        img = img_trans(img)
        lbl = np.array(lbl).astype(np.uint8)
        lbl = ade_lbl_trans(lbl)

        return img, lbl


def ade_test():
    dataset = ADEDataset(cfg.ade_root, 'validation')
    dataloader = DataLoader(dataset, 4, True)
    for data in dataloader:
        print(data)
        break

if __name__ == '__main__':
    ade_test()

