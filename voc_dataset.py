import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import cfg
import random
import numpy as np

img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])


def lbl_trans(label):
    label = torch.from_numpy(label).long()
    label[label == 255] = 0
    return label


class VOCDataset(Dataset):
    def __init__(self, root, split):
        super(VOCDataset, self).__init__()
        self.root = root
        self.split = split

        self._img_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self._label_path = os.path.join('{}', 'SegmentationClass', '{}.png')
        self.ids = list()
        for year, subset in split:
            sub_path = os.path.join(root, 'VOC' + str(year))
            for line in open(os.path.join(sub_path, 'ImageSets', 'Segmentation', subset + '.txt')):
                self.ids.append((sub_path, line.strip()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = self._img_path.format(*self.ids[idx])
        img = Image.open(img_path).convert('RGB')
        label_path = self._label_path.format(*self.ids[idx])
        label = Image.open(label_path)

        img, label = img.resize((cfg.size, cfg.size)), label.resize((cfg.size, cfg.size))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        img = img_trans(img)

        label = np.array(label).astype(np.uint8)
        label = lbl_trans(label)

        return img, label


def voc_test():
    dataset = VOCDataset(cfg.voc_root, [(2007, 'test')])
    print(len(dataset))
    dataloader = DataLoader(dataset, 4, True)

    for data in dataloader:
        print(data)
        break


if __name__ == '__main__':
    voc_test()
