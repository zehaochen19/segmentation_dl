import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import cfg
import augment
import numpy as np
import matplotlib.pyplot as plt

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

to_pil = transforms.ToPILImage()


def lbl_trans(label):
    label = torch.from_numpy(label).long()
    label[label == 255] = 0
    return label


class VOCDataset(Dataset):
    def __init__(self, root, split, transform=augment.basic_trans):
        super(VOCDataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self._img_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self._label_path = os.path.join('{}', 'SegmentationClass', '{}.png')
        self.ids = list()
        year,subset = split
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

        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.imshow(img)
        fig.add_subplot(2, 2, 2)
        plt.imshow(label)

        img, label = self.transform(img, label)

        img_, label_ = to_pil(img), Image.fromarray(label.numpy().astype(np.uint8))
        fig.add_subplot(2, 2, 3)
        plt.imshow(img_)
        fig.add_subplot(2, 2, 4)
        plt.imshow(label_)
        plt.show()

        return img, label


def voc_test():
    dataset = VOCDataset(cfg.voc_root, (2012, 'trainval'), augment.augmentation)
    print(len(dataset))
    dataloader = DataLoader(dataset, 10, True)

    for data in dataloader:
        print(data)
        break


if __name__ == '__main__':
    voc_test()
