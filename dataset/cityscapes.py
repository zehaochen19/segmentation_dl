from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import cfg
import transform
import torch


# import numpy as np
# import matplotlib.pyplot as plt
# to_pil = transforms.ToPILImage()


class CityScapes(Dataset):
    def __init__(self, root, split, transform):
        super(CityScapes, self).__init__()
        self.root = os.path.join(root, 'leftImg8bit', split)
        self.transform = transform

        self.img_paths = []

        self._img = os.path.join(root, 'leftImg8bit', split,
                                 '{}_leftImg8bit.png')
        self._lbl = os.path.join(root, 'gtFine', split,
                                 '{}_gtFine_labelIds.png')

        cities = [
            city for city in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, city))
        ]

        for city in cities:
            for img in os.listdir(os.path.join(self.root, city)):
                if len(img) > 3 and img[-4:] == '.png':
                    self.img_paths.append(os.path.join(city, img[:-16]))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, lbl_path = self._img.format(
            self.img_paths[idx]), self._lbl.format(self.img_paths[idx])
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        # fig = plt.figure()
        # fig.add_subplot(2, 2, 1)
        # plt.imshow(img)
        # fig.add_subplot(2, 2, 2)
        # plt.imshow(lbl)

        img, lbl = self.transform(img, lbl)

        # print(img.size(),lbl.size())
        # img_, label_ = to_pil(img), Image.fromarray(lbl.numpy().astype(np.uint8))
        # fig.add_subplot(2, 2, 3)
        # plt.imshow(img_)
        #
        # fig.add_subplot(2, 2, 4)
        # plt.imshow(label_)
        # plt.show()

        return img, lbl


class CityScapesTest(Dataset):
    def __init__(self, root, split, transform):
        super(CityScapesTest, self).__init__()
        self.root = os.path.join(root, 'leftImg8bit', split)
        self.transform = transform

        self.img_paths = []

        self._img = os.path.join(root, 'leftImg8bit', split,
                                 '{}_leftImg8bit.png')

        cities = [
            city for city in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, city))
        ]

        for city in cities:
            for img in os.listdir(os.path.join(self.root, city)):
                if len(img) > 3 and img[-4:] == '.png':
                    self.img_paths.append(os.path.join(city, img[:-16]))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self._img.format(self.img_paths[idx])
        img = Image.open(img_path)
        img = self.transform(img)

        return self.img_paths[idx].split('/')[1], img


def cs_test():
    dataset = CityScapes(cfg.cityscapes_root, 'train',
                         transform.cityscapes_train)
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=24, shuffle=True)
    for data in loader:
        print(data)
        break


def test_collate(data):
    names = []
    imgs = []

    for name, img in data:
        names.append(name)
        imgs.append(img)

    imgs = torch.stack(imgs)

    return names, imgs


if __name__ == '__main__':
    a = CityScapesTest(cfg.cityscapes_root, 'test', transform.cityscapes_test)
    loader = DataLoader(a, batch_size=4, shuffle=True,collate_fn=test_collate)
    for x in loader:
        print(x)
        break
