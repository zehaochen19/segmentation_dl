import torchvision
from PIL import Image
import math
import random
import numpy as np
import torch
import cfg


def resized_crop(img, lbl, i, j, h, w, size):
    img = img.crop((j, i, j + w, i + h))
    lbl = lbl.crop((j, i, j + w, i + h))
    img = img.resize((size, size), Image.BILINEAR)
    lbl = lbl.resize((size, size), Image.BILINEAR)

    return img, lbl


class RandomResizedCrop:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.2, 1.0) * area
            aspect_ratio = random.uniform(3. / 5, 5. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        print('fallback')
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, lbl):
        i, j, h, w = self.get_params(img)
        return resized_crop(img, lbl, i, j, h, w, self.size)


class RandomHorizontalFlip:
    def __call__(self, img, lbl):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl):
        return img.resize((self.size, self.size), Image.BILINEAR), lbl.resize((self.size, self.size), Image.BILINEAR)


class ToTensor:
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img, lbl):
        img = self.to_tensor(img)
        lbl = np.array(lbl).astype(np.uint8)
        lbl[lbl == 255] = 0
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


class Normalize:
    def __init__(self, mean, std):
        self.img_normalizer = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img, lbl):
        return self.img_normalizer(img), lbl


class Compose:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, lbl):
        for t in self.trans:
            img, lbl = t(img, lbl)
        return img, lbl


augmentation = Compose([RandomResizedCrop(cfg.size),
                        RandomHorizontalFlip(),
                        ToTensor(),
                        Normalize(cfg.mean, cfg.std)])

basic_trans = Compose([Resize(cfg.size),
                       ToTensor(),
                       Normalize(cfg.mean, cfg.std)])
