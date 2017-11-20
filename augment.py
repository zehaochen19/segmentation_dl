import torchvision
from PIL import Image
import random
import numpy as np
import torch
import cfg


def resized_crop(img, lbl, i, j, h, w, size):
    img = img.crop((j, i, j + w, i + h))
    lbl = lbl.crop((j, i, j + w, i + h))
    img = img.resize((size, size), Image.BILINEAR)
    lbl = lbl.resize((size, size), Image.NEAREST)

    return img, lbl


class RandomHorizontalFlip:
    def __call__(self, img, lbl):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl


class Resize:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img, lbl):
        return img.resize((self.w, self.h), Image.BILINEAR), lbl.resize(
            (self.w, self.h), Image.NEAREST)


class TestResize:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call(self, img, lbl):
        return img.resize((self.w, self.h), Image.BILINEAR), lbl


class ResizeShort:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl):
        w, h = img.size

        if w < h:
            new_h = int(round(h / w) * self.size)
            img = img.resize((self.size, new_h), Image.BILINEAR)
            lbl = lbl.resize((self.size, new_h), Image.NEAREST)
        else:
            new_w = int(round(w / h) * self.size)
            img = img.resize((new_w, self.size), Image.BILINEAR)
            lbl = lbl.resize((new_w, self.size), Image.NEAREST)


class ToTensor:
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img, lbl):
        img = self.to_tensor(img)
        lbl = np.array(lbl).astype(np.uint8)
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


class RandomCrop:
    def __init__(self, crop_size=None):
        # assert min_scale <= max_scale

        self.crop_size = crop_size

        # self.min_scale = min_scale
        # self.max_scale = max_scale

    def __call__(self, img, lbl):
        if self.crop_size:
            crop = self.crop_size
        else:
            crop = min(img.size)

        x = random.randint(0, img.size[0] - crop)
        y = random.randint(0, img.size[1] - crop)

        img = img.crop((x, y, x + crop, y + crop))
        lbl = lbl.crop((x, y, x + crop, y + crop))
        return img, lbl


class UnitResize:
    def __init__(self, unit):
        self.unit = unit

    def __call__(self, img, lbl):
        w, h = img.size
        if w % self.unit == 0 and h % self.unit == 0:
            return img, lbl
        w = int(round(w / self.unit) * self.unit)
        h = int(round(h / self.unit) * self.unit)
        return img.resize((w, h), Image.BILINEAR), lbl.resize((w, h))


cityscapes_train = Compose([
    Resize(cfg.pre_resize_w, cfg.pre_resize_h),
    RandomCrop(crop_size=cfg.size),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(cfg.mean, cfg.std)
])

cityscapes_val = Compose([
    TestResize(cfg.pre_resize_w, cfg.pre_resize_h),
    ToTensor(),
    Normalize(cfg.mean, cfg.std)
])

cityscapes_test = Compose(
    [UnitResize(32), ToTensor(),
     Normalize(cfg.mean, cfg.std)])

cityscapes_t = Compose(
    [Resize(cfg.size * 2, cfg.size),
     ToTensor(),
     Normalize(cfg.mean, cfg.std)])
