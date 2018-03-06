import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from models.deeplab import DeepLab
from models.res_lkm import ResLKM
import cfg

normalizer = transforms.Normalize(cfg.mean, cfg.std)
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def demo_main(img_root):
    imgs = os.listdir(img_root)

    net = ResLKM(cfg.n_class)
    state_dict = torch.load(
        os.path.join('save', 'LKM_cityscapes512', 'weights'),
        map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict=state_dict)
    net.eval()
    for _ in range(10):
        i = random.randrange(0, len(imgs))
        img = os.path.join(img_root, imgs[i])
        img = Image.open(img)
        w, h = img.size

        img_ = img.resize((cfg.pre_resize_w, cfg.pre_resize_h), Image.BILINEAR)

        img_ = normalizer(to_tensor(img_)).unsqueeze(0)
        img_ = Variable(img_, volatile=True)

        pred = net(img_).data.squeeze().max(0)[1]
        print(pred)
        pred = Image.fromarray(
            pred.numpy().astype(np.uint8), mode='L').resize((w, h))

        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        fig.add_subplot(1, 2, 2)
        plt.imshow(pred)
        plt.show()


if __name__ == '__main__':
    city_root = os.path.join(cfg.cityscapes_root, 'leftImg8bit/val/frankfurt')
    demo_main(city_root)
