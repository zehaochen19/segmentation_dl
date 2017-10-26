import cfg
from fcn import FCN
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import os
import torch
import random
from PIL import Image
from torch.autograd import Variable

normalizer = transforms.Normalize(cfg.mean, cfg.std)
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def demo_main(img_root):
    imgs = os.listdir(img_root)

    net = FCN()
    net.load_state_dict(torch.load(os.path.join('save', 'weights'), map_location=lambda storage, loc: storage))
    net.eval()
    for _ in range(10):
        i = random.randrange(0, len(imgs))
        img = os.path.join(img_root, imgs[i])
        img = Image.open(img)
        w, h = img.size

        img_ = img.resize((cfg.size, cfg.size))
        img_ = normalizer(to_tensor(img_)).unsqueeze(0)
        img_ = Variable(img_, volatile=True)

        pred = net(img_).data.squeeze().max(0)[1]
        print(pred.size())
        pred = Image.fromarray(pred.numpy().astype(np.uint8)).resize((w, h))

        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        fig.add_subplot(1, 2, 2)
        plt.imshow(pred)
        plt.show()


if __name__ == '__main__':
    img_root = os.path.join(cfg.voc_root, 'VOC2012', 'JPEGImages')
    demo_main(img_root)
