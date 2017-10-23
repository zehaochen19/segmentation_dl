import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon

import cfg
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        base = list(models.vgg16_bn(pretrained=True).features.children())
        self.base1 = nn.Sequential(*base[:24])
        self.base2 = nn.Sequential(*base[24:34])
        self.base3 = nn.Sequential(*base[34:])

        self.score32 = nn.Sequential(
            nn.Conv2d(512, 3072, kernel_size=3, padding=1),
            nn.BatchNorm2d(3072, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(3072, 3072, kernel_size=1),
            nn.BatchNorm2d(3072, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(3072, cfg.n_class, kernel_size=1)
        )

        self.score8 = nn.Conv2d(256, cfg.n_class, kernel_size=1)
        self.score16 = nn.Conv2d(512, cfg.n_class, kernel_size=1)

        self.upsample32 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, 4, stride=2, padding=1)
        self.upsample16 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, 4, stride=2, padding=1)
        self.upsample8 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, 16, stride=8, padding=4)

    def forward(self, x):
        x = self.base1(x)
        score8 = self.score8(x)

        x = self.base2(x)
        score16 = self.score16(x)

        x = self.base3(x)
        score32 = self.score32(x)

        score16 = score16 + self.upsample32(score32)
        score8 = score8 + self.upsample16(score16)

        return self.upsample8(score8)


def fcn_test():
    x = torch.randn(1,3,288,288)
    x = Variable(x)
    net = FCN()
    print(net)
    y = net(x)
    print(y.size())



if __name__ == '__main__':
    fcn_test()
