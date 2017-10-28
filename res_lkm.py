import cfg
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


class GlobalConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(GlobalConvolutionalNetwork, self).__init__()
        pad = k // 2
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=(pad, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, k), padding=(0, pad)),
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding=(0, pad)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(k, 1), padding=(pad, 0)),
        )

    def forward(self, x):
        return self.left(x) + self.right(x)


class BoundaryRefineModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BoundaryRefineModule, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.layer(x)


class ResLKM(nn.Module):
    def __init__(self):
        super(ResLKM, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn_4 = GlobalConvolutionalNetwork(256, cfg.n_class, 7)
        self.gcn_8 = GlobalConvolutionalNetwork(512, cfg.n_class, 7)
        self.gcn_16 = GlobalConvolutionalNetwork(1024, cfg.n_class, 7)
        self.gcn_32 = GlobalConvolutionalNetwork(2048, cfg.n_class, 7)

        self.br_1 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_2 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_4_1 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_4_2 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_8_1 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_8_2 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_16_1 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_16_2 = BoundaryRefineModule(cfg.n_class, cfg.n_class)
        self.br_32 = BoundaryRefineModule(cfg.n_class, cfg.n_class)

        self.deconv_2 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, kernel_size=4, stride=2, padding=1)
        self.deconv_4 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, kernel_size=4, stride=2, padding=1)
        self.deconv_8 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, kernel_size=4, stride=2, padding=1)
        self.deconv_16 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, kernel_size=4, stride=2, padding=1)
        self.deconv_32 = nn.ConvTranspose2d(cfg.n_class, cfg.n_class, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.layer0(x)  # 1 / 2
        x = self.layer1(x)  # 1 / 4

        score4 = self.br_4_1(self.gcn_4(x))

        x = self.layer2(x)  # 1 / 8

        score8 = self.br_8_1(self.gcn_8(x))

        x = self.layer3(x)  # 1 / 16

        score16 = self.br_16_1(self.gcn_16(x))

        x = self.layer4(x)  # 1 / 32

        score32 = self.br_32(self.gcn_32(x))

        score16 = self.br_16_2(score16 + self.deconv_32(score32))
        score8 = self.br_8_2(score8 + self.deconv_16(score16))
        score4 = self.br_4_2(score4 + self.deconv_8(score8))

        return self.br_1(self.deconv_2(self.br_2(self.deconv_4(score4))))


def gcn_test():
    x = Variable(torch.randn(1, 3, cfg.size, cfg.size))
    net = ResLKM()
    y = net(x)
    print(y)


if __name__ == '__main__':
    gcn_test()
