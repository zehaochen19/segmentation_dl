import cfg
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(AtrousSpatialPyramidPooling, self).__init__()
        # self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pools = nn.ModuleList()
        for r in config:
            self.pools.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r),
                                            nn.BatchNorm2d(out_channels)))
        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        result = [self.bn(self.conv(x))]
        for p in self.pools:
            result.append(p(x))
        x = F.upsample(self.img_pool(x), x.size()[2:])
        result.append(x)
        result = torch.cat(result, 1)
        return result


class DeepLab(nn.Module):
    rates = [1, 2, 4]

    def __init__(self):
        super(DeepLab, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.layer4[0].downsample[0].stride = (1, 1)
        for i in range(3):
            self.layer4[i].conv2.dilation = (self.rates[i], self.rates[i])
            self.layer4[i].conv2.padding = (self.rates[i], self.rates[i])
            self.layer4[i].conv2.stride = (1, 1)

        self.aspp = AtrousSpatialPyramidPooling(2048, 256, [6, 12, 18])
        self.conv = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn = nn.BatchNorm2d(256)
        self.pred = nn.Conv2d(256, cfg.n_class, kernel_size=1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.pred(x)

        return F.upsample_bilinear(x, (cfg.size, cfg.size))


def deeplab_test():
    x = Variable(torch.randn(1, 3, 512, 512))
    net = DeepLab()
    print(net)
    y = net(x)

    print(y)


if __name__ == '__main__':
    deeplab_test()
