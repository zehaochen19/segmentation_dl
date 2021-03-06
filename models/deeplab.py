import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.pools = nn.ModuleList()
        for r in config:
            self.pools.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=r,
                        dilation=r), nn.BatchNorm2d(out_channels)))

    def forward(self, x):
        result = []
        for p in self.pools:
            result.append(p(x))
        result = torch.cat(result, 1)
        result = F.relu(result)
        return result


class DeepLab(nn.Module):
    def __init__(self, n_class):
        super(DeepLab, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.layer4[0].downsample[0].stride = (1, 1)
        for i in range(3):
            self.layer4[i].conv2.dilation = (2, 2)
            self.layer4[i].conv2.padding = (2, 2)
            self.layer4[i].conv2.stride = (1, 1)

        self.conv1 = nn.Conv2d(2048, 1024, 1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.aspp = AtrousSpatialPyramidPooling(2048, 512, [3, 6, 9, 12, 15])
        self.conv2 = nn.Conv2d(2560, 1024, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.pred = nn.Conv2d(1024, n_class, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(
            n_class, n_class, kernel_size=32, stride=16, padding=8)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.aspp(x)
        x2 = self.bn2(self.conv2(x2))
        x = x1 + x2
        x = self.pred(x)

        return self.upsample(x)


def deeplab_test():
    x = Variable(torch.randn(1, 3, 512, 512))
    net = DeepLab(34)
    print(net)
    y = net(x)

    print(y.size())


if __name__ == '__main__':
    deeplab_test()
