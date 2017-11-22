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
    rates = [1, 2, 4]

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
            self.layer4[i].conv2.dilation = (self.rates[i], self.rates[i])
            self.layer4[i].conv2.padding = (self.rates[i], self.rates[i])
            self.layer4[i].conv2.stride = (1, 1)
        # self.resnext = resnext101_32x4d().features
        # self.resnext = resnext101_32x4d().features
        # for i in range(len(self.resnext[7])):
        #     self.resnext[7][i][0][0][0][3].stride = (1, 1)
        #     self.resnext[7][i][0][0][0][3].dilation = (self.rates[i % 3],
        #                                                self.rates[i % 3])
        #     self.resnext[7][i][0][0][0][3].padding = (self.rates[i % 3],
        #                                               self.rates[i % 3])

        # self.resnext[7][0][0][1][0].stride = (1, 1)

        self.aspp = AtrousSpatialPyramidPooling(2048, 512, [1, 4, 8, 12])
        self.conv = nn.Conv2d(2048, 1024, kernel_size=1)
        self.bn = nn.BatchNorm2d(1024)
        self.pred = nn.Conv2d(1024, n_class, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(
            n_class, n_class, kernel_size=32, stride=16, padding=8)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.resnext(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = F.relu(self.bn(x))
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
