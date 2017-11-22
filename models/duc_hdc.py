import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models


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


class DucHdc(nn.Module):
    layer3_rates = [1, 2, 5, 9]
    layer4_rates = [5, 9, 17]

    def __init__(self, n_class):
        super(DucHdc, self).__init__()
        self.n_class = n_class
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.layer3[0].downsample[0].stride = (1, 1)
        for i in range(3):
            self.layer3[i].conv2.dilation = (self.layer3_rates[i % 4],
                                             self.layer3_rates[i % 4])
            self.layer3[i].conv2.padding = (self.layer3_rates[i % 4],
                                            self.layer3_rates[i % 4])
            self.layer3[i].conv2.stride = (1, 1)

        self.layer4[0].downsample[0].stride = (1, 1)
        for i in range(3):
            self.layer4[i].conv2.dilation = (self.layer4_rates[i % 3],
                                             self.layer4_rates[i % 3])
            self.layer4[i].conv2.padding = (self.layer4_rates[i % 3],
                                            self.layer4_rates[i % 3])
            self.layer4[i].conv2.stride = (1, 1)

        self.aspp = AtrousSpatialPyramidPooling(2048, 512, [1, 6, 12, 18, 24])

        out_channels = 8 * 8 * n_class
        self.duc = nn.Sequential(
            nn.Conv2d(2560, out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(8))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)

        x = self.duc(x)

        return x


def duchdc_test():
    x = Variable(torch.randn(1, 3, 512, 512))
    net = DucHdc(34)
    print(net)
    y = net(x)

    print(y.size())
