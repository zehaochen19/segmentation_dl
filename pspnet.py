import cfg
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    sizes = [6, 3, 2, 1]

    def __init__(self, in_channels, out_channels):
        super(PyramidPoolingModule, self).__init__()
        self.sub_layers = nn.ModuleList()
        for size in self.sizes:
            self.sub_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        result = [x]
        original_size = x.size()[2:]
        for layer in self.sub_layers:
            result.append(F.upsample(layer(x), original_size, mode='bilinear'))
        result = torch.cat(result, 1)
        return result


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        resnet = models.resnet101(pretrained=False)
        self.base0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.base1, self.base2, self.base3, self.base4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.base3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.base4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = PyramidPoolingModule(2048, 512)

        self.auxiliary = nn.Conv2d(1024, cfg.n_class, kernel_size=1)

        self.prediction_layer = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, cfg.n_class, kernel_size=1)
        )

    def forward(self, x):
        original_size = x.size()[2:]
        x = self.base0(x)
        x = self.base1(x)
        x = self.base2(x)
        x = self.base3(x)
        if self.training:
            aux = self.auxiliary(x)
        x = self.base4(x)
        x = self.ppm(x)
        x = self.prediction_layer(x)

        if self.training:
            return F.upsample(x, original_size, mode='bilinear'), F.upsample(aux, original_size, mode='bilinear')
        else:
            return F.upsample(x, original_size, mode='bilinear')


def psp_test():
    x = Variable(torch.randn(1, 3, 320, 320))
    net = PSPNet()
    y = net(x)
    print(y)


if __name__ == '__main__':
    psp_test()
