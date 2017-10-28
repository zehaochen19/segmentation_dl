import cfg
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class DUCHDC(nn.Module):
    layer3_rates = [1, 2, 4, 7]
    layer4_rates = [4, 7, 11]

    def __init__(self):
        super(DUCHDC, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = (self.layer3_rates[idx % 3], self.layer3_rates[idx % 3])
            self.layer3[idx].conv2.padding = (self.layer3_rates[idx % 3], self.layer3_rates[idx % 3])
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = (self.layer4_rates[idx], self.layer4_rates[idx])
            self.layer4[idx].conv2.padding = (self.layer4_rates[idx], self.layer4_rates[idx])

        out_channels = 8 * 8 * cfg.n_class
        self.duc = nn.Sequential(
            nn.Conv2d(2048, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.duc(x)

        x = x.view(-1, cfg.n_class, cfg.size, cfg.size)
        return x


def duchdc_test():
    x = Variable(torch.randn(1, 3, 320, 320))
    net = DUCHDC()
    y = net(x)
    print(y)


if __name__ == '__main__':
    duchdc_test()
