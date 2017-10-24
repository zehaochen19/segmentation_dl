from torchvision.models import vgg16_bn
from subprocess import call
import time

def whole_vgg_test():
    v = vgg16_bn(pretrained=True)
    v.cuda()
    time.sleep(10)
    call(['nvidia-smi'])
    time.sleep(5)

def part_vgg_test():
    v = vgg16_bn(pretrained=True).features
    v.cuda()
    time.sleep(10)
    call(['nvidia-smi'])
    time.sleep(5)

if __name__ == '__main__':
    part_vgg_test()