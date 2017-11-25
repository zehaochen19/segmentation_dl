import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cfg
from dataset.cityscapes import CityScapesTest, test_collate
import transform
from models.res_lkm import ResLKM
from models.deeplab import DeepLab
import argparse
from PIL import Image
from subprocess import call
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation networks with Pytorch')
    # network name
    parser.add_argument(
        '--name',
        help='name of the network',
        dest='name',
        type=str,
        default='LKM_cityscapes512')
    # batch size
    parser.add_argument(
        '--batch_size',
        help='batch size',
        dest='batch_size',
        type=int,
        default=12)
    # checkpoint
    parser.add_argument(
        '--num_workers',
        help='the number of workers for dataloader',
        dest='num_workers',
        type=int,
        default=4,
    )

    args = parser.parse_args()
    return args


args = parse_arg()
nets = {'LKM': ResLKM, 'DeepLab': DeepLab}


def test_main():
    if not os.path.exists('test'):
        call(['mkdir', 'test'])

    dataset = CityScapesTest(cfg.cityscapes_root, 'test', transform.cityscapes_test)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate
    )
    net = nets[args.name.split('_')[0]](cfg.n_class)
    name = args.name
    save_root = os.path.join('save', name)
    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(os.path.join(save_root, 'weights')))
    else:
        net.load_state_dict(
            torch.load(
                os.path.join(save_root, 'weights'),
                map_location=lambda storage, loc: storage))
    net.eval()

    for names, imgs in loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)

        preds = net(imgs).data
        preds = torch.max(preds, 1)[1]
        preds = preds.cpu()

        for i, name in enumerate(names):
            pred = preds[i].numpy().astype(np.uint8)
            pred = Image.fromarray(pred).resize((2048, 1024))

            pred.save(os.path.join('test', name + '.png'), 'PNG')


if __name__ == '__main__':
    test_main()
