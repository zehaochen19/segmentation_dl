import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cfg
from dataset.cityscapes import CityScapes
import transform
from models.res_lkm import ResLKM
from models.deeplab import DeepLab
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation networks with Pytorch')
    # network name
    parser.add_argument(
        '--name',
        help='name of the network',
        dest='name',
        type=str,
        default='DeepLab_cityscapes640-512')
    # batch size
    parser.add_argument(
        '--batch_size',
        help='batch size',
        dest='batch_size',
        type=int,
        default=8)
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


def evaluate_accuracy(net, loader):
    correct = 0
    total = 0

    for img, lbl in loader:
        if torch.cuda.is_available():
            img, lbl = img.cuda(), lbl.cuda()
        img = Variable(img, volatile=True)
        pred = net(img).data
        # pred = F.upsample_bilinear(pred, (1024, 2048)).data
        pred = torch.max(pred, 1)[1]
        correct += torch.sum(pred == lbl)
        total += lbl.numel()

    return correct / total


def evaluate_miou(net, loader):
    was_training = net.training
    intersect = [0] * (cfg.n_class - 1)
    union = [0] * (cfg.n_class - 1)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for img, lbl in loader:
        if torch.cuda.is_available():
            img, lbl = img.cuda(), lbl.cuda()
        img = Variable(img, volatile=True)
        pred = net(img).data
        # pred = F.upsample_bilinear(pred, (1024, 2048)).data
        pred = torch.max(pred, 1)[1]
        for i in range(1, cfg.n_class):
            match = (lbl == i) + (pred == i)
            it = torch.sum(match == 2)
            un = torch.sum(match > 0)

            intersect[i - 1] += it
            union[i - 1] += un

    iou = []
    for i in range(len(intersect)):
        if union[i] != 0:
            iou.append(intersect[i] / union[i])
    print('Available number of categories: {}'.format(len(iou)))
    if was_training:
        net.train()
    return sum(iou) / len(iou)


def main():
    dataset = CityScapes(cfg.cityscapes_root, 'val', transform.cityscapes_val)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
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
    accuracy = evaluate_accuracy(net, loader)
    print('Accuracy : {:.6f}%'.format(accuracy * 100))
    miou = evaluate_miou(net, loader)
    print('mIOU : {:.6f}%'.format(miou * 100))


if __name__ == '__main__':
    main()
