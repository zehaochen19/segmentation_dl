import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cfg
from dataset.cityscapes import CityScapes
import augment

from models.res_lkm import ResLKM
import numpy as np


def evaluate_accuracy(net, val_loader):
    was_training = net.training
    correct = 0
    total = 0
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for img, lbl in val_loader:
        if torch.cuda.is_available():
            img, lbl = img.cuda(), lbl.cuda()
        img = Variable(img, volatile=True)
        pred = net(img).data
        pred = torch.max(pred, 1)[1]
        correct += torch.sum(pred == lbl)
        total += lbl.numel()

    if was_training:
        net.train()

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

    if was_training:
        net.train()
    return sum(iou) / len(iou)


def main():
    dataset = CityScapes(cfg.cityscapes_root, 'val', augment.cityscapes_val)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    net = ResLKM()
    name = 'LKM'

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
    miou = evaluate_miou(net, loader)
    print('mIOU : {:.6f}%'.format(miou * 100))
    accuracy = evaluate_accuracy(net, loader)
    print('Accuracy : {:.6f}%'.format(accuracy * 100))


def draft():
    dataset = CityScapes(cfg.cityscapes_root, 'val', augment.cityscapes_val)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    lbl_set = set()
    for img, lbl in loader:
        lbl_set |= set(np.unique(lbl.numpy()))
        print(lbl_set)


if __name__ == '__main__':
    main()
