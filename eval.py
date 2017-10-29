import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import cfg
from dataset.voc_dataset import VOCDataset
from models.pspnet import PSPNet


def evaluate_accuracy(net, val_loader):
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

    net.train()

    return correct / total


def evaluate_miou(net, loader):
    intersect = [0] * cfg.n_class
    diff = [0] * cfg.n_class + 1
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for img, lbl in loader:
        if torch.cuda.is_available():
            img, lbl = img.cuda(), lbl.cuda()
        img = Variable(img, volatile=True)
        pred = net(img).data
        pred = torch.max(pred, 1)[1]
        for i in range(1, cfg.n_class + 1):
            intersect[i] += torch.sum((lbl == i) == (pred == i))
            diff[i] += torch.sum((lbl == i) != (pred == i))
    iou = [i / (i + d) for (i, d) in zip(intersect, diff)]
    return sum(iou) / cfg.n_class


if __name__ == '__main__':
    val_dataset = VOCDataset(cfg.voc_root, (2007, 'test'))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    net = PSPNet()
    net.load_state_dict(torch.load(os.path.join('save', 'weights'), map_location=lambda storage, loc: storage))
    net.eval()
    correct = 0
    total = 0
    for img, lbl in val_loader:
        img = Variable(img, volatile=True)
        pred = net(img).data
        pred = torch.max(pred, 1)[1]
        correct += torch.sum(pred == lbl)
        total += lbl.numel()
    acc = correct / total
    print(acc)
