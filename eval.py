import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cfg
from dataset.cityscapes import CityScapes
import augment
from models.deeplab import DeepLab
from models.res_lkm import ResLKM


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
    diff = [0] * (cfg.n_class - 1)
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
            intersect[i - 1] += torch.sum((lbl == i) == (pred == i))
            diff[i - 1] += torch.sum((lbl == i) != (pred == i))
    iou = [i / (i + d) for (i, d) in zip(intersect, diff)]

    if was_training:
        net.train()
    return sum(iou) / cfg.n_class


def main():
    dataset = CityScapes(cfg.cityscapes_root, 'val', augment.cityscapes_val)
    loader = DataLoader(dataset, batch_size=24, shuffle=False)
    net = DeepLab()
    name = 'DeepLab'

    save_root = os.path.join('save', name)
    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(os.path.join(save_root, 'weights')))
    else:
        net.load_state_dict(torch.load(os.path.join(save_root, 'weights'), map_location=lambda storage, loc: storage))

    net.eval()
    accuracy = evaluate_accuracy(net, loader)
    print('Accuracy : {:.6f}%'.format(accuracy * 100))
    miou = evaluate_miou(net, loader)
    print('mIOU : {:.6f}%'.format(miou * 100))


if __name__ == '__main__':
    main()
