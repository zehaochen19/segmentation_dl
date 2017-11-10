import os
import pickle
import time
from subprocess import call
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import augment
import cfg
from dataset.cityscapes import CityScapes
import math
from torch.optim.lr_scheduler import LambdaLR
from models.res_lkm import ResLKM
from models.deeplab import DeepLab
import argparse

nets = {'LKM': ResLKM, 'DeepLab': DeepLab}


def parse_arg():
    parser = argparse.ArgumentParser(
        description='Training segmentation networks with Pytorch')
    # network name
    parser.add_argument(
        '--name',
        help='name of the network',
        dest='name',
        type=str,
        default='LKM_512_cityscapes')
    # use dropbox
    parser.add_argument(
        '--dropbox',
        help='copy save files to dropbox',
        dest='dropbox',
        type=bool,
        action='store_true')
    # learning rate
    parser.add_argument(
        '--lr', help='learning rate', dest='lr', type=float, default=0.005)
    # weight decay
    parser.add_argument(
        '--weight_decay',
        help='weight decay',
        dest='wd',
        type=float,
        default=0.0001)
    # batch size
    parser.add_argument(
        '--batch_size',
        help='batch size',
        dest='batch_size',
        type=int,
        default=8)
    # num epoch
    parser.add_argument(
        '--num_epoch',
        help='number of epoch',
        dest='num_epoch',
        type=int,
        default=90)
    # checkpoint
    parser.add_argument(
        '--checkpoint',
        help='period of epochs to checkpoint',
        dest='checkpoint',
        type=int,
        default=1)

    args = parser.parse_args()
    return args


args = parse_arg()


def train(name, train_loader, load_checkpoint, learning_rate, num_epochs,
          weight_decay, checkpoint, dropbox):
    net = nets[name.split('_' [0])]()
    records = {'losses': []}
    if torch.cuda.is_available():
        net.cuda()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9)
    save_root = os.path.join('save', name)
    if not os.path.exists(save_root):
        call(['mkdir', '-p', save_root])

    if load_checkpoint:
        save_files = set(os.listdir(save_root))
        if {'weights', 'optimizer', 'records'} <= save_files:
            print('Loading checkpoint')
            net.load_state_dict(torch.load(os.path.join(save_root, 'weights')))
            optimizer.load_state_dict(
                torch.load(os.path.join(save_root, 'optimizer')))
            with open(os.path.join(save_root, 'records'), 'rb') as f:
                records = pickle.load(f)

        else:
            print('Checkpoint files don\'t exist.')
            print('Skip loading checkpoint')

    last_epoch = len(records['losses']) - 1

    scheduler = LambdaLR(
        optimizer,
        lambda e: math.pow(1 - e / num_epochs, 0.9),
        last_epoch=last_epoch)

    for epoch in range(last_epoch + 1, num_epochs):
        iter_count = 0
        t0 = time.time()
        scheduler.step()

        running_loss = 0.0

        for img, lbl in train_loader:
            if torch.cuda.is_available():
                img, lbl = img.cuda(), lbl.cuda()
            img, lbl = Variable(
                img, requires_grad=False), Variable(
                    lbl, requires_grad=False)

            pred = net(img)
            optimizer.zero_grad()
            loss = criterion(pred, lbl)

            loss.backward()
            optimizer.step()

            _loss = loss.data[0]
            running_loss += _loss

            print(
                '\rEpoch {} Iter {} Loss {:.4f}'.format(
                    epoch, iter_count, _loss),
                end='')
            iter_count += 1

        t1 = time.time()
        # accuracy = evaluate_accuracy(net, val_loader)
        print('\rEpoch {} : Loss {:.4f}  Time {:.2f}min'.format(
            epoch, running_loss, (t1 - t0) / 60))
        records['losses'].append(running_loss)
        # records['accuracies'].append(accuracy)

        if (epoch + 1) % checkpoint == 0:
            print('\rSaving checkpoint', end='')
            torch.save(net.state_dict(), os.path.join(save_root, 'weights'))
            torch.save(optimizer.state_dict(),
                       os.path.join(save_root, 'optimizer'))
            with open(os.path.join(save_root, 'records'), 'wb') as f:
                pickle.dump(records, f)
            if dropbox:
                call(
                    ['cp', '-r', save_root,
                     os.path.join(cfg.home, 'Dropbox')])
            print('\rFinish saving checkpoint', end='')

    print('Finish training')


def main():
    train_dataset = CityScapes(cfg.cityscapes_root, 'train',
                               augment.cityscapes_train)

    if torch.cuda.is_available():
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False)

    train(args.name, train_loader, True, args.lr, args.num_epoch, args.wd,
          args.checkpoint, args.dropbox)


if __name__ == '__main__':
    main()
