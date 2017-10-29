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
from eval import evaluate_miou
from models.res_lkm import ResLKM


def train(net, name, train_loader, val_loader, load_checkpoint, learning_rate, num_epochs, weight_decay, checkpoint,
          dropbox):
    records = {'losses': [], 'mious': []}
    if torch.cuda.is_available():
        net.cuda()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_root = os.path.join('save', name)
    if not os.path.exists(save_root):
        call(['mkdir', '-p', save_root])

    if load_checkpoint:
        save_files = set(os.listdir('save'))
        if {'weights', 'optimizer', 'losses', 'acc'} <= save_files:
            print('Loading checkpoint')
            net.load_state_dict(torch.load(os.path.join('save', 'weights')))
            optimizer.load_state_dict(torch.load(os.path.join('save', 'optimizer')))
            with open(os.path.join('save', 'records'), 'rb') as f:
                records = pickle.load(f)

        else:
            print('Checkpoint files don\'t exist.')
            print('Skip loading checkpoint')

    last_epoch = len(records['losses']) - 1

    # scheduler = lr_scheduler.LambdaLR(optimizer, lambda e: math.pow((1 - e / num_epochs), 0.9), last_epoch)
    # accuracy = evaluate_accuracy(net, val_loader)
    # print('Accuracy before training {}'.format(accuracy))
    print('Start training')
    iter_loss = 0.0
    iter_count = 0
    for epoch in range(last_epoch + 1, num_epochs):
        t0 = time.time()
        # scheduler.step()
        running_loss = 0.0
        for img, lbl in train_loader:
            if torch.cuda.is_available():
                img, lbl = img.cuda(), lbl.cuda()
            img, lbl = Variable(img, requires_grad=False), Variable(lbl, requires_grad=False)
            # pred, aux = net(img)
            pred = net(img)
            optimizer.zero_grad()
            loss = criterion(pred, lbl)
            loss.backward()
            optimizer.step()

            _loss = loss.data[0]
            running_loss += _loss
            iter_loss += _loss
            iter_count = (iter_count + 1) % 10
            if iter_count % 10 == 0:
                print('\rLoss of last 10 iterations {:.2f}'.format(iter_loss), end='')

                iter_loss = 0.0
        t1 = time.time()
        miou = evaluate_miou(net, val_loader)
        print('\rEpoch {} : Loss {:.2f} mIOU {:.4f}% Time {:.2f}min'.format(
            epoch + 1, running_loss, miou * 100, (t1 - t0) / 60))
        records['losses'].append(running_loss)
        records['mious'].append(miou)

        if (epoch + 1) % checkpoint == 0:
            print('\rSaving checkpoint', end='')
            torch.save(net.state_dict(), os.path.join(save_root, 'weights'))
            torch.save(optimizer.state_dict(), os.path.join(save_root, 'optimizer'))
            with open(os.path.join(save_root, 'records'), 'wb') as f:
                pickle.dump(records, f)
            if dropbox:
                call(['cp', '-r', save_root, os.path.join(cfg.home, 'Dropbox')])
            print('\rFinish saving checkpoint', end='')

    print('Finish training')


def main():
    train_dataset = CityScapes(cfg.cityscapes_root, 'train', augment.cityscapes_trans)
    val_dataset = CityScapes(cfg.cityscapes_root, 'val', augment.cityscapes_val)
    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=False)
    net = ResLKM()
    train(net, 'LKM', train_loader, val_loader, True, 0.0001, 200, 0.0, 1, True)


if __name__ == '__main__':
    main()
