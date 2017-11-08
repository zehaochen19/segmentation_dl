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


def train(net, name, train_loader, load_checkpoint, learning_rate,
          num_epochs, weight_decay, checkpoint, dropbox):
    records = {'losses': []}
    if torch.cuda.is_available():
        net.cuda()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    save_root = os.path.join('save', name)
    if not os.path.exists(save_root):
        call(['mkdir', '-p', save_root])

    if load_checkpoint:
        save_files = set(os.listdir(save_root))
        if {'weights', 'optimizer', 'records'} <= save_files:
            print('Loading checkpoint')
            net.load_state_dict(torch.load(os.path.join(save_root, 'weights')))
            optimizer.load_state_dict(torch.load(
                os.path.join(save_root, 'optimizer')))
            with open(os.path.join(save_root, 'records'), 'rb') as f:
                records = pickle.load(f)

        else:
            print('Checkpoint files don\'t exist.')
            print('Skip loading checkpoint')

    last_epoch = len(records['losses']) - 1

    scheduler = LambdaLR(optimizer,
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
            img, lbl = Variable(img, requires_grad=False), Variable(lbl, requires_grad=False)

            pred = net(img)
            optimizer.zero_grad()
            loss = criterion(pred, lbl)

            loss.backward()
            optimizer.step()

            _loss = loss.data[0]
            running_loss += _loss

            print('\rEpoch {} Iter {} Loss {:.4f}'.format(
                epoch, iter_count, _loss), end='')
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
                call(['cp', '-r', save_root,
                      os.path.join(cfg.home, 'Dropbox')])
            print('\rFinish saving checkpoint', end='')

    print('Finish training')


def main():
    train_dataset = CityScapes(
        cfg.cityscapes_root, 'train', augment.cityscapes_train)

    if torch.cuda.is_available():
        train_loader = DataLoader(
            train_dataset, batch_size=9, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=9, shuffle=True, pin_memory=False)

    net = ResLKM()
    train(net, 'LKM', train_loader, True, 0.001, 100, 0.0, 1, True)


if __name__ == '__main__':
    main()
