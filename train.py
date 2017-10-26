import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.optim import lr_scheduler
import augment
from voc_dataset import VOCDataset
from pspnet import PSPNet
import cfg
import os
import pickle
from subprocess import call
from eval import evaluate_accuracy
import time


def train(net, train_loader, val_loader, load_checkpoint, learning_rate, num_epochs, weight_decay, checkpoint, dropbox):
    losses = []

    accuracies = []

    if torch.cuda.is_available():
        net.cuda()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if not os.path.exists('save'):
        call(['mkdir', 'save'])

    if load_checkpoint:
        save_files = set(os.listdir('save'))
        if {'weights', 'optimizer', 'losses', 'acc'} <= save_files:
            print('Loading checkpoint')
            net.load_state_dict(torch.load(os.path.join('save', 'weights')))
            optimizer.load_state_dict(torch.load(os.path.join('save', 'optimizer')))
            with open(os.path.join('save', 'losses'), 'rb') as f:
                losses = pickle.load(f)
            with open(os.path.join('save', 'acc'), 'rb') as f:
                accuracies = pickle.load(f)
        else:
            print('Checkpoint files don\'t exist.')
            print('Skip loading checkpoint')

    last_epoch = len(losses) - 1
    scheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.7, last_epoch=last_epoch)
    # accuracy = evaluate_accuracy(net, val_loader)
    # print('Accuracy before training {}'.format(accuracy))
    # print('Start training')
    iter_loss = 0.0
    iter_count = 0
    for epoch in range(last_epoch + 1, num_epochs):
        t0 = time.time()
        scheduler.step()
        running_loss = 0.0
        for img, lbl in train_loader:
            if torch.cuda.is_available():
                img, lbl = img.cuda(), lbl.cuda()
            img, lbl = Variable(img, requires_grad=False), Variable(lbl, requires_grad=False)
            pred, aux = net(img)
            # pred = net(img)
            optimizer.zero_grad()
            loss = criterion(pred, lbl) + 0.4 * criterion(aux, lbl)
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
        accuracy = evaluate_accuracy(net, val_loader)
        print('\rEpoch {} : Loss {:.2f} Accuracy {:.4f}% Time {:.2f}min'.format(epoch + 1, running_loss, accuracy * 100,
                                                                                (t1 - t0) / 60))
        losses.append(running_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % checkpoint == 0:
            print('\rSaving checkpoint', end='')
            torch.save(net.state_dict(), os.path.join('save', 'weights'))
            torch.save(optimizer.state_dict(), os.path.join('save', 'optimizer'))
            with open(os.path.join('save', 'losses'), 'wb') as f:
                pickle.dump(losses, f)
            with open(os.path.join('save', 'acc'), 'wb') as f:
                pickle.dump(accuracies, f)
            if dropbox:
                call(['cp', '-r', './save', os.path.join(cfg.home, 'Dropbox')])
            print('\rFinish saving checkpoint', end='')

    print('Finish training')


def main():
    train_dataset = VOCDataset(cfg.voc_root, (2012, 'trainval'), transform=augment.augmentation)
    val_dataset = VOCDataset(cfg.voc_root, (2007, 'test'), transform=augment.basic_trans)
    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=21, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=False)
    net = PSPNet()
    train(net, train_loader, val_loader, True, 0.001, 300, 0.0, 1, True)


if __name__ == '__main__':
    main()
