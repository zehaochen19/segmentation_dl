import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim, nn

from voc_dataset import VOCDataset
from fcn import FCN
import cfg
import os
import pickle
from subprocess import call


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


def train(train_loader, val_loader, load_checkpoint, learning_rate, num_epochs, weight_decay, checkpoint, dropbox):
    net = FCN()
    losses = []
    accuracies = []

    if torch.cuda.is_available():
        net.cuda()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if load_checkpoint:
        net.load_state_dict(torch.load(os.path.join('save', 'weights')))
        optimizer.load_state_dict(torch.load(os.path.join('save', 'optimizer')))
        with open(os.path.join('save', 'losses'), 'rb') as f:
            losses = pickle.load(f)
        with open(os.path.join('save', 'acc'), 'rb') as f:
            accuracies = pickle.load(f)

    print(losses)
    print(accuracies)
    accuracy = evaluate_accuracy(net, val_loader)
    print('Accuracy before training {}'.format(accuracy))

    print('Start training')
    iter_loss = 0.0
    iter_count = 0
    for epoch in range(len(losses), num_epochs):
        running_loss = 0.0
        for img, lbl in train_loader:
            if torch.cuda.is_available():
                img, lbl = img.cuda(), lbl.cuda()
            img, lbl = Variable(img), Variable(lbl)
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
                print('\rLoss of last 10 iterations: {}'.format(iter_loss), end='')
                iter_loss = 0.0

        accuracy = evaluate_accuracy(net, val_loader)
        print('\rEpoch {} : Loss {} Accuracy {}'.format(epoch + 1, running_loss, accuracy))
        losses.append(running_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % checkpoint == 0:
            torch.save(net.state_dict(), os.path.join('save', 'weights'))
            torch.save(optimizer.state_dict(), os.path.join('save', 'optimizer'))
            with open(os.path.join('save', 'losses'), 'wb') as f:
                pickle.dump(losses, f)
            with open(os.path.join('save', 'acc'), 'wb') as f:
                pickle.dump(accuracies, f)
            if dropbox:
                call(['cp', '-r', './save', os.path.join(cfg.home, 'Dropbox')])
    print('Finish training')


def main():
    train_dataset = VOCDataset(cfg.voc_root, [(2007, 'trainval'), (2012, 'trainval')])
    val_dataset = VOCDataset(cfg.voc_root, [(2007, 'test')])
    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=False)

    train(train_loader, val_loader, True, 0.0005, 1000, 0.0, 1, True)


if __name__ == '__main__':
    main()
