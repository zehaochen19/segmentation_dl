import argparse


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

print(args)
