#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def conv(ni, nf, ks=3, stride=1, padding=1, **kwargs):
    """
    Thin wrapper around nn.Conv2d to ensure kaiming initialization. The rest of the api
    remains the same
    """
    _conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=False, **kwargs)
    nn.init.kaiming_normal_(_conv.weight)
    return _conv


def block(ni, nf, **kwargs):
    """
    Block function to play around with convolutions, activations and batchnorms.
    Each part can be changed easily here and then used in the main function get_model()
    Downsampling is done using stride 2

    Args:
        ni: number of input channels to conv layer
        nf: number of output channels to conv layer

    Returns:
        A conv block

    """
    _conv = conv(ni, nf, ks=3, stride=2, padding=1, **kwargs)
    return nn.Sequential(_conv, nn.BatchNorm2d(nf), nn.ReLU())


def get_model():
    """
    Create the main model using blocks, AdaptiveAvgPool and a Linear layer
    """
    return nn.Sequential(
        block(3, 32),
        block(32, 64),
        block(64, 128),
        block(128, 256),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 8))


def get_dataloaders(path: Path, img_size: int, batch_size: int, num_workers: int):
    """
    Get train and test dataloaders from the dataset

    Args:
        path: path to dataset
        img_size: size of image to be used fo training (bilinear interpolation is used)
        batch:size: batch size for dataloaders
        num_workers: workers to be used for loading data

    Returns:
        train and test dataloaders
    """

    # configure data transforms for trainig and testing
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4273, 0.4523, 0.4497],
                             std=[0.4273, 0.4523, 0.4497])])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4273, 0.4523, 0.4497],
                             std=[0.4273, 0.4523, 0.4497])])

    # prepare datasets
    train_data = datasets.ImageFolder(path/'train', transform=train_transform)
    test_data = datasets.ImageFolder(path/'test', transform=test_transform)

    # prepare dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def plot_hist(train_stat, test_stat, stat_name='accuracy', baseline=None, xmax=20, location='lower right'):
    """
    create matplotlib figures for training statistics
    """
    plt.plot(train_stat)
    plt.plot(test_stat)
    if baseline is not None:
        plt.hlines(baseline, 0, xmax, 'g')
    plt.title(f"Model {stat_name}")
    plt.ylabel(f"{stat_name}")
    plt.xlabel('epoch')
    if baseline is not None:
        plt.legend(['train', 'validation', 'baseline - keras'], loc=location)
    else:
        plt.legend(['train', 'validation'], loc=location)
    plt.savefig(f"{stat_name}.png")
    plt.close()


def parse_args(args=sys.argv[1:]):
    """
    Utility function for parsing command line arguments
    """
    parser = argparse.ArgumentParser(description='A simple script for training an image classifier')
    parser.add_argument('--exp_name', type=str, default='baseline',
                        help='name of experiment - for saving model and tensorboard dir')
    parser.add_argument(
        "--data_path", default="/home/adityassrana/datatmp/Datasets/MIT_split", help="path to Dataset")
    parser.add_argument("--max_epochs", type=int, default=5,
                        help="number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="base learning rate to use for training")
    parser.add_argument("--image_size", type=int, default=64, help="image size for training")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for loading data")
    parser.add_argument("--save_model", action="store_true",
                        help="to save the model at the end of each epoch")
    parser.add_argument("--tb", action="store_true", help="to write results to tensorboard")
    parser.add_argument("--plot_stats", action="store_true",
                        help="to save matplotlib plots of train-test loss and accuracy")
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':

    # parse command line arguments
    args = parse_args()

    print(args)

    # check for CUDA availabilitu
    if torch.cuda.is_available():
        print('CUDA is available, setting device to CUDA')
    # set device to  CUDA for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloaders
    train_loader, test_loader = get_dataloaders(
        Path(args.data_path), args.image_size, args.batch_size, args.num_workers)
    print('Dataloaders ready')

    # get training model and plot summary
    model = get_model()
    #summary(model, (3, args.image_size, args.image_size), device='cpu')
    # send model to GPU
    model.to(device)

    # get loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup Tensorboard
    # We're uisng two writers to visualiz train and test results together
    if args.tb:
        writer_train = SummaryWriter(f'tb/{args.exp_name}/train')
        writer_test = SummaryWriter(f'tb/{args.exp_name}/test')

    # histograms
    train_acc_hist = []
    test_acc_hist = []
    train_loss_hist = []
    test_loss_hist = []

    # Training and Testing Loop
    for epoch in range(args.max_epochs):
        model.train()

        # training statistics
        losses, acc, count = [], [], []
        for batch_idx, (xb, yb) in enumerate((train_loader)):
            # transfer data to GPU
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # calculating this way to account for the fact that the
            # last batch may have different batch size
            bs = xb.shape[0]
            # get number of right predictions
            correct_predictions = (preds.argmax(dim=1) == yb).float().sum()
            # add to list
            losses.append(bs*loss.item()), count.append(bs), acc.append(correct_predictions)

            # tensorboard
            if args.tb:
                writer_train.add_scalar('per_batch/train_loss', loss.item(),
                                        epoch*len(train_loader) + batch_idx)

        # accumulate/average statistics
        n = sum(count)
        train_loss_epoch = sum(losses)/n
        train_acc_epoch = sum(acc)/n

        train_loss_hist.append(train_loss_epoch)
        train_acc_hist.append(train_acc_epoch)

        if args.tb:
            # write to tensorboard
            writer_train.add_scalar('per_epoch/losses', train_loss_epoch, epoch)
            writer_train.add_scalar('per_epoch/accuracy', train_acc_epoch, epoch)

        model.eval()
        with torch.no_grad():
            losses, acc, count = [], [], []
            for batch_idx, (xb, yb) in enumerate((test_loader)):
                # transfer data to GPU
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                bs = xb.shape[0]
                # get number of right predictions
                correct_predictions = (preds.argmax(dim=1) == yb).float().sum()
                # add to list
                losses.append(bs*loss.item()), count.append(bs), acc.append(correct_predictions)

                if args.tb:
                    writer_test.add_scalar('per_batch/test_loss', loss.item(),
                                           epoch*len(test_loader) + batch_idx)

        # accumulate/average statistics
        n = sum(count)
        test_loss_epoch = sum(losses)/n
        test_acc_epoch = sum(acc)/n

        test_loss_hist.append(test_loss_epoch)
        test_acc_hist.append(test_acc_epoch)

        if args.tb:
            # write to tensorboard
            writer_test.add_scalar('per_epoch/losses', test_loss_epoch, epoch)
            writer_test.add_scalar('per_epoch/accuracy', test_acc_epoch, epoch)

        print(f"Epoch{epoch}, train_accuracy:{train_acc_epoch:.4f}, test_accuracy:{test_acc_epoch:.4f}, train_loss:{train_loss_epoch:.4f}, test_loss:{test_loss_epoch:.4f}")

        if args.save_model:
            torch.save(model.state_dict(), f"{args.exp_name}_epoch{epoch}_acc{train_acc_epoch:.4f}")
            print("Model saved")

    print("Finished training")

    if args.plot_stats:
        plot_hist(train_acc_hist, test_acc_hist, 'accuracy',
                  xmax=args.max_epochs, location='lower right')
        plot_hist(train_loss_hist, test_loss_hist, 'loss',
                  xmax=args.max_epochs, location='upper right')
        print("Finished plotting")
