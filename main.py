from __future__ import print_function
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from prepare_data import prepare_data
from model import Softmax, TwoLayer, ConvNet


def main(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # decide which device to use; assumes at most one GPU is available
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    # decide if we're using a validation set
    # if not, don't evaluate every epoch
    evaluate = args.train_split < 1.

    # prep data laoders
    if args.train_split == 1:
        train_loader, _, test_loader = prepare_data(args)
    else:
        train_loader, val_loader, test_loader = prepare_data(args)

    # setup model
    if args.model == 'linear':
        model = Softmax().to(device)
    elif args.model == 'neuralnet':
        model = TwoLayer().to(device)
    else:
        model = ConvNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    for epoch in range(args.epochs):
        # training
        print('\n================== TRAINING ==================')
        model.train()
        correct = 0
        train_num = int(len(train_loader.dataset) * (1 - args.test_split) * args.train_split)
        for ix, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1] # get the index of the max logit
            correct += pred.eq(label.view_as(pred)).sum().item()
            if ix % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (ix + 1) * len(img), train_num,
                    100. * ix / len(train_loader), loss.item()))
        print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(correct, train_num, 100. * correct / train_num))

        # evaluation
        if evaluate:
            print('\n================== VALIDATION ==================')
            model.eval()
            val_loss = 0.
            val_correct = 0
            val_num = int(len(val_loader.dataset) * (1 - args.test_split) * (1 - args.train_split))
            with torch.no_grad():
                for img, label in val_loader:
                    img, label = img.to(device), label.to(device)
                    output = model(img)
                    val_loss += F.cross_entropy(output, label, size_average=False).item() # sum up batch loss
                    pred = output.max(1, keepdim=True)[1] # get the index of the max logit
                    val_correct += pred.eq(label.view_as(pred)).sum().item()

            val_loss /= val_num

            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                val_loss, val_correct, val_num,
                100. * val_correct / val_num))

    print('\n================== TESTING ==================')
    model.eval()
    test_loss = 0.
    test_correct = 0
    test_num = int(len(test_loader.dataset) * args.test_split)
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            test_loss += F.cross_entropy(output, label, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max logit
            test_correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= test_num
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, test_num,
        100. * test_correct / test_num))




if __name__=='__main__':
    # Training settings/hyperparams
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, choices=['linear', 'neuralnet', 'convnet'],
                        required=True, metavar='CHAR',
                        help='what kind of model to train (required)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--test-split', type=float, default=.2, metavar='P',
                        help='percent of training data to hold out for test set (default: .2)')
    parser.add_argument('--train-split', type=float, default=1., metavar='P',
                        help='percent of non-test data to use for training (default: 1.)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10; ignored if model is `linear`)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001; ignored if model is `linear`)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 200)')
    parser.add_argument('--data-folder', type=str, default='./data/', metavar='CHAR',
                        help='root path for folder containing MNIST data download (default: ./data/)')
    args = parser.parse_args()

    main(args)