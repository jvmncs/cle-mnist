from __future__ import print_function
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Softmax, TwoLayer, ConvNet
from prepare_data import KaggleMNIST

def main(args):
    # append extension to filename if needed
    try:
        assert args.submission_name[-4:] == '.csv'
    except AssertionError:
        args.submission_name += '.csv'

    # load necessaries from checkpoint
    check = torch.load(args.checkpoint)
    model_name = check['model']
    state_dict = check['state_dict']

    # enable cuda if available and desired
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # torch dataset and dataloader
    mnist_transform = transforms.Normalize((0.1307,), (0.3081,))
    dataset = KaggleMNIST(args.data_folder, train=False, transform=mnist_transform)
    loader = DataLoader(dataset, batch_size=len(dataset))

    # construct checkpoint's corresponding model type
    if model_name == 'linear':
        model = Softmax().to(device)
    elif model_name == 'neuralnet':
        model = TwoLayer().to(device)
    else:
        model = ConvNet().to(device)

    # load trained weights into model
    model.load_state_dict(state_dict)

    # make predictions with trained model on test data
    # loader has only one element (e.g. batch), so we don't need a loop
    # this won't work for large datasets; use a loop in those cases
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    logits = model(imgs)
    _, preds = torch.max(logits, dim=1) # returns max prob and argmax (e.g. corresponding class)
                                        # when dim is supplied

    # construct numpy array with two columns: ids, digit predictions
    # we'll use that array to create our text file using the np.savetxt function
    ids = (np.arange(len(preds)) + 1).reshape(-1, 1)
    preds = preds.view(-1,1).numpy()
    submission = np.concatenate((ids, preds), axis = 1)

    # writing submisison array to text file with proper formatting
    np.savetxt(args.data_folder + args.submission_name,
                submission,
                fmt='%1.1i',
                delimiter=',',
                header='ImageId,Label',
                comments='')


if __name__ == '__main__':
    # parses arguments when running from terminal/command line
    parser = argparse.ArgumentParser(description='PyTorch MNIST Kaggle Submission')
    parser.add_argument('--checkpoint', type=str, required=True, metavar='PATH',
                        help='model checkpoint to use to generate predictions')
    parser.add_argument('--submission-name', type=str, required=True, metavar='CHAR',
                        help='name of submission file; `.csv` extension will be appended if not supplied')
    parser.add_argument('--data-folder', type=str, default='./data/', metavar='PATH',
                        help='root path for folder containing MNIST data download (default: ./data/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    main(args)