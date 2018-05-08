import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# Could've used a TensorDataset here, but made this for demonstrative purposes
class KaggleMNIST(Dataset):
    """Kaggle version of MNIST dataset"""
    def __init__(self, root, train=True, transform=None):
        file = 'train.npy' if train else 'test.npy'
        path = os.path.join(root, file)
        self._train = train
        self._data = np.load(path)
        self._transform = transform
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, ix):
        img = torch.from_numpy(self._data[ix, 1:]).type(torch.float32).view(1, 28, 28)
        label = torch.tensor(self._data[ix, 0], dtype=torch.int64) if self._train else None
        if self._transform is not None:
            img = self._transform(img)
        return img.view(28 * 28), label


def prepare_data(args):
    """Prepare Kaggle version of MNIST dataset with optional validation split"""
    process_csv(args.data_folder)

    mnist_transform = transforms.Normalize((0.1307,), (0.3081,))

    train_set = KaggleMNIST(args.data_folder, train=True, transform=mnist_transform)
    val_set = KaggleMNIST(args.data_folder, train=True, transform=mnist_transform)
    test_set = KaggleMNIST(args.data_folder, train=True, transform=mnist_transform)
    
    num_train = len(train_set)
    indices = np.arange(num_train)
    mask = np.random.sample(num_train) < args.test_split
    other_ix = indices[~mask]
    other_mask = np.random.sample(np.sum(~mask)) < args.train_split

    train_ix = other_ix[other_mask]
    test_ix = indices[mask]

    if not np.all(other_mask):
        val_ix = other_ix[~other_mask]
    else:
        val_ix = None

    train_sampler = SubsetRandomSampler(train_ix)
    val_sampler = SubsetRandomSampler(val_ix) if val_ix is not None else None
    test_sampler = SubsetRandomSampler(test_ix)

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.use_cuda else {}
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
        sampler=train_sampler, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size,
        sampler=val_sampler, **kwargs) if val_ix is not None else None
    test_loader = DataLoader(train_set, batch_size=args.test_batch_size,
        sampler=test_sampler, **kwargs)

    return train_loader, val_loader, test_loader

def process_csv(root='./data/'):
    for pickle_file in ['train.npy', 'test.npy']:
        if os.path.isfile(root + pickle_file):
            continue
        else:
            loaded = np.loadtxt(root + pickle_file[:-4] + '.csv', delimiter=',', skiprows=1)
            np.save(root + pickle_file, loaded)
