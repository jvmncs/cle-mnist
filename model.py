import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    """Simple linear classifier using cross entropy loss"""
    def __init__(self):
        super().__init__()
        self.logits = nn.Linear(784, 10)

    def forward(self, x):
        return self.logits(x)

class TwoLayer(nn.Module):
    """Feedforward neural network with one hidden layer"""
    def __init__(self, hidden=800, dropout=.4):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        x = nn.functional.relu(self.hidden(x))
        if self.dropout:
            x = self.dropout(x)
        return self.out(x)

class ConvNet(nn.Module):
    """
    ConvNet from TensorFlow CNN MNIST tutorial
    (see: https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier)
    """
    def __init__(self):
        super().__init__()
        self.block1 = self.conv_block(1, 32)
        self.block2 = self.conv_block(32, 64)
        self.hidden = nn.Linear(4 * 4 * 64, 1024)
        self.dropout = nn.Dropout(.4)
        self.out = nn.Linear(1024, 10)

    @classmethod
    def conv_block(cls, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )

    @classmethod
    def flatten(cls, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)

    def forward(self, x):
        # make each image 3d for compatibility with convolutional blocks
        x = x.view(-1, 1, 28, 28)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x) # flatten output of convolutional section
        x = self.dropout(self.hidden(x))
        return self.out(x)
