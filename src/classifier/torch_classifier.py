import torch.nn as nn

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch

class Mlp(nn.Module):
    def __init__(self, inp_size, out_size, hidden_sizes=[128, ], nonlinearity=nn.ReLU):
        super().__init__()

        layer_sizes = hidden_sizes + [out_size]
        layers = [nn.Linear(inp_size, layer_sizes[0])]
        for i in range(1, len(layer_sizes)):
            layers.extend([
                nonlinearity(),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ConvNN(nn.Module):
    def __init__(self, numChannels, classes):
        super(ConvNN, self).__init__()
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = Linear(in_features=1250, out_features=700)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=700, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
    
    def forward(self, x):
        if x.shape[1] == 1: # if mnist
            x = torch.reshape(x, (-1, 1, 28, 28))
        elif x.shape[1] == 3: # if cifar
            x = torch.reshape(x, (-1, 3, 32, 32))
        else:
            raise ValueError("Invalid input shape")
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output
    