import torch
import torch.nn as nn


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
