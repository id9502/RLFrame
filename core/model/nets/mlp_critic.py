import torch.nn as nn
from core.model.nets.nn_base import NN
import numpy as np


class ValueNet(NN):
    def __init__(self, input_shape, output_shape=(1,), hidden_size=(256, 256), activation="relu"):
        super(ValueNet, self).__init__(input_shape, output_shape)

        hl = []
        last_dim = int(np.prod(self.input_shape))
        for nh in hidden_size:
            hl.append(nn.Linear(last_dim, nh))

            if activation == "tanh":
                hl.append(nn.Tanh())
            elif activation == "relu":
                hl.append(nn.ReLU())
            elif activation == "sigmoid":
                hl.append(nn.Sigmoid())
            last_dim = nh

        self.hidden_layers = nn.Sequential(*hl)

        self.value_head = nn.Linear(last_dim, int(np.prod(self.output_shape)))
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.zero_()

    def forward(self, x):
        x = self.hidden_layers(x.view(x.size(0), -1))
        value = self.value_head(x)
        return self._reshape_output(value)

