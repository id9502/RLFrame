import torch
import numpy as np
import torch.nn as nn
from core.model.nets.nn_base import NN

# refer to: https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/policies.py
_LOG_2_PI = np.log(2. * np.pi)


def normal_log_density(x, mean, log_std):
    log_density = -(x - mean).pow(2) / (2 * torch.exp(log_std * 2)) - 0.5 * _LOG_2_PI - log_std
    return log_density.sum(1, keepdim=True)


class PolicyNet(NN):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -2
    is_disc_action = False

    def __init__(self, state_dim, action_dim, hidden_size=(256, 256), activation="relu"):
        super(PolicyNet, self).__init__(state_dim, action_dim)
        
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

        self.action_mean = nn.Linear(last_dim, int(np.prod(self.output_shape)))
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()

        self.action_log_std = nn.Linear(last_dim, int(np.prod(self.output_shape)))
        self.action_log_std.weight.data.mul_(0.1)
        self.action_log_std.bias.data.zero_()

    def forward(self, x):
        x = self.hidden_layers(x.view(x.size(0), -1))

        # mlp output the mean of this normal dist
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std(x)
        action_log_std = torch.clamp(action_log_std, PolicyNet.LOG_SIG_MIN, PolicyNet.LOG_SIG_MAX)

        return action_mean, action_log_std

    def select_action(self, x, deterministic=False):
        action_mean, action_log_std = self.forward(x)
        if deterministic:
            action = action_mean
        else:
            eps = torch.empty_like(action_mean).normal_()
            action = action_mean + action_log_std.exp() * eps
        return self._reshape_output(action)

    def get_kl(self, x):
        mean1, log_std1 = self.forward(x)
        std1 = torch.exp(log_std1)
        mean0 = mean1.detach().requires_grad_()
        log_std0 = log_std1.detach().requires_grad_()
        std0 = std1.detach().requires_grad_()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        # using the likelihood function of normal dist to cal the log likelihood
        action_mean, action_log_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std)

    def get_entropy(self, x):
        mean, log_std = self.forward(x)
        entropy = (0.5 + 0.5 * _LOG_2_PI + log_std).sum(-1, keepdim=True)
        return entropy

    def get_log_prob_entropy(self, x, actions):
        action_mean, action_log_std = self.forward(x)
        log_prob = normal_log_density(actions, action_mean, action_log_std)
        entropy = (0.5 + 0.5 * _LOG_2_PI + action_log_std).sum(-1, keepdim=True)
        return log_prob, entropy

