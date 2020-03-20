import torch
from core.model import Policy
from torch.optim.adam import Adam
from torch.nn.functional import mse_loss
from core.common import ParamDict


def bc_step(config: ParamDict, policy: Policy, demo):
    lr_init, lr_factor, l2_reg, bc_method, batch_sz, i_iter = \
        config.require("lr", "lr factor", "l2 reg", "bc method", "batch size", "current training iter")
    states, actions = demo

    # ---- annealing on learning rate ---- #
    lr = max(lr_init + lr_factor * i_iter, 1.e-8)

    optimizer = Adam(policy.policy_net.parameters(), weight_decay=l2_reg, lr=lr)

    # ---- define BC from demonstrations ---- #
    total_len = states.size(0)
    idx = torch.randperm(total_len, device=policy.device)
    err = 0.
    for i_b in range(int(total_len // batch_sz) + 1):
        idx_b = idx[i_b * batch_sz: (i_b+1) * batch_sz]
        s_b = states[idx_b]
        a_b = actions[idx_b]

        optimizer.zero_grad()
        a_mean_pred, a_logvar_pred = policy.policy_net(s_b)
        bc_loss = mse_loss(a_mean_pred + 0. * a_logvar_pred, a_b)
        err += bc_loss.item() * s_b.size(0) / total_len
        bc_loss.backward()
        optimizer.step()

    return err
