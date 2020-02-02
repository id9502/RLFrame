import torch
import numpy as np
from torch.optim.adam import Adam
from core.model.policy_with_value import PolicyWithValue
from core.common.types import StepDictList
from core.common.config import ParamDict


def get_tensor(policy, reply_memory, device):
    policy.estimate_advantage(reply_memory)
    states = []
    actions = []
    advantages = []
    returns = []

    for b in reply_memory:
        advantages.extend([[tr["advantage"]] for tr in b["trajectory"]])
        returns.extend([[tr["return"]] for tr in b["trajectory"]])
        states.extend([tr['s'] for tr in b["trajectory"]])
        actions.extend([tr['a'] for tr in b["trajectory"]])

    states = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    return states, actions, advantages, returns


def update_value_net(value_net, states, returns, l2_reg, mini_batch_sz=512, iter=10):
    optim = Adam(value_net.parameters(), weight_decay=l2_reg)

    inds = torch.arange(returns.size(0))

    for i in range(iter):
        np.random.shuffle(inds)
        for i_b in range(returns.size(0) // mini_batch_sz):
            b_ind = inds[i_b*mini_batch_sz: min((i_b+1)*mini_batch_sz, inds.size(0))]

            value_loss = (value_net(states[b_ind]) - returns[b_ind]).pow(2).mean()
            optim.zero_grad()
            value_loss.backward()
            optim.step()


def a2c_step(config: ParamDict, policy: PolicyWithValue, replay_memory: StepDictList):
    config.require("l2 reg")
    l2_reg = config["l2 reg"]

    policy.estimate_advantage(replay_memory)
    states, actions, returns, advantages = get_tensor(policy, replay_memory, policy.device)

    """update critic"""
    update_value_net(policy.value_net, returns, states, l2_reg)

    """update policy"""
    optimizer_policy = Adam(policy.policy_net.parameters(), lr=1.e-3, weight_decay=l2_reg)
    log_probs = policy.policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()
