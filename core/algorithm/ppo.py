import torch
from torch.optim.adam import Adam
from core.model.policy_with_value import PolicyWithValue
from core.common import SampleBatch, ParamDict
import torch.nn.functional as F


def get_tensor(batch, device):
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    return states, actions, advantages, returns


def ppo_step(config: ParamDict, batch: SampleBatch, policy: PolicyWithValue):
    lr, l2_reg, clip_epsilon, policy_iter, i_iter, max_iter, mini_batch_sz = \
        config.require("lr", "l2 reg", "clip eps", "optimize policy epochs",
                       "current training iter", "max iter", "optimize batch size")
    lam_entropy = 0.
    states, actions, advantages, returns = get_tensor(batch, policy.device)

    lr_mult = max(1.0 - i_iter / max_iter, 0.)
    clip_epsilon = clip_epsilon * lr_mult

    optimizer_policy = Adam(policy.policy_net.parameters(), lr=lr * lr_mult, weight_decay=l2_reg)
    optimizer_value = Adam(policy.value_net.parameters(), lr=lr * lr_mult, weight_decay=l2_reg)

    with torch.no_grad():
        fixed_log_probs = policy.policy_net.get_log_prob(states, actions).detach()

    for _ in range(policy_iter):
        inds = torch.randperm(states.size(0))

        """perform mini-batch PPO update"""
        for i_b in range(inds.size(0) // mini_batch_sz):
            slc = slice(i_b * mini_batch_sz, (i_b+1) * mini_batch_sz)

            states_i = states[slc]
            actions_i = actions[slc]
            returns_i = returns[slc]
            advantages_i = advantages[slc]
            log_probs_i = fixed_log_probs[slc]

            """update critic"""
            for _ in range(1):
                value_loss = F.mse_loss(policy.value_net(states_i), returns_i)
                optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.value_net.parameters(), 0.5)
                optimizer_value.step()

            """update policy"""
            log_probs, entropy = policy.policy_net.get_log_prob_entropy(states_i, actions_i)
            ratio = torch.exp(log_probs - log_probs_i)
            surr1 = ratio * advantages_i
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_i
            policy_surr = -torch.min(surr1, surr2).mean() - entropy.mean() * lam_entropy
            optimizer_policy.zero_grad()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm_(policy.policy_net.parameters(), 0.5)
            optimizer_policy.step()
