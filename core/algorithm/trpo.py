import torch
from core.math import cg
from torch.optim.adam import Adam
from torch.optim.lbfgs import LBFGS
from core.model.policy_with_value import PolicyWithValue
from core.common import SampleBatch, ParamDict


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        model.set_flat_params(x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


# def update_value_net_new(value_net, states, returns, l2_reg, mini_batch_sz=512, iter=10):
#     optim = Adam(value_net.parameters(), weight_decay=l2_reg)
#
#     inds = torch.arange(returns.size(0))
#
#     for i in range(iter):
#         np.random.shuffle(inds)
#         for i_b in range(int(np.ceil(returns.size(0) / mini_batch_sz))):
#             b_ind = inds[i_b*mini_batch_sz: min((i_b+1)*mini_batch_sz, inds.size(0))]
#
#             value_loss = (value_net(states[b_ind]) - returns[b_ind]).pow(2).mean()
#             optim.zero_grad()
#             value_loss.backward()
#             optim.step()


def get_tensor(batch, device):
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    return states, actions, advantages, returns


def update_value_net(value_net, states, returns, l2_reg):
    optimizer = LBFGS(value_net.parameters(), max_iter=25, history_size=5)

    def closure():
        optimizer.zero_grad()
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss = value_loss + param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss

    optimizer.step(closure)


def trpo_step(config: ParamDict, batch: SampleBatch, policy: PolicyWithValue):
    max_kl, damping, l2_reg = config.require("max kl", "damping", "l2 reg")
    states, actions, advantages, returns = get_tensor(batch, policy.device)

    """update critic"""
    update_value_net(policy.value_net, states, returns, l2_reg)

    """update policy"""
    with torch.no_grad():
        fixed_log_probs = policy.policy_net.get_log_prob(states, actions).detach()
    """define the loss function for TRPO"""

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                log_probs = policy.policy_net.get_log_prob(states, actions)
        else:
            log_probs = policy.policy_net.get_log_prob(states, actions)
        action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
        return action_loss.mean()

    """define Hessian*vector for KL"""

    def Fvp(v):
        kl = policy.policy_net.get_kl(states).mean()

        grads = torch.autograd.grad(kl, policy.policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy.policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    for _ in range(2):
        loss = get_loss()
        grads = torch.autograd.grad(loss, policy.policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = cg(Fvp, -loss_grad, nsteps=10)

        shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
        lm = (max_kl / shs).sqrt_()
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)

        prev_params = policy.policy_net.get_flat_params().detach()
        success, new_params = line_search(policy.policy_net, get_loss, prev_params, fullstep, expected_improve)
        policy.policy_net.set_flat_params(new_params)
        if not success:
            return False
    return True
