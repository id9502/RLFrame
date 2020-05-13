import torch


__all__ = ["log_sum_exp", "log_sum_exp_elem"]


def log_sum_exp(tensor: torch.Tensor, dim: int, keepdim: bool = False):
    bias = tensor.max(dim=dim, keepdim=True)[0].detach()
    ans = bias + (tensor - bias).exp().sum(dim=dim, keepdim=True).log()
    if keepdim:
        return ans
    else:
        return ans.squeeze(dim=dim)


def log_sum_exp_elem(*a):
    """
    :param a: elements
    :return: (a[0].exp() + a[1].exp() + ...).log()
    """
    bias = max(a).detach()
    ans = bias + sum([(ai-bias).exp() for ai in a]).log()
    return ans
