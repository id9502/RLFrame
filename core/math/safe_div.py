import torch


def safe_div(a, b, f=1e7):
    if b == 0:
        r = a * f
    else:
        r = a / b
    if isinstance(r, float):
        return max(min(r, f), -f)
    else:
        return torch.clamp(r, -f, f)
