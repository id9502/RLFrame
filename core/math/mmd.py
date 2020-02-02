import torch

# This file implements Maximum Mean Discrepancy: a discrepancy based on two-sample tests
# (http://jmlr.csail.mit.edu/papers/v13/gretton12a.html)

__all__ = ["mmd"]


def mmd(x, y, pre_calculated=None):
    """
    :param x: float tensor with shape B1 x Nd, can be back-propagated
    :param y: float tensor with shape B2 x Nd, fixed
    :param pre_calculated: if not None, cyy and diag_y in it will be used for speed up
    :return: mmd(x, y), size = 1
    """
    num_x = x.size(0)
    num_y = y.size(0)
    # notice detached tensor will share same data as original
    # std_x = x.std().detach().clamp_(min=1.e-5)
    # if "std_y" not in pre_cal_y:
    #     std_y = y.std().detach().clamp_(min=1.e-5)
    #     pre_cal_y["std_y"] = std_y
    # else:
    #     std_y = pre_cal_y["std_y"]

    std_x = 1.
    std_y = 1.

    # -- cyy -- #
    if pre_calculated is None or "cyy" not in pre_calculated:
        if num_y > 1:
            with torch.no_grad():
                yy = y @ y.t()
                diag_y = yy.diag()
                ry = diag_y.unsqueeze(0).expand_as(yy)
                cyy = ((-(ry.t() + ry - 2 * yy) / 2. / std_y ** 2).exp_().sum() - num_y) / float(num_y * (num_y - 1))
        else:
            cyy = 0.
            diag_y = y @ y.t()
        if isinstance(pre_calculated, dict):
            pre_calculated["cyy"] = cyy
            pre_calculated["diag_y"] = diag_y
    else:
        cyy = pre_calculated["cyy"]
        diag_y = pre_calculated["diag_y"]

    # -- cxx -- #
    if num_x > 1:
        xx = x @ x.t()
        diag_x = xx.diag()
        rx = diag_x.unsqueeze(0).expand_as(xx)
        cxx = ((-(rx.t() + rx - 2 * xx) / 2. / std_x ** 2).exp().sum() - num_x) / float(num_x * (num_x - 1))
    else:
        cxx = 0.
        diag_x = x @ x.t()

    # -- cxy -- #
    xy = x @ y.t()
    rx = diag_x.unsqueeze(1).expand_as(xy)
    ry = diag_y.unsqueeze(0).expand_as(xy)
    cxy = 2. * (-(rx + ry - 2 * xy) / 2. / std_x / std_y).exp().sum() / float(num_x * num_y)

    mmd = cyy + cxx - cxy
    return mmd


# def kernelized_set_distance(x, y):
#     """
#     :param x: batch of sample that will be used to calculate the distance to set
#     :param y: set
#     :param pre_calculated: if not None, cyy and diag_y in it will be used for speed up
#     :return:
#     """
#     num_y = y.size(0)
#
#     std_x = 1.
#     std_y = 10.
#
#     # -- cxy --
#     xy = x @ y.t()
#     rx = (x * x).sum(1).unsqueeze(1).expand_as(xy)
#     ry = (y * y).sum(1).unsqueeze(0).expand_as(xy)
#     cxy = 2. * (-(rx + ry - 2 * xy) / 2. / std_x / std_y).exp().sum(1) / float(num_y)
#
#     return -cxy

def kernelized_set_distance(x, y, pre_calculated=None):
    """
    :param x: batch of sample that will be used to calculate the distance to set
    :param y: set
    :param pre_calculated: if not None, cyy and diag_y in it will be used for speed up
    :return:
    """
    num_x = x.size(0)
    num_y = y.size(0)

    # if "std_x" in pre_calculated:
    #     std_xy = pre_calculated["std_xy"]
    #     std_y = pre_calculated["std_y"]
    # else:
    #     yy = y @ y.t()
    #     diag_y = yy.diag()
    #     ry = diag_y.unsqueeze(0).expand_as(yy)
    #     std_y = (ry.t() + ry - 2 * yy).mean()
    #
    #     xy = x @ y.t()
    #     rx = (x * x).sum(1).unsqueeze(1).expand_as(xy)
    #     ry = diag_y.unsqueeze(0).expand_as(xy)
    #     std_xy = (rx + ry - 2 * xy).mean()
    #
    #     pre_calculated["std_xy"] = std_xy
    #     pre_calculated["std_y"] = std_y

    std_y = std_xy = 10.
    # -- cxx -- #
    if num_x > 1:
        xx = x @ x.t()
        diag_x = xx.diag()
        rx = diag_x.unsqueeze(0).expand_as(xx)

        l2_x = rx.t() + rx - 2 * xx

        cxx = (- l2_x / std_xy).exp().sum(1) / float(num_x)# + (- l2_x / std_y).exp().sum(1) / float(num_x)
    else:
        cxx = 0.
        diag_x = x @ x.t()

    # -- cxy --
    xy = x @ y.t()
    rx = diag_x.unsqueeze(1).expand_as(xy)
    ry = (y * y).sum(1).unsqueeze(0).expand_as(xy)

    l2_xy = rx + ry - 2 * xy

    cxy = (- l2_xy / std_y).exp().sum(1) / float(num_y)# + (- l2_xy / std_xy).exp().sum(1) / float(num_y)

    mmd = cxx - cxy
    return mmd

