import torch

# This file implements Conjugate Gradients linear algebra solver, which solves:
#       min_x f(x) = 0.5 * x.T @ A @ x - b.T @ x
# by finding
#       min_x ||A @ x - b||_2
__all__ = ["cg"]


def cg(Avp_f, b, x0=None, nsteps=None, rdotr_tol=1e-10):
    """
    :param Avp_f: function handler f(x) that calculates A @ x on given x
    :param b: see description above, same shape with x
    :param x0: initial point of solver, default is 0
    :param nsteps: maximum try steps, default equals to dim b
    :param rdotr_tol: maximum rhs for terminate loop
    :return: x
    """
    nsteps = b.size(0) * 5 if nsteps is None else nsteps
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone().detach()
        p = b.clone().detach()
    else:
        x = x0
        r = (b - Avp_f(x)).detach()
        p = r.clone().detach()

    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        if rdotr < rdotr_tol:
            break
        Avp = Avp_f(p).detach()
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x


if __name__ == "__main__":
    A = torch.randn(5, 5)
    A = A @ A.transpose(0, 1)
    x = torch.randn(5)

    b = A @ x

    def fvp(v):
        return A @ v
    xe = cg(fvp, b)
    print((x-xe).pow(2).sum())
