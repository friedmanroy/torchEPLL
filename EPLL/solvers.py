import numpy as np
import torch as tp
from tqdm import tqdm
from typing import Callable, Union

__prec = 1e-6


def conjgrad(A: Union[Callable, tp.Tensor], b: tp.Tensor, x0: tp.Tensor=None, precision: float=__prec,
             max_its: int=5000):
    """
    Basic implementation of the Conjugate Gradient method. The method solves the problem Ax=b using conjugate gradients,
    where A can either be given explicitly (i.e. is a torch tensor) or functionally (i.e. A is a callable function)
    :param A: a positive-definite matrix represented either as torch tensor or a callable function that receives a
              tensor and returns a tensor of the same shape
    :param b: a torch tensor
    :param x0: the initial guess for the solution - if not given, the initial guess is the 0 vector
    :param precision: the minimal precision for the exiting criterion
    :param max_its: maximal number of iterations to run the algorithm
    :return: the (maybe approximate) solution to Ax=b
    """
    dev = b.device
    exit_cond = lambda a: np.sqrt(tp.mean(a*a).cpu().numpy()) < precision
    if x0 is None: x0 = tp.zeros(b.shape)

    call = callable(A)
    x = x0.clone().to(dev)

    r = b - A(x) if call else b - A@x
    if exit_cond(r): return x

    p = r.clone().to(dev)

    for i in range(max_its):
        Ap = A(p) if call else A@p
        r_norm = tp.sum(r*r)

        alpha = r_norm/tp.sum(p*Ap)
        x = x + alpha*p.clone()
        r = r - alpha*p.clone()
        print(np.sqrt(tp.mean(r*r).cpu().numpy()))
        if exit_cond(r): return x

        p = r + tp.sum(r*r)*p/r_norm
    return x


def BiCGSTAB(A: Union[Callable, tp.Tensor], b: tp.Tensor, x0: tp.Tensor=None, precision: float=__prec,
             max_its: int=5000):
    """
    Implementation according to:
    https://utminers.utep.edu/xzeng/2017spring_math5330/MATH_5330_Computational_Methods_of_Linear_Algebra_files/ln07.pdf
    Solves the set of equations Ax=b
    :param A: if A is a torch tensor, then Ax=b is solved; otherwise A(x)=b is solved, where A() is assumed to be a
              linear operator
    :param b: the target value
    :param x0: the initial value that will be used for x in the BiCGSTAB algorithm
    :param precision: the minimal acceptable value for ||Ax-b||
    :param max_its: maximal number of iterations the algorithm should run
    :return: the value of x that solves the set of equations Ax=b
    """
    dev = b.device
    exit_cond = lambda a: np.sqrt(tp.sum(a*a).cpu().numpy()) < precision
    if x0 is None: x0 = tp.zeros(b.shape)

    call = callable(A)
    x = x0.clone().to(dev)

    r = b - A(x) if call else b - A@x
    r0 = r.clone().to(dev)
    p = r.clone().to(dev)

    if exit_cond(r): return x

    for i in range(max_its):
        Ap = A(p) if call else A@p
        alpha = tp.sum(r*r0)/tp.sum(r0*Ap)

        s = r - alpha*Ap
        if exit_cond(s): return x + alpha*p
        As = A(s) if call else A@s

        w = tp.sum(s*As)/tp.sum(As*As)
        x = x + alpha*p + w*s
        r_n = s - w*As
        if exit_cond(r_n): return x

        b = (alpha/w) * (tp.sum(r_n*r0))/tp.sum(r*r0)
        p = r + b*(p-w*Ap)

        r = r_n
        if np.abs(tp.sum(r*r0).item()) < precision:
            r0 = r.clone()
            p = r.clone()
    return x


def hamilMC(x0: tp.Tensor, loss_fcn: Callable, L: int=1000, dt: float=.01, scale: tp.Tensor=None,
            grad_scale: bool=False, verbose: bool=False):
    """
    Basic implementation of Hamiltonian Monte Carlo
    :param x0: initial sample position
    :param loss_fcn: loss function for x used to calculate gradients
    :param L: number of leap-frog iterations to run the algorithm
    :param dt: step size between different time points - the smaller this is the more accurate the algorithm will be,
               but will also need more iterations to find an uncorrelated sample
    :param scale: a scaling factor for different coordinates - if not given, the scaling is assumed to be 1 (no-scaling)
    :param grad_scale: whether to scale the coordinates according to the size of the gradient at x0
    :return: a sample from the PDF defined by the exponent of the negative potential energy whose gradient is given
    """
    x0 = x0.requires_grad_(True)
    with tp.enable_grad(): loss = loss_fcn(x0)
    if scale is None: scale = tp.abs(x0.grad) if grad_scale else tp.ones(x0.shape, device=x0.device)
    p = scale*tp.randn(x0.shape, device=x0.device)
    x = x0.clone().requires_grad_(True)
    with tp.enable_grad(): loss = loss_fcn(x)
    pbar = tqdm(range(L), disable=not verbose)
    for i in pbar:
        p = p - dt * x.grad / 2
        x = (x.data + dt * p / scale).requires_grad_(True)
        with tp.enable_grad(): loss = loss_fcn(x)
        pbar.set_postfix_str(f'loss: {loss.item():.2f}')
        p = p - dt * x.grad / 2
    return x
