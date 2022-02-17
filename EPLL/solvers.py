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

    :param A:
    :param b:
    :param x0:
    :param precision:
    :param max_its:
    :return:
    """
    dev = b.device
    exit_cond = lambda a: np.sqrt(tp.mean(a*a).cpu().numpy()) < precision
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


def hamilMC(x0: tp.Tensor, grad: Callable, L: int=1000, dt: float=.01, scale: tp.Tensor=None, grad_scale: bool=False):
    """
    Basic implementation of Hamiltonian Monte Carlo
    :param x0: initial sample position
    :param grad: a function that receives as an input a vector x and returns the gradient with respect to the
                 potential energy at the point x (which will have the same shape as x)
    :param L: number of leap-frog iterations to run the algorithm
    :param dt: step size between different time points - the smaller this is the more accurate the algorithm will be,
               but will also need more iterations to find an uncorrelated sample
    :param scale: a scaling factor for different coordinates - if not given, the scaling is assumed to be 1 (no-scaling)
    :param grad_scale: whether to scale the coordinates according to the size of the gradient at x0
    :return: a sample from the PDF defined by the exponent of the negative potential energy whose gradient is given
    """
    if scale is None: scale = tp.abs(grad(x0)) if grad_scale else tp.ones(x0.shape, device=x0.device)
    p = scale*tp.randn(x0.shape, device=x0.device)
    x = x0.clone()
    for i in range(L):
        p = p - dt * grad(x) / 2
        x = x + dt * p / scale
        p = p - dt * grad(x) / 2
    return x


def newton_optimize(x0: float, function: Callable, delta: float=.1, precision: float=__prec,
                    max_its: int=100, return_vals: bool=True, verbose: bool=False,
                    min_val: float=None, max_val: float=None):

    xs = [x0]
    fxs = []
    pbar = tqdm(range(max_its), disable=not verbose)
    for i in pbar:
        pbar.set_postfix_str(f'x={xs[-1]:.2f}')
        fxs.append(function(xs[-1]))
        if i > 1 and np.abs(xs[-1]-xs[-2]) <= precision: break
        fx_p, fx_m = function(xs[-1]+delta), function(xs[-1]-delta)
        dfx_p = (fx_p - fxs[-1])/delta
        dfx_m = (fxs[-1] - fx_m)/delta
        dfx = .5*dfx_p + .5*dfx_m
        ddfx = (fx_p + fx_m - 2*fxs[-1])/(delta**2)
        new = xs[-1] - dfx/ddfx
        if min_val is not None and new < min_val: new = min_val
        if max_val is not None and new > max_val: new = max_val
        xs.append(new)
    fxs.append(function(xs[-1]))
    if return_vals: return xs[-1], xs, fxs
    return xs[-1]


def bisection_optimize(x0: float, function: Callable, delta: float=.1, precision: float=__prec,
                       max_its: int=100, return_vals: bool=True, verbose: bool=False,
                       min_val: float=0, max_val: float=1e6):
    """

    :param x0:
    :param function:
    :param delta:
    :param precision:
    :param max_its:
    :param return_vals:
    :param verbose:
    :param min_val:
    :param max_val:
    :return:
    """
    xs = [x0]
    upper = max_val
    lower = min_val
    fxs = []
    others = []
    pbar = tqdm(range(max_its), disable=not verbose)
    dfx = 0
    for i in pbar:
        if i > 1 and np.abs(xs[-1]-xs[-2]) <= precision: break
        pbar.set_postfix_str(f'x={xs[-1]:.2f}; fx={fxs[-1] if len(fxs) > 0  else 0:.2f}; '
                             f'dfx={dfx:.2f}')

        # save statistics
        outp = function(xs[-1])
        if type(outp) == tuple:
            fxs.append(outp[0])
            if len(outp) > 1: others.append(outp[1])
        else:
            fxs.append(outp)

        # approximate derivative
        dfx = (function(xs[-1]+delta)[0] - fxs[-1])/delta

    # update lower and upper bounds
        if dfx < 0:
            new = .5*lower + .5*xs[-1]
            upper = xs[-1]
        else:
            new = .5*upper + .5*xs[-1]
            lower = xs[-1]
        xs.append(new)

    if return_vals:
        outp = function(xs[-1])
        if type(outp) == tuple:
            fxs.append(outp[0])
            if len(outp) > 1: others.append(outp[1])
        else:
            fxs.append(outp)
        return xs[-1], xs, fxs, others
    return xs[-1]


def poisson_deblock(im: tp.Tensor, block_size: int=8, alpha=.1) -> tp.Tensor:
    from scipy.ndimage import convolve
    shp = im.shape[:-1]
    d_kern = np.array([[1, -1], [1, -1], [1, -1]])
    dd_kern = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
    lap = lambda x: tp.from_numpy(convolve(x.cpu().numpy().reshape(shp), dd_kern).flatten()).float() + \
                    tp.from_numpy(convolve(x.cpu().numpy().reshape(shp), dd_kern.T).flatten()).float() + \
                    alpha*x
    DC = tp.mean(im, dim=(0, 1))
    sol = tp.zeros(*im.shape)
    for i in range(3):
        grad_x = convolve(im[:, :, i].cpu().numpy(), d_kern)
        grad_y = convolve(im[:, :, i].cpu().numpy(), d_kern.T)

        grad_x[:, ::block_size] = 0
        grad_y[::block_size] = 0

        div = tp.from_numpy(convolve(grad_x, d_kern) + convolve(grad_y, d_kern.T)).float().flatten()
        chan_sol = BiCGSTAB(lap, div, precision=1e-5)
        sol[:, :, i] = chan_sol.reshape(shp) + DC[i].cpu()
    return sol.to(device=im.device)
