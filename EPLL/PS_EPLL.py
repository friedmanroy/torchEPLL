import numpy as np
import torch as tp
from typing import Callable, Union
from tqdm import tqdm
from .EPLL_utils import (grid_denoise, _tensor, _callable_beta, pad_im, trim_im, _choose_grids, _default_sched,
                         optimize_function)
from models import Denoiser
from .solvers import BiCGSTAB
tp.set_grad_enabled(False)


def sample_denoise(im: _tensor, noise_var: float, denoiser: Denoiser, p_sz: int, its: int=10,
                   beta_sched: Union[float, Callable]=None, n_grids: int=16, resample_grids: bool=False, verbose: bool=True,
                   low_mem: bool=False, pad: bool=True, init: _tensor=None):
    """
    Sample a denoising using the EPLL prior
    :param im: the image to denoise as a torch tensor
    :param noise_var: the variance of the noise, as a float
    :param denoiser: the denoiser function to use in order to denoise the patches; this should be a function that
                     receives as an input a torch tensor of shape [B, p_sz, p_sz(, 3)] as well as the noise variance,
                     such that the signature is denoiser(patches, noise_variance)
    :param p_sz: the patch size to use for denoising, as a single int (patches assumed to be square)
    :param its: number of iterations to use
    :param beta_sched: an update schedule for beta that will be used in the denoising process
    :param n_grids: number of grids to use for denoising
    :param resample_grids: whether to sample new grids every iteration or to use the same grids throughout
    :param verbose: whether a progressbar should be printed or not
    :param low_mem: if this is set to true, the denoising of each grid is carried out in batches (this is useful when
                    the image is very large)
    :param pad: whether to pad the image with reflection padding before denoising
    :param init: initialization point for the algorithm; if none supplied, starts with noisy image
    :return: the denoised image a torch tensor
    """
    # make sure that there is a schedule for beta and that it is a callable
    if beta_sched is None: beta_sched = _default_sched(noise_var)
    beta_sched = _callable_beta(beta_sched)

    # rescale noise variance to take into account number of grids
    noise_var = noise_var/n_grids

    # if the image is to be padded, do so
    if pad: im = pad_im(im.clone(), 2*p_sz)
    dev = im.device

    # define the grids that will be used
    if n_grids > p_sz**2: n_grids = p_sz**2
    if init is None:
        grids = tp.ones(n_grids, *im.shape, device=dev)*im.clone()[None, ...]
        x = im.clone()
    else:
        grids = tp.ones(n_grids, *im.shape, device=dev)*init.clone()[None, ...]
        x = init.clone()

    x0, y0 = _choose_grids(p_sz, n_grids)

    pbar = tqdm(range(its), disable=not verbose)
    for i in pbar:
        b = beta_sched(i)
        pbar.set_postfix_str(f'beta: {b:.1f}')
        if resample_grids: x0, y0 = _choose_grids(p_sz, n_grids)

        for g in range(n_grids):
            grids[g] = grid_denoise(x, 1/b, x0[g], y0[g], denoiser, p_sz)

        x = (grids.sum(dim=0)*b + im.clone()/noise_var)/(b*n_grids + 1/noise_var)
        if i < its-1: x += tp.randn(*x.shape, device=x.device)/np.sqrt(b*n_grids + 1/noise_var)
    pbar.close()

    return trim_im(x, p_sz) if pad else x


def sample_decorrupt(im: _tensor, noise_var: float, H: Callable, denoiser: Denoiser, p_sz: int, its: int=10,
                     beta_sched: Union[float, Callable]=100., n_grids: int=16, resample_grids: bool=False, verbose: bool=True,
                     pad: bool=True, opt_its: int=500, optimizer: str='adam', lr: float=1e-2, init: _tensor=None):
    # make sure that there is a schedule for beta and that it is a callable
    if beta_sched is None: beta_sched = _default_sched(noise_var)
    beta_sched = _callable_beta(beta_sched)
    noise_var = noise_var/n_grids

    # if the image is to be padded, do so
    if pad: im = pad_im(im.clone(), 2*p_sz)
    dev = im.device

    # define the grids that will be used
    if n_grids > p_sz**2: n_grids = p_sz**2
    if init is None: init = im.clone()
    grids = tp.ones(n_grids, *init.shape, device=dev)*init.clone()[None, ...]
    x = init.clone()

    x0, y0 = _choose_grids(p_sz, n_grids)

    def update(x: _tensor, grids: _tensor, beta: float):
        eps1, eps2 = tp.randn(*im.shape, device=im.device), tp.randn(*x.shape, device=x.device)
        s, G = np.sqrt(noise_var), np.sqrt(beta*n_grids)
        eps1, eps2 = eps1*s, eps2/G
        loss_func = lambda x: tp.sum((H(x)-im+eps1)*(H(x)-im+eps1))/noise_var + \
                              beta*tp.sum((grids-x[None]+eps2[None])**2)
        return optimize_function(loss_func, x, its=opt_its, lr=lr, optimizer=optimizer)

    pbar = tqdm(range(its), disable=not verbose)
    for i in pbar:
        b = beta_sched(i)
        pbar.set_postfix_str(f'beta: {b:.1f}')
        if resample_grids: x0, y0 = _choose_grids(p_sz, n_grids)

        for g in range(n_grids):
            grids[g] = grid_denoise(x, 1/b, x0[g], y0[g], denoiser, p_sz)
        x = update(x, grids, b)
    return trim_im(x, p_sz) if pad else x