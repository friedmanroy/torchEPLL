import numpy as np
import torch as tp
from typing import Callable, Union
from tqdm import tqdm
from .EPLL_utils import (grid_denoise, _tensor, _callable_beta, pad_im, trim_im, _choose_grids)
from models import Denoiser
from .solvers import BiCGSTAB
tp.set_grad_enabled(False)


def _default_sched(noise_var: float): return lambda i: min((2**i)/noise_var, 1e8)


def denoise(im: _tensor, noise_var: float, denoiser: Denoiser, p_sz: int, its: int=10,
            beta_sched: Union[float, Callable]=None, n_grids: int=16, resample_grids: bool=False, verbose: bool=True,
            low_mem: bool=False, pad: bool=True):
    """
    Denoise an image using the EPLL algorithm
    :param im: the image to denoise as a torch tensor
    :param noise_var: the variance of the noise, as a float
    :param denoiser: the denoiser function to use in order to denoise the patches; this should be a function that
                     receives as an input a torch tensor of shape [B, p_sz, p_sz(, 3)] as well as the noise variance,
                     such that the signature is denoiser(patches, noise_variance)
    :param p_sz: the patch size to use for denoising, as a single int (patches assumed to be square)
    :param its: number of iterations to run the algorithm for
    :param beta_sched: an update schedule for beta that will be used in the denoising process
    :param n_grids: number of grids to use for denoising
    :param resample_grids: whether to sample new grids every iteration or to use the same grids throughout
    :param verbose: whether a progressbar should be printed or not
    :param low_mem: if this is set to true, the denoising of each grid is carried out in batches (this is useful when
                    the image is very large)
    :param pad: whether to pad the image with reflection padding before denoising
    :return: the denoised image a torch tensor
    """
    if beta_sched is None: beta_sched = _default_sched(noise_var)
    beta_sched = _callable_beta(beta_sched)

    if pad: im = pad_im(im.clone(), 2*p_sz)
    dev = im.device

    if n_grids > p_sz**2: n_grids = p_sz**2
    grids = tp.ones(n_grids, *im.shape, device=dev)*im.clone()[None, ...]
    x = im.clone()

    x0, y0 = _choose_grids(p_sz, n_grids)

    pbar = tqdm(range(its), disable=not verbose)
    for i in pbar:
        b = beta_sched(i)
        pbar.set_postfix_str(f'beta: {b:.1f}')
        if resample_grids: x0, y0 = _choose_grids(p_sz, n_grids)

        for g in range(n_grids):
            grids[g] = grid_denoise(x, 1/b, x0[g], y0[g], denoiser, p_sz, low_mem=low_mem)
        # g = np.random.choice(n_grids, 1)[0]
        # grids[g] = grid_denoise(x.clone(), 1/b, x0[g], y0[g], denoiser, p_sz, low_mem=low_mem)
        x = (grids.sum(dim=0)*b + im.clone()/noise_var)/(b*n_grids + 1/noise_var)
    pbar.close()

    return trim_im(x, p_sz) if pad else x


def decorrupt(im: _tensor, noise_var: float, rest_loss: Callable, denoiser: Denoiser, p_sz: int, its: int=10,
              beta_sched: Union[float, Callable]=100., n_grids: int=16, resample_grids: bool=False, verbose: bool=True,
              low_mem: bool=False, pad: bool=True):
    beta_sched = _callable_beta(beta_sched)
    if pad: im = pad_im(im, 2*p_sz)
    dev = im.device
    grids = tp.ones(n_grids, *im.shape, device=dev)*im[None, ...]
    x = im.clone()

    x0, y0 = _choose_grids(p_sz, n_grids)

    pbar = tqdm(range(its), disable=not verbose)
    pbar.close()
    for i in pbar:
        b = beta_sched(i)
        pbar.set_postfix_str(f'beta: {b:.1f}')
        if resample_grids: x0, y0 = _choose_grids(p_sz, n_grids)

        for g in range(n_grids):
            grids[g] = grid_denoise(x, 1/b, x0[g], y0[g], denoiser, p_sz, low_mem=low_mem)

    return trim_im(x, 2*p_sz) if pad else x