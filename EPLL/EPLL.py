import numpy as np
import torch as tp
from typing import Callable, Union
from tqdm import tqdm
from .EPLL_utils import (grid_denoise, _tensor, _callable_beta, pad_im, trim_im)
from .solvers import BiCGSTAB
tp.set_grad_enabled(False)


def _choose_grids(p_sz: int, n_grids: int):
    # choose random, different, grids
    chs = np.random.choice(p_sz**2, n_grids, replace=False)
    x0, y0 = chs//p_sz, chs%p_sz
    return [int(a) for a in x0], [int(a) for a in y0]


def denoise(im: _tensor, noise_var: float, denoiser: Callable, p_sz: int, its: int=10, beta_sched: Union[float, Callable]=100.,
            n_grids: int=16, resample_grids: bool=False, verbose: bool=True, low_mem: bool=False, pad: bool=True):
    beta_sched = _callable_beta(beta_sched)
    if pad: im = pad_im(im, 2*p_sz)
    dev = im.device
    grids = tp.ones(n_grids, *im.shape, device=dev)*im[None, ...]
    x = im.clone()

    x0, y0 = _choose_grids(p_sz, n_grids)

    pbar = tqdm(range(its), disable=not verbose)
    for i in pbar:
        b = beta_sched(i)
        pbar.set_postfix_str(f'beta: {b:.3f}')
        if resample_grids: x0, y0 = _choose_grids(p_sz, n_grids)

        for g in range(n_grids):
            grids[g] = grid_denoise(x, 1/b, x0[g], y0[g], denoiser, p_sz, low_mem=low_mem)
        x = (grids.sum(dim=0)*b + im.clone()/noise_var)/(b*n_grids + 1/noise_var)
    pbar.close()

    return trim_im(x, 2*p_sz) if pad else x


def decorrupt(im: _tensor, noise_var: float, denoiser: Callable, p_sz: int, its: int=10, beta_sched: Union[float, Callable]=100.,
            n_grids: int=16, resample_grids: bool=False, verbose: bool=True, low_mem: bool=False, pad: bool=True):
    beta_sched = _callable_beta(beta_sched)
    if pad: im = pad_im(im, 2*p_sz)
    dev = im.device
    grids = tp.ones(n_grids, *im.shape, device=dev)*im[None, ...]
    x = im.clone()

    x0, y0 = _choose_grids(p_sz, n_grids)

    pbar = tqdm(range(its), disable=not verbose)
    pbar.close()

    return trim_im(x, 2*p_sz) if pad else x