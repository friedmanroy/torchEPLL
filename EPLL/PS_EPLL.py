import numpy as np
import torch as tp
from typing import Callable, Union
from tqdm import tqdm
from .EPLL_utils import (grid_denoise, _tensor, _callable_beta, pad_im, trim_im, _choose_grids, _default_sched,
                         optimize_function)
from models import Denoiser
from .solvers import BiCGSTAB
tp.set_grad_enabled(False)


def sample_denoise(im: _tensor, noise_var: Union[float, _tensor], denoiser: Denoiser, p_sz: int, its: int=10,
                   beta_sched: Union[float, Callable]=None, n_grids: int=16, resample_grids: bool=True,
                   verbose: bool=True, low_mem: bool=False, pad: bool=True, init: _tensor=None):
    """
    Sample a denoising using the EPLL prior
    :param im: the image to denoise as a torch tensor
    :param noise_var: the variance of the noise, as a float or a torch tensor (in which case it should be the same shape
                      as the image)
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
    if pad:
        im = pad_im(im.clone(), 2*p_sz)
        if isinstance(noise_var, _tensor): noise_var = pad_im(noise_var, 2*p_sz)
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
        if i < its-1:
            std = 1/tp.sqrt(b*n_grids + 1/noise_var) if isinstance(noise_var, _tensor) \
                else 1/np.sqrt(b*n_grids + 1/noise_var)
            x += tp.randn(*x.shape, device=x.device)*std
    pbar.close()

    return trim_im(x, 2*p_sz) if pad else x


def sample_decorrupt(im: _tensor, noise_var: float, H: Callable, denoiser: Denoiser, p_sz: int, its: int=6,
                     beta_sched: Union[float, Callable]=None, n_grids: int=16, resample_grids: bool=True,
                     verbose: bool=True, pad: bool=True, opt_its: int=150, optimizer: str='adam', lr: float=1e-2,
                     init: _tensor=None):
    """
    Sample a decorruption using the EPLL prior
    :param im: the image to denoise as a torch tensor
    :param noise_var: the variance of the noise, as a float
    :param H: the corruption process applied to the image - this should be a differentiable function of the input x so
              that the matrix multiplication H@x is evaluated through the call H(x)
    :param denoiser: the denoiser function to use in order to denoise the patches; this should be a function that
                     receives as an input a torch tensor of shape [B, p_sz, p_sz(, 3)] as well as the noise variance,
                     such that the signature is denoiser(patches, noise_variance)
    :param p_sz: the patch size to use for denoising, as a single int (patches assumed to be square)
    :param its: number of iterations to run the algorithm for
    :param beta_sched: an update schedule for beta that will be used in the decorruption process
    :param n_grids: number of grids to use for the decorruption process
    :param resample_grids: whether to sample new grids every iteration or to use the same grids throughout
    :param verbose: whether a progressbar should be printed or not
    :param pad: whether to pad the image with reflection padding before denoising
    :param opt_its: number of inner optimization iterations to use when decorrupting
    :param optimizer: which optimizer to use; defaults to 'adam'
    :param lr: the learning rate to use with the optimizer; defaults to 0.01
    :param init: the initialization the decorruption process should start at; if the decorruption process (H) outputs
                 the same dimensionality as the original image, this defaults to the corrupted image, otherwise an
                 initialization must be supplied
    :return: the decorrupted image, as a torch tensor
    """
    # make sure that there is a schedule for beta and that it is a callable
    if beta_sched is None: beta_sched = _default_sched(noise_var)
    beta_sched = _callable_beta(beta_sched)
    noise_var = noise_var/n_grids
    dev = im.device

    # define the grids that will be used
    if n_grids > p_sz**2: n_grids = p_sz**2
    if init is None: init = im.clone()
    if pad: init = pad_im(init, 2*p_sz)
    grids = tp.ones(n_grids, *init.shape, device=dev)*init.clone()[None, ...]
    x = init.clone()

    x0, y0 = _choose_grids(p_sz, n_grids)

    def update(x: _tensor, grids: _tensor, beta: float, sample: bool):
        eps1, eps2 = tp.randn(*im.shape, device=im.device), tp.randn(*x.shape, device=x.device)
        s, G = np.sqrt(noise_var), np.sqrt(beta*n_grids)
        eps1, eps2 = eps1*s, eps2/G
        if not sample: eps1, eps2 = 0*eps1, 0*eps2
        H_f = lambda x: H(trim_im(x, 2*p_sz)) if pad else H(x)
        loss_func = lambda x: tp.sum((H_f(x)-im+eps1)*(H_f(x)-im+eps1))/noise_var + \
                              beta*tp.sum((grids-x[None]+eps2[None])**2)
        return optimize_function(loss_func, x, its=opt_its, lr=lr, optimizer=optimizer)

    pbar = tqdm(range(its), disable=not verbose)
    for i in pbar:
        b = beta_sched(i)
        pbar.set_postfix_str(f'beta: {b:.1f}')
        if resample_grids: x0, y0 = _choose_grids(p_sz, n_grids)

        for g in range(n_grids):
            grids[g] = grid_denoise(x, 1/b, x0[g], y0[g], denoiser, p_sz)
        x = update(x, grids, b, sample=i < its-1)
    return trim_im(x, 2*p_sz) if pad else x