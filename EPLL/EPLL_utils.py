import numpy as np
import torch as tp
from typing import Union, Callable, Tuple, List
from models import Denoiser
import warnings
tp.set_grad_enabled(False)

_tensor = tp.Tensor


def _default_sched(noise_var: float) -> Callable: return lambda i: (1 if i==0 else 2**(i+1))/noise_var


def _callable_beta(beta: Union[int, float, Callable]) -> Callable:
    """
    Makes sure that beta is a callable function, even if the input was an int or a float
    :param beta: the value for beta; if an int or float, then a callable that always returns the same value is returned
    :return: a callable function that returns the value of beta at each iteration
    """
    if not callable(beta):
        num = beta
        return lambda i: num
    return beta


def _choose_grids(p_sz: int, n_grids: int) -> Tuple[List, List]:
    """
    Helper function to sample the placement of the grids
    :param p_sz: the sizes of the patches used
    :param n_grids: the number of grids that should be chosen
    :return: two lists, the first for the xs and the second for the ys of the upper left corner of each grid
    """
    chs = np.random.choice(p_sz**2, n_grids, replace=False)
    chs = chs
    x0, y0 = chs//p_sz, chs%p_sz
    return [int(a) for a in x0], [int(a) for a in y0]


def pad_im(im: _tensor, p_sz: int, mode: str='reflect', end_values=0.) -> _tensor:
    """
    Reflection pad image with given pad size
    :param im: the image to pad; a torch tensor with shape [N, M] or [N, M, 3]
    :param p_sz: an int dictating total size to pad
    :param mode: padding mode to be passed down to numpy.pad
    :return: the pad image; a torch tensor with shape [N + p_sz, M + p_sz, ...]
    """
    if mode in ['linear_ramp', 'constant']:
        if im.ndim > 2:
            return tp.from_numpy(np.pad(im.cpu().numpy(), ((p_sz // 2, p_sz // 2), (p_sz // 2, p_sz // 2), (0, 0)),
                                        mode=mode, end_values=end_values)).float().to(im.device)
        else:
            return tp.from_numpy(np.pad(im.cpu().numpy(), ((p_sz // 2, p_sz // 2), (p_sz // 2, p_sz // 2)),
                                        mode=mode, end_values=end_values)).float().to(im.device)

    if im.ndim > 2:
        return tp.from_numpy(np.pad(im.cpu().numpy(), ((p_sz // 2, p_sz // 2), (p_sz // 2, p_sz // 2), (0, 0)),
                                    mode=mode)).float().to(im.device)
    else:
        return tp.from_numpy(np.pad(im.cpu().numpy(), ((p_sz // 2, p_sz // 2), (p_sz // 2, p_sz // 2)),
                                    mode=mode)).float().to(im.device)


def trim_im(im: _tensor, p_sz: int) -> _tensor: return im[p_sz//2:-p_sz//2][:, p_sz//2:-p_sz//2]


def to_patches(im: _tensor, p_sz: int, x0: int, y0: int) -> Tuple[_tensor, Tuple]:
    """
    Break image into non-overlapping patches while storing initial positions
    :param im: the image to break into patches
    :param p_sz: an int depicting the size of the patches (only square patches allowed)
    :param x0: an int depicting the X position of the top left corner of the grid
    :param y0: an int depicting the y position of the top left corner of the grid
    :return: a torch tensor containing all of the (whole) patches on the grid and shape for reconstruction
    """
    clr = im.ndim == 3
    x1, y1 = im.shape[0] - (im.shape[0]-x0)%p_sz, im.shape[1] - (im.shape[1]-y0)%p_sz
    if clr: outp = tp.zeros(x1//p_sz, y1//p_sz, p_sz, p_sz, 3, device=im.device)
    else: outp = tp.zeros(x1//p_sz, y1//p_sz, p_sz, p_sz, device=im.device)
    for i in range(outp.shape[0]):
        for j in range(outp.shape[1]):
            outp[i, j] = im[x0+i*p_sz:x0+(i+1)*p_sz, y0+j*p_sz:y0+(j+1)*p_sz]
    return outp.reshape(-1, *outp.shape[2:]), outp.shape


def to_image(im: _tensor, ps: _tensor, shape: tuple, x0: int, y0: int) -> _tensor:
    """
    Reassemble patches on the grid into a full image
    :param im: the original image that was broken into patches; a torch tensor with shape [N, M, ...]
    :param ps: a list of lists of patches (same as in the output of to_patches)
    :param shape: a tuple of the original patch matrix shape
    :param x0: an int depicting the X position of the top left corner of the grid
    :param y0: an int depicting the y position of the top left corner of the grid
    :return: the reassembled image as a torch tensor with the same shape as im
    """
    outp = im.clone()
    ps = ps.reshape(shape)
    p_sz = ps.shape[2]
    for i in range(ps.shape[0]):
        for j in range(ps.shape[1]):
            outp[x0+i*p_sz:x0+(i+1)*p_sz, y0+j*p_sz:y0+(j+1)*p_sz] = ps[i, j]
    return outp


def grid_denoise(im: _tensor, var: float, x0: int, y0: int, denoiser: Denoiser, p_sz: int) -> _tensor:
    """
    Denoise a grid corrupted by isotropic noise
    :param im: the image to denoise; a torch tensor with shape [N, M] or [N, M, 3]
    :param var: the variance of the noise added to the image as a positive float
    :param x0: an int depicting the X position of the top left corner of the grid
    :param y0: an int depicting the y position of the top left corner of the grid
    :param denoiser: the denoiser function to use in order to denoise the patches; this should be a Callable that
                     receives as an input a torch tensor of shape [B, p_sz, p_sz(, 3)] as well as the noise variance,
                     such that the signature is denoiser(patches, noise_variance)
    :return: the cleaned image as a torch tensor with the same shape as the input image
    """
    # break to patches
    ps, shp = to_patches(im, p_sz, x0, y0)

    # denoise according to shape
    den_ps = denoiser.denoise(ps, var)

    # build denoised image back from patches
    im = to_image(im, den_ps, shp, x0, y0)

    return im


def optimize_function(loss_func: Callable, params: _tensor, its: int, optimizer: str='adam', lr: float=1e-2,
                      momentum: float=.9) -> _tensor:
    """
    Helper function for optimizations
    :param loss_func: a callable loss function that returns a differentiable torch tensor
    :param params: the parameters that should be optimized, as a torch tensor, which the loss relies on
    :param its: number of iterations to run the optimization process
    :param optimizer: a string representing the optimizer that should be used; the known optimizers are: 'sgd',
                      'adam', 'rmsprop' - if an unkown optimizer is given, will default to 'adam'
    :param lr: the learning rate that the optimzer should use
    :param momentum: the momentum the optimizer should use
    :return: the optimized parameters as a torch tensor
    """
    if optimizer.lower()=='adam': opt = tp.optim.Adam([params], lr=lr)
    elif optimizer.lower()=='sgd': opt = tp.optim.SGD([params], lr=lr, momentum=momentum)
    elif optimizer.lower()=='rmsprop': opt = tp.optim.RMSprop([params], lr=lr, momentum=momentum)
    else:
        warnings.warn(f'Optimizer "{optimizer}" unkown. The list of known optimizers is: Adam, SGD and RMSprop.'
                      f'Optimizer defaulting to Adam.')
        opt = tp.optim.Adam([params], lr=lr)

    params = params.requires_grad_()
    with tp.enable_grad():
        for i in range(its):
            opt.zero_grad()
            loss = loss_func(params)
            loss.backward()
            opt.step()
    return params.data