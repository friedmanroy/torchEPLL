import numpy as np
import torch as tp
from typing import Union, Callable, Tuple
from models import Denoiser
tp.set_grad_enabled(False)

_tensor = tp.Tensor


def _callable_beta(beta: Union[int, float, Callable]):
    if not callable(beta):
        num = beta
        return lambda i: num
    return beta


def _choose_grids(p_sz: int, n_grids: int):
    # choose random, different, grids
    chs = np.random.choice(p_sz**2, n_grids, replace=False)
    x0, y0 = chs//p_sz, chs%p_sz
    return [int(a) for a in x0], [int(a) for a in y0]


def pad_im(im: _tensor, p_sz: int, mode: str='reflect', end_values=0.):
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


def trim_im(im: _tensor, p_sz: int): return im[p_sz:-p_sz][:, p_sz:-p_sz]


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
    outp = tp.zeros(x1//p_sz, y1//p_sz, p_sz, p_sz, 3, device=im.device)
    for i in range(outp.shape[0]):
        for j in range(outp.shape[1]):
            outp[i, j] = im[x0+i*p_sz:x0+(i+1)*p_sz, y0+j*p_sz:y0+(j+1)*p_sz]
    # outp = im.clone()[x0:x1, y0:y1]
    # outp = outp.unfold(1, p_sz, p_sz).unfold(0, p_sz, p_sz)
    # outp = outp.permute((0, 1, -1, -2, 2)) if clr else outp.permute((0, 1, -1, -2))
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
    clr = im.ndim == 3
    outp = im.clone()
    ps = ps.reshape(shape)
    p_sz = ps.shape[2]
    for i in range(ps.shape[0]):
        for j in range(ps.shape[1]):
            outp[x0+i*p_sz:x0+(i+1)*p_sz, y0+j*p_sz:y0+(j+1)*p_sz] = ps[i, j]
    # ret = ps.reshape(shape)
    # ret = ret.permute((0, 1, -1, -2, 2)) if clr else ret.permute((0, 1, -1, -2))
    # ret = ps.reshape(shape[0]*shape[2], shape[1]*shape[3], 3) \
    #     if clr else ps.reshape(shape[0]*shape[2], shape[1]*shape[3])
    # outp = im.clone()
    # outp[x0:x0+ret.shape[0], y0:y0+ret.shape[1]] = ret
    return outp


def grid_denoise(im: _tensor, var: float, x0: int, y0: int, denoiser: Denoiser, p_sz: int, low_mem: bool=False):
    """
    Denoise a grid corrupted by isotropic noise
    :param im: the image to denoise; a torch tensor with shape [N, M] or [N, M, 3]
    :param var: the variance of the noise added to the image as a positive float
    :param x0: an int depicting the X position of the top left corner of the grid
    :param y0: an int depicting the y position of the top left corner of the grid
    :param denoiser: the denoiser function to use in order to denoise the patches; this should be a Callable that
                     receives as an input a torch tensor of shape [B, p_sz, p_sz(, 3)] as well as the noise variance,
                     such that the signature is denoiser(patches, noise_variance)
    :param low_mem: a boolean indicating whether low memory should be assumed; if this is set to true, the patches are
                    denoised in batches, instead of denoising all of them at the same time
    :return: the cleaned image as a torch tensor with the same shape as the input image
    """
    batch_sz = 2500
    # break to patches
    ps, shp = to_patches(im, p_sz, x0, y0)

    # denoise according to shape
    den_ps = denoiser.denoise(ps, var)

    # build denoised image back from patches
    return to_image(im, den_ps, shp, x0, y0)
