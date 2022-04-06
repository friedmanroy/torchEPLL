from . import GMM
import numpy as np, torch as tp

_tensor = tp.Tensor


def _GMM_resp(y: _tensor, gmm: GMM, sig: float):
    L, U = gmm.evd
    L = L + sig
    det = tp.sum(tp.log(L), dim=-1)
    meaned = (y[None] - gmm.mu[:, None])@U
    mahala = tp.sum(meaned*(meaned@tp.diag_embed((1/L))), dim=-1).transpose(0, 1)
    ll = tp.log(gmm.pi[None]) - .5*(mahala + det[None])
    return tp.exp(ll - tp.logsumexp(ll, dim=1)[:, None])


def _GMM_MAP(y: _tensor, gmm: GMM, sig: float, ks: list):
    means = sig*gmm.mu[ks][..., None] + gmm.S[ks]@y[..., None]
    L, U = gmm.evd
    L = L + sig
    return ((U@tp.diag_embed(1/L)@U.transpose(-2, -1))[ks]@means)[..., 0]


def _GMM_sample(y: _tensor, gmm: GMM, sig: float, ks: list):
    m = _GMM_MAP(y, gmm, sig, ks)
    L, U = gmm.evd
    L = 1/L + 1/sig
    samp = (tp.diag_embed(1/tp.sqrt(L))@U.transpose(-2, -1))[ks]@tp.randn(*m.shape, 1, device=y.device)
    return m + samp[..., 0]


def GMM_denoiser(gmm: GMM):
    if gmm.evd is None: gmm.calculate_evd()

    def denoiser(y: tp.Tensor, sig: float):
        shp = y.shape
        y = y.clone().reshape(y.shape[0], -1)
        r = _GMM_resp(y, gmm, sig)
        ks = tp.argmax(r, dim=1).tolist()
        return _GMM_MAP(y, gmm, sig, ks).reshape(shp)
    return denoiser


def samp_GMM_denoiser(gmm: GMM):
    if gmm.evd is None: gmm.calculate_evd()

    def denoiser(y: tp.Tensor, sig: float):
        shp = y.shape
        y = y.clone().reshape(y.shape[0], -1)
        r = _GMM_resp(y, gmm, sig)
        ks = tp.multinomial(r, 1)[:, 0].tolist()
        return _GMM_sample(y, gmm, sig, ks).reshape(shp)
    return denoiser