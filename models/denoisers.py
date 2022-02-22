from . import GMM
import numpy as np, torch as tp

_tensor = tp.Tensor


def _GMM_resp(y: _tensor, gmm: GMM, sig: float):
    L, U = gmm.evd
    L += sig
    det = tp.sum(tp.log(L), dim=-1)
    meaned = (y[None] - gmm.mu[:, None])@U
    mahala = tp.sum(meaned*(meaned@tp.diag_embed((1/L))), dim=-1).transpose(0, 1)
    ll = tp.log(gmm.pi[None]) - .5*(mahala + det[None])
    return tp.exp(ll - tp.logsumexp(ll, dim=1)[:, None])


def _GMM_MAP(y: _tensor, gmm: GMM, sig: float, ks: list):
    means = sig*gmm.mu[ks] + gmm.S[ks]@y
    L, U = gmm.evd
    L += sig
    return (U@tp.diag_embed(1/L)@U.transpose(-2, -1))[ks]@means


def GMM_denoiser(gmm: GMM):
    if gmm.evd is None: gmm.calculate_evd()

    def denoiser(y: tp.Tensor, sig: float):
        shp = y.shape
        y = y.reshape(y.shape[0], -1)
        r = _GMM_resp(y, gmm, sig)
        ks = tp.argmax(r, dim=1).tolist()
        return _GMM_MAP(y, gmm, sig, ks).reshape(shp)
    return denoiser
