from . import GMM
import numpy as np, torch as tp
from ImResize import resize
import torch.nn.functional as funct

_tensor = tp.Tensor


class Denoiser:
    def denoise(self, y: _tensor, sig: float): raise NotImplementedError


class GMMDenoiser(Denoiser):

    def __init__(self, gmm: GMM, MAP: bool=True):
        super(GMMDenoiser).__init__()
        if gmm.evd is None: gmm.calculate_evd()
        self.gmm = gmm
        self._MAP = MAP

    def _resp(self, y: _tensor, sig: float):
        L, U = self.gmm.evd
        L = L.clone() + sig
        det = tp.sum(tp.log(L), dim=-1)
        meaned = (y[None] - self.gmm.mu[:, None])@U
        mahala = tp.sum(meaned*(meaned@tp.diag_embed((1/L))), dim=-1).transpose(0, 1)
        ll = tp.log(self.gmm.pi[None]) - .5*(mahala + det[None])
        return tp.exp(ll - tp.logsumexp(ll, dim=1)[:, None])

    def _map(self, y: _tensor, sig: float, ks: list):
        means = sig*self.gmm.mu[ks][..., None] + self.gmm.S[ks]@y[..., None]
        L, U = self.gmm.evd
        L = L.clone() + sig
        return ((U@tp.diag_embed(1/L)@U.transpose(-2, -1))[ks]@means)[..., 0]

    def _sample(self, y: _tensor, sig: float, ks: list):
        m = self._map(y, sig, ks)
        L, U = self.gmm.evd
        L = 1/L.clone() + 1/sig
        samp = (tp.diag_embed(1/tp.sqrt(L))@U.transpose(-2, -1))[ks]@tp.randn(*m.shape, 1, device=y.device)
        return m + samp[..., 0]

    def denoise(self, y: _tensor, sig: float):
        shp = y.shape
        y = y.clone().reshape(y.shape[0], -1)
        r = self._resp(y, sig)

        if self._MAP:
            ks = tp.argmax(r, dim=1).tolist()
            return self._map(y, sig, ks).reshape(shp)

        else:
            ks = tp.multinomial(r, 1)[:, 0].tolist()
            return self._sample(y, sig, ks).reshape(shp)
