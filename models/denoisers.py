from . import GMM
import torch as tp

_tensor = tp.Tensor


class Denoiser:
    """
    Denoiser class interface which should be used for any patch-denoiser to be plugged into EPLL
    """

    def denoise(self, y: _tensor, noise_var: float):
        """
        The main denoising function
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param noise_var: the variance of the noise that should be removed from the patches
        :return: the denoised patches, as a torch tensor of shape [N, p_sz, p_sz, 3]
        """
        raise NotImplementedError


class GMMDenoiser(Denoiser):
    """
    A denoiser class that uses a GMM in order to denoise the patches
    """

    def __init__(self, gmm: GMM, MAP: bool=True):
        """
        Initializes the GMM denoiser
        :param gmm: the GMM model which should be used in order to denoise patches
        :param MAP: whether MAP denoisings are to be returned or posterior samples
        """
        super(GMMDenoiser).__init__()
        if gmm.evd is None: gmm.calculate_evd()
        self.gmm = gmm
        self._MAP = MAP

    def _resp(self, y: _tensor, sig: float):
        """
        Calculates the responsibility of each cluster in the GMM for each patch to be denoised
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param sig: the noise variance to remove
        :return: the responsibilities of the GMM clusters, as a torch tensor of shape [k, N], where k is the number of
                 clusters in the GMM
        """
        L, U = self.gmm.evd
        L = L.clone() + sig
        det = tp.sum(tp.log(L), dim=-1)
        meaned = (y[None] - self.gmm.mu[:, None])@U
        mahala = tp.sum(meaned*(meaned@tp.diag_embed((1/L))), dim=-1).transpose(0, 1)
        ll = tp.log(self.gmm.pi[None]) - .5*(mahala + det[None])
        return tp.exp(ll - tp.logsumexp(ll, dim=1)[:, None])

    def _map(self, y: _tensor, sig: float, ks: list):
        """
        Returns the (approximate) MAP denoisings of each patch using the GMM model
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param sig: the noise variance to remove
        :param ks: a list of clusters to be used in order to denoise the patches (a list of length N)
        :return: the denoised patches
        """
        means = sig*self.gmm.mu[ks][..., None] + self.gmm.S[ks]@y[..., None]
        L, U = self.gmm.evd
        L = L.clone() + sig
        return ((U@tp.diag_embed(1/L)@U.transpose(-2, -1))[ks]@means)[..., 0]

    def _sample(self, y: _tensor, sig: float, ks: list):
        """
        Samples denoisings from the GMM
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param sig: the noise variance to remove
        :param ks: a list of clusters to be used in order to denoise the patches (a list of length N)
        :return: the denoised patches
        """
        m = self._map(y, sig, ks)
        L, U = self.gmm.evd
        L = 1/L.clone() + 1/sig
        samp = (tp.diag_embed(1/tp.sqrt(L))@U.transpose(-2, -1))[ks]@tp.randn(*m.shape, 1, device=y.device)
        return m + samp[..., 0]

    def denoise(self, y: _tensor, noise_var: float):
        """
        Main denoising function
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param noise_var: the variance of the noise that should be removed from the patches
        :return: the denoised patches, as a torch tensor of shape [N, p_sz, p_sz, 3]
        """
        shp = y.shape
        y = y.clone().reshape(y.shape[0], -1)
        r = self._resp(y, noise_var)

        # return MAP denoisings
        if self._MAP:
            ks = tp.argmax(r, dim=1).tolist()
            return self._map(y, noise_var, ks).reshape(shp)
        # sample patch denoisings from the posterior
        else:
            ks = tp.multinomial(r, 1)[:, 0].tolist()
            return self._sample(y, noise_var, ks).reshape(shp)
