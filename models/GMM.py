import numpy as np
import torch as tp
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
import pickle
tp.set_grad_enabled(False)


def _T(mat: tp.Tensor):
    return mat.transpose(-2, -1)


def _tr(tens: tp.Tensor):
    return tp.einsum('...kk->...k', tens).sum(dim=-1)


class GMM(nn.Module):

    def __init__(self, k: int=50, pi: tp.Tensor=None, mu: tp.Tensor=None, Sigma: tp.Tensor=None):
        """
        Initialize a GMM model
        :param k: number of clusters the GMM should have, as an int
        :param pi: the cluster assignment probabilities; as a tensor of length k that sums up to 1   (optional)
        :param mu: the cluster means; as a tensor with shape [k, d] where d is the data dimension   (optional)
        :param Sigma: the cluster covariances; as a tensor with shape [k, d, d]   (optional)
        """
        super().__init__()
        self.k = k if mu is None else mu.shape[0]
        # mixture probabilities
        self.pi = nn.Parameter(pi if pi is not None else tp.ones(k)/k, requires_grad=False)
        # means
        self.mu = nn.Parameter(mu if mu is not None else tp.zeros(1), requires_grad=False)
        # covariances
        self.S = nn.Parameter(Sigma if Sigma is not None else tp.zeros(1), requires_grad=False)

        # cholesky decomposition of covariances
        if Sigma is None: self.L = nn.Parameter(tp.zeros(1), requires_grad=False)
        else:
            try: self.L = nn.Parameter(tp.linalg.cholesky(Sigma), requires_grad=False)
            except RuntimeError:
                self.L = nn.Parameter(tp.linalg.cholesky(Sigma+1e-6*tp.eye(Sigma.shape[-1],
                                                                           device=self.mu.device)[None, ...]),
                                      requires_grad=False)
        self.L.data = self.L.data.float()

        self._d = 0 if mu is None else mu.shape[1]
        self.shape = [self._d]
        self._precision = 1e-10
        self.evd = None

    def __str__(self):
        return 'GMM_k{}'.format(self.k)

    def __repr__(self):
        return 'GMM_k{}'.format(self.k)

    def __copy__(self):
        ret = GMM(pi=self.pi, mu=self.mu, Sigma=self.S)
        ret.shape = self.shape
        return ret

    def handle_input(self, X: tp.Tensor):
        """
        Handles the input to the model, which may have more than 2 dimensions. If the model hasn't been initialized yet,
        this function does so
        :param X: the data to handle as a tensor of shape [N, ...] where N are the number of samples
        :return: the data as a tensor of shape [N, d] where d is the dimension of the model
        """
        shp = [0] if X.ndimension() == 1 else X.shape[1:]
        X = X.reshape([X.shape[0], np.prod(X[..., None].shape[1:])])
        X = X.to(device=self.mu.device)
        if self._d>0: assert X.shape[1]==self._d, 'Dimension of supplied data ({}) is different than dimension the ' \
                                                  'model was trained to handle ({})'.format(X.shape[1], self._d)
        else:
            self._d = X.shape[1]
            self.shape = shp
            self._param_init(X)
        return X

    def _param_init(self, X):
        """
        Initializes the parameters of the model using a simple heuristic
        :param X: the example data to generate the initial guess from
        """
        self.mu.data = tp.zeros([self.k, self._d], device=self.mu.device)
        self.S.data = tp.zeros([self.k, self._d, self._d], device=self.mu.device)
        for k in range(self.k):
            inds = np.random.choice(X.shape[0], 2*self._d)
            m = tp.mean(X[inds], dim=0)[None, :]
            self.mu.data[k] = m[0] + .1*tp.randn(m.shape[1], device=self.mu.device)
            self.S.data[k] = _T(X[inds] - m)@(X[inds] - m)/(2*self._d - 1) + .1*tp.eye(self._d, device=self.mu.device)
        self.L.data = tp.cholesky(self.S)

    def _log_like(self, X: tp.Tensor):
        """
        Calculates the log-likelihood of each sample-cluster pair
        :param X: the data; as a tensor of shape [N, d] where d is the dimension of the data
        :return: a tensor of the log-likelihood of each sample-cluster pair; a tensor of shape [N, k]
        """
        m = _T(tp.triangular_solve(_T(X[None, :, :]) - self.mu[:, :, None], self.L, upper=False)[0])
        m = _T(tp.sum(m * m, dim=-1))
        det = 2 * _tr(tp.log(tp.clamp(self.L, self._precision, 1e10)))
        return tp.log(self.pi)[None, :] - .5 * (det[None, :] + m) - .5*self._d*np.log(2*np.pi)

    def _part_log_like(self, X: tp.Tensor, observed):
        """
        Calculates the log-likelihood of each partial sample-cluster pair
        :param X: the partially observed data; a tensor of shape [N, len(observed)]
        :param observed: the observed features of the data; a list of observed indices
        :return: a tensor of the log-likelihood of each sample-cluster pair; a tensor of shape [N, k]
        """
        if type(observed)==int: observed = [observed]
        mu = self.mu[:, observed]
        L = self.L[:, observed][:, :, observed]
        m = _T(tp.triangular_solve(_T(X[None, :, :]) - mu[:, :, None], L, upper=False)[0])
        m = _T(tp.sum(m * m, dim=-1))
        det = 2 * _tr(tp.log(tp.clamp(L, self._precision, 1e10)))
        return tp.log(self.pi)[None, :] - .5 * (det[None, :] + m)

    def log_likelihood(self, X: tp.Tensor, observed=None):
        """
        Calculates the log-likelihood of the samples
        :param X: the data; a tensor of shape [N, ...]
        :param observed: which features in the data are observed, as a list of indices or a boolean array (optional);
                         if this value is given, the shape of X must be [N, len(observed)]
        :return: a tensor of the log-likelihood of each sample-cluster pair; a tensor of shape [N, k]
        """
        if observed is None:
            X = self.handle_input(X)
            return self._log_like(X)
        return self._part_log_like(X, observed)

    def responsibilities(self, X: tp.Tensor, observed=None):
        """
        Calculates the cluster responsibilities of the data
        :param X: the data; a tensor of shape [N, ...]
        :param observed: which features in the data are observed, as a list of indices or a boolean array (optional);
                         if this value is given, the shape of X must be [N, len(observed)]
        :return: a tensor of the responsibilities of each sample-cluster pair; a tensor of shape [N, k]
        """
        ll = self.log_likelihood(X, observed)
        return tp.exp(ll - tp.logsumexp(ll, dim=1)[:, None])

    def fit(self, X: tp.Tensor, its: int=30, verbose: bool=True, return_ll: bool=False, sample_feats: float=None):
        """
        Estimates the maximum likelihood parameters for the data given in X
        :param X: the training data; a tensor of shape [N, ...]
        :param its: the number of iterations to train, as an int (default=30)
        :param verbose: a boolean indicating whether or not updates regarding the training procedure should be printed
                        (default=True)
        :param return_ll: a boolean indicating whether the train log-likelihood of each step should be returned
                         (default=True)
        :param sample_feats: the fraction of features to sample during training (optional); if given, only a fraction of
                             the features will be used in order to calculate the responsibilities in each iteration;
                             must be a float greater than 0 and smaller than 1
        :return: the trained model if return_ll is False, otherwise a list of log-likelihoods for each iteration is also
                 returned
        """
        if sample_feats is not None: assert 0 < sample_feats <= 1, 'Ratio of features to sample must be greater ' \
                                                                   'than 0 and smaller than 1'
        if sample_feats == 1: sample_feats = None

        X = self.handle_input(X)
        N = X.shape[0]
        pbar = tqdm(range(its), disable=not verbose)
        lls = []
        for _ in pbar:
            if sample_feats is None: resp = self.log_likelihood(X)
            else:
                inds = np.random.choice(self._d, int(np.ceil(sample_feats * self._d)), replace=False)
                resp = self.log_likelihood(X[:, inds], observed=inds)
            lls.append(tp.logsumexp(resp.cpu(), dim=1).mean().numpy())
            pbar.set_postfix_str('log-likelihood:{:.2f}'.format(lls[-1]))
            resp = tp.exp(resp - tp.logsumexp(resp, dim=1)[:, None])
            resp = tp.max(resp, tp.tensor(self._precision, device=resp.device))
            norm = tp.sum(resp, dim=0)

            self.pi.data = tp.max(tp.sum(resp, dim=0)/N, tp.tensor(self._precision, device=resp.device))
            self.pi.data = self.pi/tp.sum(self.pi)

            self.mu.data = tp.sum(resp[:, :, None]*X[:, None, :], dim=0)/norm[:, None]
            m = X[:, None, :]-self.mu[None, :, :]
            self.S.data = (resp[:, :, None]*m).permute((1, 2, 0))@tp.transpose(m, dim0=0, dim1=1) / norm[:, None, None]
            self.L.data = tp.cholesky(self.S + self._precision*tp.eye(self._d, device=self.mu.device)[None, :, :])
        if not return_ll: return self
        return self, lls

    def fit_batch(self, train_data: Dataset, batch_size: int=1028, its: int=30, verbose: bool=True,
                  sample_feats: float=None, num_workers: int=0, save_path: str=None, return_ll: bool=False) -> 'GMM':
        """
        Estimates the maximum likelihood parameters for the data given in a batched manner
        :param train_data: a torch.Dataset that loads the training data as a tensor
        :param batch_size: the size of each batch, as an int
        :param its: the number of iterations to train, as an int (default=30)
        :param verbose: a boolean indicating whether or not updates regarding the training procedure should be printed
                        (default=True)
        :param sample_feats: the fraction of features to sample during training (optional); if given, only a fraction of
                             the features will be used in order to calculate the responsibilities in each iteration;
                             must be a float greater than 0 and smaller than 1
        :param num_workers: the number of workers that should be used to load the batches, as an int
        :param save_path: a path to save the model after each iteration; if no path is given, the model isn't saved
                          after each iteration
        :param return_ll:
        :return: the trained model
        """
        if sample_feats is not None: assert 0 < sample_feats <= 1, 'Fraction of features to sample must be greater ' \
                                                                   'than 0 and smaller than 1'
        if sample_feats == 1: sample_feats = None

        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        d = np.product(train_data[0].shape)
        if len(train_data) > self.k*d: _ = self.handle_input(tp.stack([train_data[i] for i in range(self.k * d)]))
        else: _ = self.handle_input(tp.stack([train_data[i] for i in range(len(train_data))]))

        pbar = tqdm(range(its), disable=not verbose)
        N = len(train_data)
        lls = []
        for iter in pbar:
            r_norm = tp.zeros(self.k, device=self.mu.device)
            mu_upd = tp.zeros((self.k, self._d), device=self.mu.device)
            S_upd = tp.zeros((self.k, self._d, self._d), device=self.mu.device)
            ll = 0
            for i, X in enumerate(loader):
                pbar.set_postfix_str('batch {}/{}'.format(i+1, len(loader)) +
                                     (';  log-likelihood: {:.2f}'.format(lls[-1]) if iter > 0 else ''))
                X = self.handle_input(X)

                if sample_feats is None: resp = self.log_likelihood(X)
                else:
                    inds = np.random.choice(self._d, int(np.ceil(sample_feats*self._d)), replace=False)
                    resp = self.log_likelihood(X[:, inds], observed=inds)

                ll += tp.logsumexp(resp.cpu(), dim=1).mean().numpy()
                resp = tp.exp(resp - tp.logsumexp(resp, dim=1)[:, None])
                resp = tp.max(resp, tp.tensor(self._precision, device=resp.device))
                r_norm += tp.sum(resp, dim=0)
                mu_upd += tp.sum(resp[:, :, None]*X[:, None, :], dim=0)
                m = X[:, None, :] - self.mu[None, :, :]
                S_upd += (resp[:, :, None]*m).permute((1, 2, 0))@m.permute((1, 0, 2))

            lls.append(ll/batch_size)
            self.pi.data = tp.max(r_norm/N, tp.tensor(self._precision, device=self.mu.device))
            self.pi.data = self.pi / tp.sum(self.pi)

            self.mu.data = mu_upd/r_norm[:, None]
            self.S.data = S_upd/r_norm[:, None, None]

            try: self.L.data = tp.cholesky(self.S.double() + self._precision*
                                           tp.eye(self._d, device=self.mu.device)[None, :, :]).float()
            except RuntimeError: self.L.data = tp.cholesky(self.S.double() + 1e-5*
                                                           tp.eye(self._d, device=self.mu.device)[None, :, :]).float()

            if save_path is not None: self.save(save_path)
        return self if not return_ll else (self, lls)

    def sample(self, N: int, k: int=None) -> tp.Tensor:
        """
        Sample new data points from the trained model
        :param N: the number of samples to generate, as an int
        :param k: which cluster the samples should be drawn from, as an int (optional)
        :return: the generated samples as a tensor of shape [N, ...]
        """
        assert self._d > 0, 'Model has not been trained yet'
        inds = np.random.choice(self.k, N, p=self.pi.cpu().numpy()).tolist()
        if k is not None:
            assert 0 <= k <= self.k
            inds = list(k*np.ones(N).astype(int))
        samples = self.mu[inds] + tp.squeeze(self.L[inds]@tp.randn(N, self._d, 1, device=self.mu.device))
        return samples.reshape([N] + list(self.shape))

    def conditional(self, X: tp.Tensor, observed, map: bool=True, sample: bool=False):
        """
        Get conditional mean (or sample) of the unobserved pixels from the data X
        :param X: the observed data; a tensor of shape [N, # observed pixels]
        :param observed: a list of the observed pixel indices
        :param map: whether to return the conditional mean of only the Gaussians with the top responsibilities; default
                    is True - if set to False, the weighted conditional mean of all of the Gaussians is returned
        :param sample: a boolean indicating whether to sample from the conditiona distribution; defauls is False
        :return: the conditional means (or samples) as a tensor of shape [N, # pixels]
        """
        orig_device = X.device
        X = X.to(device=self.mu.device)
        unobserved = np.ones(self._d).astype(bool)
        unobserved[observed] = False
        r = self.responsibilities(X, observed)
        ret = tp.zeros((X.shape[0], self._d), device=self.mu.device)
        ret[:, observed] = X

        if not map and not sample:
            meaned = tp.cholesky_solve(_T(X[None, :, :] - self.mu[:, None, observed]),
                                       self.L[:, observed][:, :, observed])[0].permute((2, 0, 1))
            meaned = tp.sum(r[..., None]*(self.mu[None, :, unobserved] +
                                          (self.S[None, :, unobserved][..., observed]@meaned[..., None])[:, :, 0]),
                            dim=1)

        elif map:
            ks = np.argmax(r.cpu().numpy(), dim=1).tolist()
            meaned = tp.cholesky_solve(X - self.mu[ks][:, observed], self.L[ks][:, observed][:, :, observed])[0]
            meaned = self.mu[ks][:, unobserved] + (self.S[ks][:, unobserved][..., observed]@meaned[..., None])[:, :, 0]

        else:
            ks = [np.random.choice(self.k, 1, p=r[i, :].cpu())[0] for i in range(X.shape[0])]
            T = tp.solve(self.S[ks][:, observed][:, :, unobserved], self.L[ks][:, observed][:, :, observed])[0]
            L = tp.cholesky(self.S[ks][:, unobserved][:, : unobserved] - _T(T)@T)
            eps = tp.solve(tp.randn(*X[:, :, None].shape), L)[0][:, :, 0]
            meaned = tp.cholesky_solve(X - self.mu[ks][:, observed], self.L[ks][:, observed][:, :, observed])[0]
            meaned = self.mu[ks][:, unobserved] + \
                     (self.S[ks][:, unobserved][..., observed]@meaned[..., None])[:, :, 0] + eps

        ret[:, unobserved] = meaned
        return ret.to(device=orig_device)

    def calculate_evd(self):
        evd = tp.linalg.eigh(self.S.double())
        vals = tp.clamp(evd[0], min=(.5/255)**2)
        vecs = evd[1]
        self.evd = [vals.float(), vecs.float()]
        self.S.data = (vecs@tp.diag_embed(vals)@_T(vecs)).float()
        self.L.data = tp.linalg.cholesky(self.S + self._precision*tp.eye(self._d, device=self.mu.device)[None, :, :])

    def save(self, path: str):
        """
        Save model parameters at given path
        :param path: a string depicting the file the parameters should be saved to
        """
        d = {
            'pi': self.pi.data.cpu(),
            'mu': self.mu.data.cpu(),
            'Sigma': self.S.data.cpu(),
            'shape': self.shape,
        }
        with open(path, 'wb') as output:
            pickle.dump(d, output)

    def scale_covariances(self, scale: float):
        self.S.data = self.S.data*scale
        self.L.data = self.L.data*np.sqrt(scale)

    @staticmethod
    def load(path: str) -> 'GMM':
        """
        Load a GMM model saved by the above save method
        :param path: the path to the file that is a saved GMM model
        :return: the loaded GMM model
        """
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            ret = GMM(pi=d['pi'], mu=d['mu'], Sigma=d['Sigma'])
            ret.shape = d['shape']
            return ret
        except Exception as e:
            print('Supplied path ({}) is not a saved GMM model; following error occured: {}'.format(path, e))
