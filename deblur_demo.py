import numpy as np
import torch
from torch.nn.functional import conv2d
from matplotlib import pyplot as plt
from models import GMM, GMMDenoiser
from EPLL import decorrupt
from time import time
torch.set_grad_enabled(False)


def gauss_kernel(sz: int, sigx: float, sigy: float, corr: float=0):
    if sz%2 == 0:
        xx, yy = np.meshgrid(np.arange(-sz//2, sz//2), np.arange(-sz//2, sz//2))
    else:
        xx, yy = np.meshgrid(np.arange(-sz//2+1, sz//2+1), np.arange(-sz//2+1, sz//2+1))
    xx, yy = xx/sigx, yy/sigy
    kern = np.repeat(np.exp(-.5*(xx**2 + 2*corr*xx*yy + yy**2))[:, :, None], 3, axis=-1)
    return kern / np.sum(kern, axis=(0, 1))[None, None, :]


def isotropic_kernel(sz: int, sigma: float):
    return gauss_kernel(sz, sigma, sigma)


# load image
im_path = 'data/16077.jpg'  # use 'data/sheep.jpg' for a slightly bigger image
im = plt.imread(im_path)/255.

# get device to use
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', dev)

# load denoiser
gmm = GMM.load('trained/GMM100.mdl')
denoiser = GMMDenoiser(gmm.to(dev))


# define kernel and corruption function
blur_scale = 2  # bigger for blurrier images
kernel = torch.from_numpy(isotropic_kernel(15, blur_scale)[:, :, 0]).float().to(dev)[None, None]
H = lambda x: conv2d(x.permute(-1, 0, 1)[:, None], kernel, padding=kernel.shape[-1]//2)[:, 0].permute(1, 2, 0)

# get corrupted image
n = 1/255
corr = H(torch.from_numpy(im).to(dev).float())
corr = corr + n*torch.randn(*corr.shape, device=corr.device)

# plot corrupted image
plt.figure(dpi=200)
plt.imshow(np.clip(corr.cpu().numpy(), 0, 1))
plt.axis('off')
plt.show()

# define schedule for beta
alpha = 1/50
beta = lambda i: min(2**i / alpha, 3000)

# decorrupt image
its = 6  # number of iterations to run the algorithm
n_grids = 16  # the smaller this is, the faster and less accurate the algorithm will be; for original EPLL, use 64
t = time()
MAP = decorrupt(corr, n**2, H, denoiser, p_sz=8, its=its, beta_sched=beta, n_grids=n_grids)

print(f'Deblurring took {time()-t:.2f} seconds')

# plot restored image
plt.figure(dpi=200)
plt.imshow(np.clip(MAP.cpu().numpy(), 0, 1))
plt.axis('off')
plt.title('deblurred')
plt.show()
