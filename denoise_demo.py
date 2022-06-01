import numpy as np
import torch
from matplotlib import pyplot as plt
from models import GMM, GMMDenoiser
from EPLL import denoise
from time import time
torch.set_grad_enabled(False)


# load image
im_path = 'data/108082.jpg'  # use 'data/sheep.jpg' for a slightly bigger image
im = plt.imread(im_path)/255.

# get device to use
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', dev)

# load denoiser
gmm = GMM.load('trained/GMM100.mdl')
denoiser = GMMDenoiser(gmm.to(dev))

# define seed for reproducibility
np.random.seed(0)

# add noise to image
noise_std = 35/255
noisy = im + noise_std*np.random.randn(*im.shape)

# plot noisy image
plt.figure(dpi=200)
plt.imshow(np.clip(noisy, 0, 1))
plt.axis('off')
plt.title('denoised')
plt.show()

# denoise image
n_grids = 16  # the smaller this is, the faster and less accurate the algorithm will be; for original EPLL, use 64
noisy = torch.from_numpy(noisy).float().to(dev)
t = time()
MAP = denoise(noisy, noise_std**2, denoiser, p_sz=8, n_grids=n_grids)

print(f'Denoising took {time()-t:.2f} seconds')

# plot denoised image
plt.figure(dpi=200)
plt.imshow(np.clip(MAP.cpu().numpy(), 0, 1))
plt.axis('off')
plt.title('denoised')
plt.show()
