import os
import torch as tp
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from skimage.transform import rescale
import pickle


image_transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class SingleFolder(Dataset):
    def __init__(self, root_dir, transform=None, ext='jpg'):
        self.root_dir = root_dir
        self.transform = transform
        self.im_list = sorted([f for f in os.listdir(root_dir) if f.endswith(ext)])
        print('Found {} images in {}.'.format(len(self.im_list), root_dir))
        if os.path.isfile(root_dir+'/latents.txt'):
            self.latents = np.loadtxt(root_dir+'/latents.txt').astype(np.float32)
        else:
            self.latents = None

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        if tp.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.root_dir, self.im_list[idx])
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.latents is not None:
            return image, self.latents[idx]
        return image


class DataWrapper(Dataset):

    def __init__(self, data: tp.Tensor):
        super(DataWrapper, self).__init__()
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, item):
        if tp.is_tensor(item): item = item.tolist()
        return self.data[item]


class GeneratorWrapper:
    def __init__(self, data: tp.Tensor, batch_size: int):
        self.data = data
        self.batch_size = batch_size

    def __next__(self):
        inds = np.random.choice(self.data.shape[0], self.batch_size).astype(int).tolist()
        return self.data[inds]

    def __iter__(self):
        return self


class PatchDataset(Dataset):

    def __init__(self, im_dir: str, N: int, patch_size: int, ext='jpg', preload: bool=True, scale: float=1):
        super(PatchDataset, self).__init__()
        self.dir = im_dir
        self.N = N
        self.ps = patch_size
        self.scale = scale
        if im_dir.endswith('.jpg'): self.im_list = [im_dir]
        else: self.im_list = sorted([f for f in os.listdir(im_dir) if f.endswith(ext)])
        l = len(self.im_list)
        self.patch_list = np.array([[np.random.rand(), np.random.rand(), np.random.choice(l, 1)[0]] for _ in range(N)])
        self.preloaded = preload
        if preload:
            print('Preloading {} image...'.format(len(self.im_list)), flush=True, end=' ')
            self.images = [self._load_im(os.path.join(self.dir, a), self.scale) for a in self.im_list]
            print('Done.', flush=True)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.patch_list[idx][0]
        y = self.patch_list[idx][1]
        if self.preloaded: image = self.images[int(self.patch_list[idx][2])]
        else:
            image = self.im_list[int(self.patch_list[idx][2])]
            image = self._load_im(os.path.join(self.dir, image), self.scale)
        a = int(np.floor((image.shape[0]-self.ps)*x))
        b = int(np.floor((image.shape[1]-self.ps)*y))
        patch = image[a:a+self.ps, b:b+self.ps]
        return tp.from_numpy(patch)

    @staticmethod
    def _load_im(path, scale):
        image = Image.open(path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        if scale != 1: return rescale((np.array(image)/255).astype(np.float32), scale=scale, preserve_range=True,
                                      anti_aliasing=True)
        else: return (np.array(image)/255).astype(np.float32)

    def save(self, path):
        d = {'preloaded': self.preloaded,
             'images': self.im_list,
             'patches': self.patch_list,
             'ps': self.ps}
        with open(path, 'wb') as output:
            pickle.dump(d, output)