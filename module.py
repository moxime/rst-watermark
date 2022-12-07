import sys
import logging
import argparse
import numpy as np
import torch
from torch import nn
import PIL
import time
import matplotlib.pyplot as plt

HALF_PLANE = True
HALF_PLANE = False

def print_time(t):
    if t < 1e-6:
        return '{:.0f} ns'.format(t * 1e9)
    elif t < 1e-3:
        return '{:.0f} us'.format(t * 1e6)
    elif t < 1:
        return '{:.0f} ms'.format(t * 1e3)
    elif t < 1e2:
        return '{:.0f} s'.format(t)
    elif t < 3600:
        return '{:.0f} m'.format(t / 60)


class Mask(nn.Module):

    def __init__(self, H, W, T=180, weighing_harmonics=2, init_lp='rand', store_tensors=False, **kw):

        super().__init__(**kw)
        self._masks = {}
        self._shape = (H, W)
        self.T = T

        self.store_tensors = store_tensors

        self.thetas = nn.Parameter(torch.linspace(0, np.pi * (1 - 1 / self.T), self.T), requires_grad=False)

        self.lp = nn.Parameter(torch.randn(weighing_harmonics + 1))

        if not init_lp:
            self.lp.data = torch.zeros_like(self.lp)
            self.lp.data[0] = 1.
            self.lp.requires_grad_(False)

        self._reset_masks()

    @property
    def device(self):
        return self.lp.device

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, v):

        if self.shape[0] != v[0] or self.shape[1] != v[1]:
            self._shape = v
            self._reset_masks()

    def _reset_masks(self, i=None):

        if i is not None:
            self._masks[i] = None
        else:
            for i in range(self.T):
                self._reset_masks(i)

    def _compute_masks(self, *i):
        i = list(i)
        logging.debug('Computing masks for {}'.format(', '.join(str(_) for _ in i)))
        H, W = self.shape

        assert not H % 2 and not W % 2

        if HALF_PLANE:
            Hmin = 0
            nH = H // 2
        else:
            Hmin = -H // 2
            nH = H

        Wmin = - W // 2
        nW = W
            
        Hmax = H // 2 - 1
        Wmax = W // 2 - 1
        
        dim_names = ('P', 'theta', 'm', 'n')
        device = self.device
        m_ = torch.linspace(Hmin, Hmax, nH, device=device)[None, None, :, None].rename(*dim_names)
        n_ = torch.linspace(Wmin, Wmax, nW, device=device)[None, None, None, :].rename(*dim_names)

        theta = self.thetas[i]
        cost = torch.cos(theta)[None, :, None, None].rename(*dim_names)
        sint = torch.sin(theta)[None, :, None, None].rename(*dim_names)

        """
        print('m  ', *m_.names, *m_.shape)
        print('cos', *cost.names, *cost.shape)
        print('n  ', *n_.names, *n_.shape)
        print('sin', *sint.names, *sint.shape)
        """

        p = torch.LongTensor([_ for _, l in enumerate(self.lp) if l])
        lp = self.lp[p]

        mt = (m_ * cost + n_ * sint).rename(None).expand(len(p), len(theta), nH, nW).rename(*dim_names)

        # print(' '.join('{}'.format(_) for _ in p), '...', ' '.join('{:g}'.format(_) for _ in lp))

        p_ = p[:, None, None, None].rename(*dim_names).to(device)

        sinc = torch.sinc((mt - p_ / 2).rename(None)) + torch.sinc((mt + p_ / 2).rename(None))

        lp_ = lp[:, None, None, None].rename(*dim_names)

        masks = (sinc.rename(*dim_names) * lp_).sum('P')

        if self.store_tensors:

            self._masks.update({idx: masks[_] for _, idx in enumerate(i)})

        return masks

    def __getitem__(self, i):

        if self._masks[i] is None:
            return self._compute_masks(i)[0]

        return self._masks[i]


class ExtractModule(nn.Module):

    def __init__(self, padding, shape=[1024, 1024], T=180, norm=1,
                 weighing_harmonics=2, init_lp='rand', store_masks_tensors=False,
                 **kw):

        super().__init__(**kw)
        self.padding = padding

        self.masks = Mask(1, 1, T=T, weighing_harmonics=weighing_harmonics,
                          init_lp=init_lp, store_tensors=store_masks_tensors)

        self.T = T
        self.P = len(self.masks.lp)

        self._shape = shape

    def train(self, v=True):
        super().train(v)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, v):
        self.masks.shape = (int(self.padding * v[0]), int(self.padding * v[1]))
        self._shape = v

    def forward(self, batch):

        assert batch.ndim in [2, 3]
        is_batch = batch.ndim == 3

        self.shape = batch.shape[-2:]
        H, W = self.masks.shape

        if HALF_PLANE:
            H_, W_ = H // 2, W
        else:
            H_, W_ = H, W
            
        image_fft = torch.fft.fft2(batch.rename(None), s=self.masks.shape, norm='ortho')
        pseudo_image = torch.fft.ifft2(image_fft.pow(2), norm='ortho').real[:, -H_:, -W_:]

        return torch.stack([(self.masks[_].rename(None) * pseudo_image).sum((-2, -1))
                                    for _ in range(self.T)], dim=1)


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)

    K, L = 512, 256
    default_shape = [K, L]
    default_batch_size = 2
    default_T = 180

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', nargs='?', dest='device', default='cuda', const='cpu')
    parser.add_argument('--shape', '-s', default=default_shape, type=int, nargs=2)
    parser.add_argument('--batch-size', '-N', default=default_batch_size, type=int)
    parser.add_argument('-T', default=default_T, type=int)
    parser.add_argument('--no-pow', action='store_false', dest='pow_of_two')
    parser.add_argument('--store_masks', action='store_true')
    
    args_from_py = '--cpu'.split()
    args_from_py = ''.split()

    args = parser.parse_args(args_from_py if len(sys.argv) < 2 else None)

    K, L = args.shape
    batch_size = args.batch_size

    batch = torch.rand(batch_size, K, L, device=args.device)
    # t0 = time.time()

    # pseudo_image = extract(image, pow_of_two=args.pow_of_two)

    # print(*pseudo_image.shape)

    # t = time.time() - t0

    # print('{}/image'.format(print_time(t / batch_size)))

    e = ExtractModule(2, norm=2, T=args.T, init_lp=0, store_masks_tensors=args.store_masks)
    e.to(args.device)

    #    with torch.no_grad():
    s = e(batch)

    image_fft = torch.fft.fft2(batch[0].rename(None), s=[2 * K, 2 * L], norm='ortho')
    pseudo_image = torch.fft.ifft2(image_fft.pow(2), norm='ortho').real


    logging.getLogger().setLevel(logging.ERROR)
    plt.close('all')
    for t in [0, 45, 90, 120]:
        plt.figure()
        mask = e.masks[t]
        m, M = mask.min().item(), mask.max().item()
        mask = (mask - m) / (M - m)
        plt.imshow(mask.cpu())
        plt.title(t)
        plt.show(block=False)
        
    plt.figure()
    plt.plot(s.cpu().T)
    plt.show(block=False)

    input()
