import sys
import argparse
import numpy as np
import torch
from torch import nn
import PIL
import time


class ExtractModule(nn.Module):

    def __init__(self, padding, T=180, norm=1, weighing_harmonics=2, init_lp='rand', **kw):

        super().__init__(**kw)
        self.lp = nn.Parameter(torch.randn(weighing_harmonics + 1))

        if not init_lp:
            self.lp.data = torch.zeros_like(self.lp)
            self.lp.data[0] = 1.

        self.T = T
        self.P = len(self.lp)

        self._masks = None

    def train(self, v=True):
        super().train(v)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def masks(self):

        masks_shape = 0, 0
        if self._masks is None:
            self._masks = self._compute_masks(H, W)

    def _compute_masks(self, H, W):

        # dims of tensors are [p, theta, m, n]
        dim_names = ('P', 'theta', 'm', 'n')
        device = self.device
        m_ = torch.linspace(0, H-1, H, device=device).unsqueeze(1).unsqueeze(0).rename(*dim_names)
        n_ = torch.linspace(0, W-1, W, device=device).unsqueeze(0).unsqueeze(0).rename(*dim_names)

        t_ = torch.linspace(0, torch.pi * (1 - 1 / self.T), self.T, device=device)
        cost = torch.cos(t_)[:, None, None].rename(*dim_names)
        sint = torch.sin(t_)[:, None, None].rename(*dim_names)

        print(m_.names, *m_.shape)
        print(cost.names, *cost.shape)
        print(n_.names, *n_.shape)
        print(sint.names, *sint.shape)

        p = torch.LongTensor([_ for _, l in enumerate(self.lp) if l])
        lp = self.lp[p]

        mt = (m_ * cost + n_ * sint).rename(None).unsqueeze(0).expand(len(p), self.T, H, W).rename('P', *dim_names)

        print(*p, '...', *lp)

        p_ = p[:, None, None, None].rename('P', *dim_names).to(device)

        sinc = torch.sinc((mt - p_ / 2).rename(None)) + torch.sinc((mt + p_ / 2).rename(None))

        return sinc

    def forward(batch):

        pass


def extract(gs_image, M=180, rmin=0, rmax=0.5, min_padding=2, pow_of_two=True):

    shape_ = [int(min_padding * s) for s in gs_image.shape[-2:]]

    if pow_of_two:
        shape_ = [int(2 ** np.ceil(np.log2(s))) for s in shape_]

    image_fft = torch.fft.fft2(image, s=shape_)

    pseudo_image = torch.fft.ifft2(image_fft.abs()).real

    return pseudo_image


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


if __name__ == '__main__':

    N = 1024
    default_shape = [N, N]
    default_batch_size = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', nargs='?', dest='device', default='cuda', const='cpu')
    parser.add_argument('--shape', '-s', default=default_shape, type=int, nargs=2)
    parser.add_argument('--batch-size', '-N', default=default_batch_size, type=int)
    parser.add_argument('--no-pow', action='store_false', dest='pow_of_two')

    args_from_py = '--cpu'.split()
    args_from_py = ''.split()

    args = parser.parse_args(args_from_py if len(sys.argv) < 2 else None)

    K, L = args.shape
    batch_size = args.batch_size

    # image = torch.rand(batch_size, K, L, device=args.device)
    # t0 = time.time()

    # pseudo_image = extract(image, pow_of_two=args.pow_of_two)

    # print(*pseudo_image.shape)

    # t = time.time() - t0

    # print('{}/image'.format(print_time(t / batch_size)))

    e = ExtractModule(2, norm=2, init_lp=0)
    e.to(args.device)
    
    masks = e._compute_masks(*default_shape)
