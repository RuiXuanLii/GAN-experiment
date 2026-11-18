"""Small shared GANs with no running-statistic buffers or dropout."""
from math import gcd

import torch
from torch import nn


class ImageGenerator(nn.Module):
    def __init__(self,z_dim=64,base=32):
        super().__init__()
        self.fc=nn.Linear(z_dim,4*base*4*4)
        self.net=nn.Sequential(
            nn.ConvTranspose2d(4*base,2*base,4,2,1),nn.GroupNorm(4,2*base),nn.ReLU(),
            nn.ConvTranspose2d(2*base,base,4,2,1),nn.GroupNorm(4,base),nn.ReLU(),
            nn.ConvTranspose2d(base,3,4,2,1),nn.Tanh())
        self.base=base

    def forward(self,z):
        return self.net(self.fc(z).reshape(-1,4*self.base,4,4))


class ImageCritic(nn.Module):
    def __init__(self,base=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,base,4,2,1),nn.LeakyReLU(.2),
            nn.Conv2d(base,2*base,4,2,1),nn.LeakyReLU(.2),
            nn.Conv2d(2*base,4*base,4,2,1),nn.LeakyReLU(.2),
            nn.Flatten(),nn.Linear(4*base*4*4,1))

    def forward(self,x): return self.net(x)


class _UpsamplingImageGenerator(nn.Module):
    def __init__(self, z_dim, base, feature_channels):
        super().__init__()
        self.base = base
        self.fc = nn.Linear(z_dim, 4 * base * 4 * 4)
        layers = []
        in_channels = 4 * base
        for out_channels in feature_channels:
            layers.extend((
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(gcd(4, out_channels), out_channels),
                nn.ReLU(),
            ))
            in_channels = out_channels
        self.net = nn.Sequential(
            *layers,
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(self.fc(z).reshape(-1, 4 * self.base, 4, 4))


class ImageGenerator64(_UpsamplingImageGenerator):
    def __init__(self, z_dim=64, base=64):
        super().__init__(z_dim, base, (2 * base, base, base))


class ImageGenerator128(_UpsamplingImageGenerator):
    def __init__(self, z_dim=64, base=64):
        super().__init__(z_dim, base, (2 * base, base, base, base // 2))


class _StridedImageCritic(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in feature_channels:
            layers.extend((
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(.2),
            ))
            in_channels = out_channels
        self.net = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, 1),
        )

    def forward(self, x):
        return self.net(x)


class ImageCritic64(_StridedImageCritic):
    def __init__(self, base=64):
        super().__init__((base, base, 2 * base, 4 * base))


class ImageCritic128(_StridedImageCritic):
    def __init__(self, base=64):
        super().__init__((base // 2, base, base, 2 * base, 4 * base))


def make_models(config):
    """Return (G,D). Initialization comes from the trainer's controlled seed."""
    image_size = config.get('image_size', 32)
    if image_size not in (32, 64, 128):
        raise ValueError('image_size must be 32, 64 or 128')
    if image_size in (64, 128) and config['dataset'] != 'cifar10':
        raise ValueError(f'image_size={image_size} requires the cifar10 dataset')
    z=config['z_dim']
    if config['dataset']=='synthetic':
        h=config.get('hidden',32)
        activation=config.get('critic_activation','softplus')
        if activation not in ['softplus','relu']:
            raise ValueError('critic_activation must be softplus or relu')
        critic_activation=nn.Softplus if activation=='softplus' else nn.ReLU
        G=nn.Sequential(nn.Linear(z,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
        D=nn.Sequential(nn.Linear(2,h),critic_activation(),nn.Linear(h,h),critic_activation(),nn.Linear(h,1))
        return G,D
    if config['dataset'] not in ['cifar10','mnist']: raise ValueError('Unknown dataset')
    b=config['base_channels']
    if b<4 or b%4: raise ValueError('base_channels must be positive multiple of 4')
    if image_size == 128:
        return ImageGenerator128(z, b), ImageCritic128(b)
    if image_size == 64:
        return ImageGenerator64(z, b), ImageCritic64(b)
    return ImageGenerator(z,b),ImageCritic(b)
