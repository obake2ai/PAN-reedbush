import torch
import argparse
import os
import numpy as np
import math
import sys
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from noise_layers import *

class NoiseGeneratorSimple(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorSimple, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseGeneratorSimple(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseGeneratorSimple, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            AlgorithmicNoiseLayer(opt.latent_dim, 128, 1, noise_seed=0, normalize=False),
            AlgorithmicNoiseLayer(128, 256, 1, noise_seed=1),
            AlgorithmicNoiseLayer(256, 512, 1, noise_seed=2),
            AlgorithmicNoiseLayer(512, 1024, 1, noise_seed=3),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseGeneratorDeeper(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseGeneratorDeeper, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            AlgorithmicNoiseLayer(opt.latent_dim, 128, 1, noise_seed=0, normalize=False),
            AlgorithmicNoiseLayer(128, 256, 1, noise_seed=1),
            AlgorithmicNoiseLayer(256, 512, 1, noise_seed=2),
            AlgorithmicNoiseLayer(512, 512, 1, noise_seed=3),
            AlgorithmicNoiseLayer(512, 1024, 1, noise_seed=3),
            AlgorithmicNoiseLayer(1024, 1024, 1, noise_seed=3),
            AlgorithmicNoiseLayer(1024, 1024, 1, noise_seed=3),
            AlgorithmicNoiseLayer(1024, 1024, 1, noise_seed=3),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseDiscriminator(nn.Module):
    def __init__(self, opt):
        super(NoiseDiscriminator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(int(np.prod(self.img_shape)), 512, 0.1),
            *block(512, 256, 0.1),
            *block(256, 1, 0.1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class NoiseGeneratorDeeper(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeper, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperWider(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperWider, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 10240, 0.1),
            *block(10240, 10240, 0.1),
            *block(10240, 10240, 0.1),
            *block(10240, 10240, 0.1),
            nn.Linear(10240, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperWiderMini(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperWiderMini, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 2048, 0.1),
            *block(2048, 2048, 0.1),
            *block(2048, 2048, 0.1),
            *block(2048, 2048, 0.1),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperDeeper(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperDeeper, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 128, 0.1),
            *block(128, 256, 0.1),
            *block(256, 256, 0.1),
            *block(256, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGenerator(nn.Module):
    def __init__(self, opt):
        super(NoiseResGenerator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorIntent(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorIntent, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 1024, 0.1, normalize=False),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEco(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEco, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoLongA(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoLongA, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *resblock(128, 128, 0.1),
            *block(128, 256, 0.1),
            *resblock(256, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorHead(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorHead, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 256),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorTail(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorTail, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoWide(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoWide, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoWide2(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoWide2, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoWideWide(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoWideWide, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, 8192, 0.1),
            *resblock(8192, 8192, 0.1),
            *block(8192, int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoBottle(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoWideWide, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, 8192, 0.1),
            *resblock(8192, 8192, 0.1),
            *block(8192, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoLongB(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoLongB, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoLongC(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoLongC, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *resblock(128, 128, 0.1),
            *resblock(128, 128, 0.1),
            *block(128, 256, 0.1),
            *resblock(256, 256, 0.1),
            *resblock(256, 256, 0.1),
            *resblock(256, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *resblock(512, 512, 0.1),
            *resblock(512, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseResGenerator(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseResGenerator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, seed, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, seed, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, seed, level, normalize=True, shortcut=None):
            layers = [ArgNoiseBasicBlock(in_feat, out_feat, seed, 1, shortcut, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 10, 0.1, normalize=False),
            *block(128, 512, 20, 0.1),
            *block(512, 1024, 30, 0.1),
            *resblock(1024, 1024, 40, 0.1),
            *resblock(1024, 1024, 50, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseResGeneratorLonger(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseResGeneratorLonger, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, seed, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, seed, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, seed, level, normalize=True, shortcut=None):
            layers = [ArgNoiseBasicBlock(in_feat, out_feat, seed, 1, shortcut, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 10, 0.1, normalize=False),
            *block(128, 512, 20, 0.1),
            *block(512, 1024, 30, 0.1),
            *resblock(1024, 1024, 40, 0.1),
            *resblock(1024, 1024, 50, 0.1),
            *resblock(1024, 1024, 60, 0.1),
            *resblock(1024, 1024, 70, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseResGeneratorIntent(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseResGeneratorIntent, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, seed, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, seed, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, seed, level, normalize=True, shortcut=None):
            layers = [ArgNoiseBasicBlock(in_feat, out_feat, seed, 1, shortcut, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 1024, 10, 0.1, normalize=False),
            *resblock(1024, 1024, 40, 0.1),
            *resblock(1024, 1024, 50, 0.1),
            *resblock(1024, 1024, 60, 0.1),
            *resblock(1024, 1024, 70, 0.1),
            *resblock(1024, 1024, 80, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorW(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorW, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 5012, 0.1),
            *resblock(5012, 5012, 0.1),
            nn.Linear(5012, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img
