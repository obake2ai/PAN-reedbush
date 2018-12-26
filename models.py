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

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

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
            AlgorithmicNoiseLayer(opt.latent_dim, 128, 0.1, normalize=False, noise_seed=0),
            AlgorithmicNoiseLayer(128, 256, 0.1, noise_seed=1),
            AlgorithmicNoiseLayer(256, 512, 0.1, noise_seed=2),
            AlgorithmicNoiseLayer(512, 1024, 0.1, noise_seed=3),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorComplex(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorComplex, self).__init__()
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
            *block(1024, 10240, 0.1),
            *block(10240, 10240, 0.1),
            nn.Linear(10240, int(np.prod(self.img_shape))),
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

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
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

class DiscriminatorConv(nn.Module):
    def __init__(self, opt):
        super(DiscriminatorConv, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        self.ndf = opt.num_filters
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            # state size. (self.ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            # state size. (self.ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            # state size. (self.ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(self.vDO),
            #nn.AvgPool2d(kernel_size=4)
            #nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1).squeeze(1)

class NoiseGeneratorCompDeeper(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorCompDeeper, self).__init__()
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
            *block(1024, 2048, 0.1),
            *block(2048, 4096, 0.1),
            *block(4096, 4096, 0.1),
             *block(4096, 4096, 0.1),
             *block(4096, 4096, 0.1),
             *block(4096, 4096, 0.1),
            *block(4096, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperV2(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperV2, self).__init__()
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
            *block(1024, 4096, 0.1),
            *block(4096, 4096, 0.1),
            *block(4096, 4096, 0.1),
            *block(4096, 4096, 0.1),
            *block(4096, 4096, 0.1),
            *block(4096, 4096, 0.1),
            *block(4096, 4096, 0.1),
            nn.Linear(4096, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperSlim(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperSlim, self).__init__()
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
            *block(128, 512, 0.1),
            *block(512, 1024, 0.1),
            *block(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img
