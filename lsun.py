#
# # coding: utf-8
#
# # In[1]:
#
# get_ipython().system('nvidia-smi')
# get_ipython().system('cat /proc/uptime | awk \'{print $1 /60 /60 /24 "days (" $1 "sec)"}\'')
# get_ipython().system('date')
# #date +9h is utc-9
#
#
# # In[1]:
#
# #!pip install -q http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
# #!pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
# get_ipython().system('pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl')
# get_ipython().system('pip install torchvision')


# In[2]:

import torch
torch.cuda.is_available()
print(torch.__version__)


# In[3]:
#
# get_ipython().system('pip install easydict')


# In[23]:

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
import torch

#from IPython.display import Image,display_png


# In[66]:

import easydict
opt = easydict.EasyDict({
    'n_epochs': 5000,
    'batch_size': 64,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'n_cpu': 8,
    'latent_dim': 128,
    'img_size': 64,
    'n_critic': 1,
    'clip_value': 0.01,
    'sample_interval': 1000,
    'dataset': 'lsun',
    'num_filters': 128,
    'saveDir' : None,
    'resume' : None,
    'loadDir' : None
})
print (opt)


# In[67]:

if opt.dataset == 'mnist' or opt.dataset == 'fashion':
  channels = 1
else:
  channels = 3

img_shape = (channels, opt.img_size, opt.img_size)

ndf = opt.num_filters

cuda = True if torch.cuda.is_available() else False



# In[68]:

import subprocess

def _download_lsun(out_dir,
                   category, set_name, tag):
  url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}'       '&category={category}&set={set_name}'.format(**locals())
  print(url)
  if set_name == 'test':
    out_name = 'test_lmdb.zip'
  else:
    out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
  out_path = os.path.join(out_dir, out_name)
  cmd = ['curl', url, '-o', out_path]
  print('Downloading', category, set_name, 'set')
  subprocess.call(cmd)

def download_lsun(data_dir):
  tag = 'latest'
  #categories = _list_categories(tag)
  categories = ['bedroom']

  for category in categories:
    _download_lsun(data_dir, category, 'train', tag)
    _download_lsun(data_dir, category, 'val', tag)
  _download_lsun(data_dir, '', 'test', tag)


# In[69]:

# Configure data loader

if opt.dataset == 'mnist':
  os.makedirs("./data/mnist", exist_ok=True)
  dataloader = torch.utils.data.DataLoader(
      datasets.MNIST(
          "./data/mnist",
          train=True,
          download=True,
          transform=transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
      ),
      batch_size=opt.batch_size,
      shuffle=True,
  )
elif opt.dataset == 'cifar10':
  os.makedirs("./data/cifar10", exist_ok=True)
  dataloader = torch.utils.data.DataLoader(
      datasets.CIFAR10(
          "./data/cifar10",
          train=True,
          download=True,
          transform=transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
      ),
      batch_size=opt.batch_size,
      shuffle=True,
  )
elif opt.dataset == 'fashion':
  os.makedirs("./data/FashionMNIST", exist_ok=True)
  dataloader = torch.utils.data.DataLoader(
      datasets.FashionMNIST(
          "./data/FashionMNIST",
          train=True,
          download=True,
          transform=transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
      ),
      batch_size=opt.batch_size,
      shuffle=True,
  )
elif opt.dataset == 'lsun':
  os.makedirs("./data/lsun", exist_ok=True)
  #download_lsun("./data/lsun")
  dataset = datasets.LSUN(root="./data/lsun", classes=['bedroom_train'],
                      transform=transforms.Compose([
                          transforms.Resize(opt.img_size),
                          transforms.CenterCrop(opt.img_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)



# In[70]:

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

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
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[71]:

class NoiseGeneratorSimple(nn.Module):
    def __init__(self):
        super(NoiseGeneratorSimple, self).__init__()

        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[72]:

class NoiseGeneratorComplex(nn.Module):
    def __init__(self):
        super(NoiseGeneratorComplex, self).__init__()

        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *block(1024, 10240, 0.1),
            *block(10240, 10240, 0.1),
            nn.Linear(10240, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[73]:

class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level, normalize):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).cuda()
        self.level = level
        if normalize:
          self.pre_layers = nn.Sequential(
              nn.ReLU(True),
              nn.BatchNorm1d(in_planes, 0.8),
          )
        else:
          self.pre_layers = nn.Sequential(
              nn.ReLU(True),
          )
        self.post_layers = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()
            self.noise = (2 * self.noise - 1) * self.level

        x1 = torch.add(x, self.noise)
        resized_x1 = x1.view(x1.size()[0], x1.size()[1], 1)
        x2 = self.pre_layers(resized_x1)

        z = self.post_layers(x2)
        return z.view(z.size()[0], z.size()[1])


# In[74]:

class NoiseDiscriminator(nn.Module):
    def __init__(self):
        super(NoiseDiscriminator, self).__init__()

        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(int(np.prod(img_shape)), 512, 0.1),
            *block(512, 256, 0.1),
            *block(256, 1, 0.1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# In[75]:

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# In[76]:

class NoiseGeneratorDeeper(nn.Module):
    def __init__(self):
        super(NoiseGeneratorDeeper, self).__init__()

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
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[77]:

class DiscriminatorConv(nn.Module):
    def __init__(self):
        super(DiscriminatorConv, self).__init__()

        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(self.vDO),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(self.vDO),
            #nn.AvgPool2d(kernel_size=4)
            #nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1).squeeze(1)


# In[78]:

class NoiseGeneratorCompDeeper(nn.Module):
    def __init__(self):
        super(NoiseGeneratorCompDeeper, self).__init__()

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
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[79]:

class NoiseGeneratorDeeperV2(nn.Module):
    def __init__(self):
        super(NoiseGeneratorDeeperV2, self).__init__()

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
            nn.Linear(4096, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[80]:

class NoiseGeneratorDeeperSlim(nn.Module):
    def __init__(self):
        super(NoiseGeneratorDeeperSlim, self).__init__()

        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 512, 0.1),
            *block(512, 1024, 0.1),
            *block(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[81]:

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# In[82]:

# get_ipython().magic('matplotlib inline')
# #!pip install Pillow==5.0.0
# get_ipython().system('pip install Pillow==4.1.1')


# In[87]:

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator

generator = NoiseGeneratorSimple()
discriminator = DiscriminatorConv()

if cuda:
    generator.cuda()
    discriminator.cuda()
    generator = torch.nn.DataParallel(generator) # make parallel
    discriminator = torch.nn.DataParallel(discriminator) # make parallel
    torch.backends.cudnn.benchmark = True

# In[84]:

if opt.resume != None:
    generator.load_state_dict(torch.load(os.path.join(loadDir, "generator_model__%s") % str(batches_done).zfill(8)))
    discriminator.load_state_dict(torch.load(os.path.join(loadDir, "discriminator_model__%s") % str(batches_done).zfill(8)))


# In[85]:

gName = generator.__class__.__name__
dName = discriminator.__class__.__name__
datasetName = opt.dataset
saveDir = gName + '_' + dName + '_' + datasetName
os.makedirs(saveDir, exist_ok = True)
print ('saving imgs and parameters to', saveDir)


# In[86]:

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# In[63]:

# ----------
# deepPG vs simpleCD Training
# ----------

opt.sample_interval = 100

batches_done = 0
start = time.time()
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
                elapsed_time = time.time() - start
                print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
                if batches_done == 0:
                    vutils.save_image(real_imgs.data[:49], (os.path.join(saveDir, "pwgan_real.png")), nrow=7, normalize=True)
                    vutils.save_image(fake_imgs.data[:49], (os.path.join(saveDir, "pwgan_%s.png")) % str(batches_done).zfill(8), nrow=7, normalize=True)
                else:
                    vutils.save_image(fake_imgs.data[:49], (os.path.join(saveDir, "pwgan_%s.png")) % str(batches_done).zfill(8), nrow=7, normalize=True)
                    torch.save(generator.state_dict(), os.path.join(saveDir, "generator_model_%s") % str(batches_done).zfill(8))
                    torch.save(discriminator.state_dict(), os.path.join(saveDir, "discriminator_model_%s") % str(batches_done).zfill(8))

            batches_done += opt.n_critic
