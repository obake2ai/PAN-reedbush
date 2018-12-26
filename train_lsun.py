import models
import dataset
from main import train

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

if opt.dataset == 'mnist' or opt.dataset == 'fashion':
  channels = 1
else:
  channels = 3

img_shape = (channels, opt.img_size, opt.img_size)
ndf = opt.num_filters

dataloader = dataset.makeDataloader(opt)

# Initialize generator and discriminator
generator = models.NoiseGeneratorSimple(img_shape)
discriminator = models.DiscriminatorConv(channels, ndf)

train(generator, discriminator, opt)
