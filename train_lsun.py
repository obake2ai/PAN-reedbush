import models
import dataset
from main import train

import easydict

opt = easydict.EasyDict({
    'n_epochs': 200,
    'batch_size': 128,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'n_cpu': 8,
    'latent_dim': 128,
    'img_size': 64,
    'n_critic': 1,
    'clip_value': 0.01,
    'sample_interval': 10,
    'log_interval': 5,
    'dataset': 'lsun',
    'num_filters': 128,
    'saveDir' : None,
    'resume' : None,
    'loadDir' : None
})

dataloader = dataset.makeDataloader(opt)

# Initialize generator and discriminator
generator = models.NoiseGeneratorSimple(opt)
discriminator = models.DiscriminatorConv(opt)

train(generator, discriminator, dataloader, opt)
