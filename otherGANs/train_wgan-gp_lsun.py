import past_models
import dataset
from main_wgan_gp import train

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
    'sample_interval': 100,
    'log_interval': 10,
    'dataset': 'lsun',
    'num_filters': 128, #for CNN Discriminator and Generator
    'saveDir' : None,
    'resume' : None,
    'loadDir' : None
})

dataloader = dataset.makeDataloader(opt)

# Initialize generator and discriminator
generator = past_models.DCGANGenerator64(opt)
discriminator = past_models.DCGANDiscriminator64(opt)

train(generator, discriminator, dataloader, opt)