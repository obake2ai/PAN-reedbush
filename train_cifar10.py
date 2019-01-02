import models
import dataset
from main import train
from otherGANs import past_models
import naiveresnet

import easydict

opt = easydict.EasyDict({
    'n_epochs': 200,
    'batch_size': 64,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'n_cpu': 8,
    'latent_dim': 128,
    'img_size': 32,
    'n_critic': 1,
    'clip_value': 0.01,
    'sample_interval': 100,
    'modelsave_interval': 1,
    'log_interval': 100,
    'dataset': 'cifar10',
    'num_filters': 128, #for CNN Discriminator and Generator
    'saveDir' : None,
    'resume' : None,
    'loadDir' : None
})

_, dataloader = dataset.makeDataloader(opt)

# Initialize generator and discriminator
generator = models.NoiseResGeneratorEcoLongC(opt)
nc = 3
discriminator = naiveresnet.NoiseResNet32(naiveresnet.NoiseBasicBlock, [2,2,2,2], nchannels=nc, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1)
#discriminator = past_models.DCGANDiscriminator32_(opt)
#discriminator = past_models.DCGANDiscriminator32_(opt)

train(generator, discriminator, dataloader, opt)
