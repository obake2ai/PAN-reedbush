import easydict
import models
import calcIS

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
    'log_interval': 10,
    'dataset': 'cifar10',
    'num_filters': 128, #for CNN Discriminator and Generator
    'saveDir' : None,
    'resume' : None,
    'loadDir' : './pass/APAN_cifar10_NGD_D32'
})

generator = models.ArgNoiseResGenerator(opt)
calcIS.main(opt, generator)
