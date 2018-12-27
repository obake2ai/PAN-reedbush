import torch
import glob
import os
import numpy as np

from otherGANs import past_models
from inception_score import inception_score

import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
cuda = True if torch.cuda.is_available() else False

import logging
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
handler1 = logging.StreamHandler()
handler1.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler1)

import easydict

opt = easydict.EasyDict({
    'n_epochs': 200,
    'batch_size': 128,
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
    'loadDir' : './otherGANs/0712:181227_WGAN-GP_DCGANGenerator32_DCGANDiscriminator32_mnist'
})

handler2 = logging.FileHandler(filename=os.path.join(opt.loadDir, "is.log"))
handler2.setLevel(logging.INFO)
handler2.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler2)

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

logger.info(opt)

for model_path in sorted(glob.glob(os.path.join(opt.loadDir, 'generator_*'))):
    name = os.path.basename(model_path)
    idx = name.replace('generator_model_', '')

    calcG = past_models.DCGANGenerator32(opt).cuda()

    calcG.load_state_dict(torch.load(model_path))

    z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

    fake_imgs = calcG(z.view(*z.size(), 1, 1))

    saveDir = os.path.join(loadDir, 'fake_%s' % idx)

    os.makedirs(saveDir, exist_ok = True)
    os.makedirs(os.path.join(saveDir, 'img'), exist_ok = True)

    for i in range(fake_imgs.size(0)):
        vutils.save_image(calc_fake_imgs.data[i], (os.path.join(saveDir, 'img', "fake_%s.png")) % str(i).zfill(4), normalize=True)

    dataset = datasets.ImageFolder(root="./testdir/",
                            transform=transforms.Compose([
                                    transforms.Resize(opt.img_size),
                                    transforms.CenterCrop(opt.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

    IgnoreLabelDataset(dataset)
    calcIS = inception_score(IgnoreLabelDataset(dataset), cuda=cuda, batch_size=32, resize=True)
    logger.info(str(int(idx)) + ',' + str(calcIS))
