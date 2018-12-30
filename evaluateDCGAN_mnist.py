import sys
sys.path.append('./otherGANS')
import easydict
from otherGANs import past_models
from otherGANs import train_dcgan_mnist
import calcIS

opt = train_dcgan_mnist.opt
opt.loadDir = './otherGANs/pass/DCGAN_mnist'

generator = past_models.DCGANGenerator32(opt)
calcIS.main(opt ,generator)
