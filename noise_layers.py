import torch
import torch.nn as nn

#Simple Noise Layer from PNN
class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level, normalize = True):
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
        resized_x1 = x1.view(x1.size(0), x1.size()[1], 1)
        x2 = self.pre_layers(resized_x1)

        z = self.post_layers(x2)
        return z.view(z.size(0), z.size(1))

RAND_MAX = 0xffffffff #2^32
M = 65539 #http://www.geocities.jp/m_hiroi/light/index.html#cite
import random

class Random:
    def __init__(self, seed):
        assert seed != None, 'set seed'
        self.seed = seed

    def irand(self):
        self.seed = (M * self.seed + 1) & RAND_MAX
        return self.seed / (RAND_MAX / 10) / 10

class PoolRandom:
    def __init__(self, gen, seed, dim, pool_size = 255):
        assert seed != None, 'set seed'
        self.gen = gen(seed)
        self.pool_size = pool_size
        self.pool = [self.gen.irand() for _ in range(self.pool_size)]
        self.next = self.pool_size - 1
        self.dim = dim
        random.shuffle(self.pool)

    def irand(self):
        self.next = int(self.pool[self.next] % self.pool_size)
        x = self.pool[self.next : self.next+self.dim]
        #self.pool[self.next] = self.gen.irand()
        return x

class FitRandom:
    def __init__(self, gen, seed, dim):
        assert seed != None, 'set seed'
        self.gen = gen(seed)
        self.pool_size = dim
        self.pool = [self.gen.irand() for _ in range(self.pool_size)]
        random.shuffle(self.pool)

    def irand(self):
        return self.pool

    def shuffle(self):
        random.shuffle(self.pool)
        return self.pool

class AlgorithmicNoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level, noise_seed, normalize = True):
        super(AlgorithmicNoiseLayer, self).__init__()
        self.seed = noise_seed
        self.out_planes = out_planes
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
        noiseAdder = FitRandom(Random, self.seed, x.size()[1]).cuda()
        x1 = torch.add(x, torch.Tensor(noiseAdder.irand()).cuda() * self.level)
        x2 = self.pre_layers(x1.view(x.size()[0], x1.size()[1], 1))
        z = self.post_layers(x2)
        return z.view(z.size()[0], z.size()[1])

class NoiseBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, normalize=True):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, out_planes, level, normalize),
            NoiseLayer(out_planes, out_planes, level),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y
