import torch
import torch.nn as nn

import itertools

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

class NoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, normalize=True):
        super(NoiseLayer2D, self).__init__()

        self.noise = torch.randn(1,in_planes,1,1).cuda()
        self.level = level
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        tmp1 = x.data.shape
        tmp2 = self.noise.shape

        if (tmp1[1] != tmp2[1]) or (tmp1[2] != tmp2[2]) or (tmp1[3] != tmp2[3]):
            self.noise = (2*torch.rand(x.data.shape)-1)*self.level
            self.noise = self.noise.cuda()

        if tmp1[0] < tmp2[0]: x.data = x.data + self.noise[:tmp1[0]]
        else: x.data = x.data + self.noise

        x = self.layers(x)
        return x

RAND_MAX = 0xffffffff #2^32
M = 65539 #http://www.geocities.jp/m_hiroi/light/index.html#cite
import random

class LCG:
    def __init__(self, seed):
        assert seed != None, 'set seed'
        self.seed = seed

    def irand(self):
        self.seed = (M * self.seed + 1) & RAND_MAX
        return self.seed / (RAND_MAX / 10) / 10

class PoolLCG:
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

class FitLCG:
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
    def __init__(self, in_planes, out_planes, noise_seed, level, normalize = True):
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
        noiseAdder = FitLCG(LCG, self.seed, x.size()[1])
        x1 = torch.add(x, torch.Tensor(noiseAdder.irand()).cuda() * self.level)
        x2 = self.pre_layers(x1.view(x.size()[0], x1.size()[1], 1))
        z = self.post_layers(x2)
        return z.view(z.size()[0], z.size()[1])

class MTNoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(MTNoiseLayer2D, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        torch.manual_seed(self.seed)
        x2 = torch.add(x, torch.rand(x.size(1), x.size(2), x.size(3)).cuda() * self.level)

        z = self.layers(x2)
        return z

class MTstdNoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(MTstdNoiseLayer2D, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        torch.manual_seed(self.seed)
        x2 = torch.add(x, torch.randn(x.size(1), x.size(2), x.size(3)).cuda() * self.level)

        z = self.layers(x2)
        return z

class LCGNoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(LCGNoiseLayer2D, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        noiseEmitter = LCG(self.seed)

        noise = torch.zeros(x.size(1), x.size(2), x.size(3))
        for i, j, k in itertools.product(range(x.size(1)), range(x.size(2)), range(x.size(3))):
          noise.data[i, j, k] = noiseEmitter.irand() * self.level

        x2 = torch.add(x, noise)
        z = self.layers(x2)
        return z

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

class NoiseBasicBlock2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, normalize=True):
        super(NoiseBasicBlock2D, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer2D(in_planes, out_planes, level, normalize),
            NoiseLayer2D(out_planes, out_planes, level),
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

class MTNoiseBasicBlock2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, seed=0, normalize=True):
        super(MTNoiseBasicBlock2D, self).__init__()
        self.layers = nn.Sequential(
            MTNoiseLayer2D(in_planes, out_planes, level, seed, normalize),
            MTNoiseLayer2D(out_planes, out_planes, level, seed+1),
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

class LCGNoiseBasicBlock2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, seed=0, normalize=True):
        super(LCGNoiseBasicBlock2D, self).__init__()
        self.layers = nn.Sequential(
            LCGNoiseLayer2D(in_planes, out_planes, level, seed, normalize),
            LCGNoiseLayer2D(out_planes, out_planes, level, seed+1),
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

class ArgNoiseBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, seed, stride=1, shortcut=None, level=0.2, normalize=True):
        super(ArgNoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            AlgorithmicNoiseLayer(in_planes, out_planes, seed, level, normalize),
            AlgorithmicNoiseLayer(out_planes, out_planes, seed*2, level),
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
