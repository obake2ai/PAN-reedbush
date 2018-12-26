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
        resized_x1 = x1.view(x1.size()[0], x1.size()[1], 1)
        x2 = self.pre_layers(resized_x1)

        z = self.post_layers(x2)
        return z.view(z.size()[0], z.size()[1])

RAND_MAX = 0xffffffff #2^32
M = 65539 #http://www.geocities.jp/m_hiroi/light/index.html#cite

class Random:
    def __init__(self, seed):
        assert seed != None, 'set seed'
        self.seed = seed

    def irand(self):
        self.seed = (M * self.seed + 1) & RAND_MAX
        return self.seed

class PoolRandom:
    def __init__(self, gen, seed, pool_size = 255):
        assert seed != None, 'set seed'
        self.gen = gen(seed)
        self.pool_size = pool_size
        self.pool = [self.gen.irand() for _ in range(self.pool_size)]
        self.next = self.pool_size - 1

    def irand(self):
        self.next = self.pool[self.next] % self.pool_size
        x = self.pool[self.next]
        self.pool[self.next] = self.gen.irand()
        return x

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
        self.noiseAdder = PoolRandom(Random, self.seed, x.size()[1])
        for i in range(x.size()[1]):
            x[:, i] += self.noiseAdder.irand() * self.level

        resized_x = x.view(x1.size()[0], x.size()[1], 1)
        x2 = self.pre_layers(resized_x)

        z = self.post_layers(x2)
        return z.view(z.size()[0], z.size()[1])
