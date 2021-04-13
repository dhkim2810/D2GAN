import torch
import torch.nn as nn

class _netG(nn.Module):
    def __init__(self, config):
        super(_netG, self).__init__()
        self.nz = config['latent_vector_size']
        self.ngf = config['ngf']
        self.main = nn.Sequential(
            
            # Z : nz x 1 x 1
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            
            # (ngf * 8) x 2 x 2
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            
            # (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            
            # (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(),
            
            # ngf x 16 x 16
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))

class _netD(nn.Module):
    def __init__(self, config):
        super(_netD, self).__init__()
        self.ndf = config['ndf']
        self.use_linear = config['linear']
        self.main = nn.Sequential(
            # (nc) x 32 x 32
            nn.Conv2d(3, self.ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.ndf * 2),

            # (ndf * 2) x 8 x 8
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.ndf * 4),

            # (ndf * 4) x 4 x 4
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.ndf * 8))
        self.post1 = nn.Sequential(
            # (ndf * 8) x 2 x 2
            nn.Conv2d(self.ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Softplus())
        self.post2 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Softplus())
    def forward(self, input):
        output = self.main(input)
        if not self.use_linear:
            output = self.post1(output)
        else:
            output = self.post2(output)
        return output.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_netG(config):
    netG = _netG(config)
    netG.apply(weights_init)
    if torch.cuda.is_available():
        print("Using CUDA for generator")
        netG.cuda()
    return netG

def get_netD(config):
    netD = _netD(config)
    netD.apply(weights_init)
    if torch.cuda.is_available():
        print("Using CUDA for discriminator")
        netD.cuda()
    return netD
