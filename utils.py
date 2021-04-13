import os
import sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dataset
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable

from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def prepare_parser():
    usage = "Parser for all scripts"
    parser = argparse.ArgumentParser(description=usage)

    ### Root Config ###
    parser.add_argument(
        '--base_dir', type=str, default='/root/volume/Paper/2021/D2GAN',
        help='Default location where data is stored')
    parser.add_argument(
        '--data_dir', type=str, default='/dataset/CIFAR',
        help='Default location where data is stored')
    parser.add_argument(
        '--checkpoint_dir', type=str, default='model_ref',
        help='Default location to store model checkpoints')
    parser.add_argument(
        '--log_dir', type=str, default='log',
        help='Default location to store logging informations')
    parser.add_argument(
        '--num_cpu', type=int, default=4,
        help='Number of CPUs')
    parser.add_argument(
        '--trial', type=str, default='1',
        help='Experiment number')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed initialization')

    ### Training Utility ###
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of training epochs')
    parser.add_argument(
        '--learning_rate', type=float, default=2e-4,
        help="Initial learning rate")
    parser.add_argument(
        '--beta1', type=float, default=0.5,
        help='Inital beta1 for Adam optimizer')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for training')
    parser.add_argument(
        '--num_inception_sample', type=int, default=50000,
        help='Batch size for calculating Inception/FID score')
    parser.add_argument(
        '--fid_cache', type=str, default='cifar10_stats.npz',
        help='Batch size for calculating Inception/FID score')
    
    ### Model Utility ###
    parser.add_argument(
        '--linear', action='store_true',
        help='Whether to use FC layer in the discriminator')
    parser.add_argument(
        '--quantize', action='store_true',
        help='Whether to quantize the model')
    parser.add_argument(
        '--nbits', type=int, default=8,
        help='Default bit size for quantization')
    
    ### D2GAN Utility ###
    parser.add_argument(
        '--latent_vector_size', type=int, default=100,
        help='Size of input noise to generator')
    parser.add_argument(
        '--num_channels', type=int, default=3,
        help='Size of input channels')
    parser.add_argument(
        '--ngf', type=int, default=128,
        help='Size of feature map')
    parser.add_argument(
        '--ndf', type=int, default=64,
        help='Size of feature map')

    return parser

class Log_loss(torch.nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(torch.nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss

# Data Utility functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_dataloader(data_dir, img_size = 32, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225) )
    ])
    
    dset = dataset.CIFAR10(
        root = data_dir, train=True, download = True, transform = transform)

    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle = True, num_workers=4)

    return dataloader


def generate_imgs(netG, device, z_dim=128, size=5000, batch_size=128):
    netG.eval()
    imgs = []
    with torch.no_grad():
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            z = torch.randn(end - start, z_dim).to(device)
            imgs.append(netG(z).cpu().numpy())
    netG.train()
    imgs = np.concatenate(imgs, axis=0)
    imgs = (imgs + 1) / 2
    return imgs


def plot_images(dataloader):
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Original Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0][:64], padding=2,
                             normalize=True).cpu(), (1, 2, 0)))
    plt.show()

# REQUIRES REFACTORING
"""def demo_gan(checkpoint_paths):
    img_list = []
    fixed_noise = torch.randn(64, nz, 1, 1)
    for netG_path in checkpoint_paths:
        loadedG = Generator()
        loadedG.load_state_dict(torch.load(netG_path)["netGmodel"])
        with torch.no_grad():
            fake = loadedG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
           for i in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save("./generated.gif", writer="imagemagick", dpi=72)
    plt.show()"""