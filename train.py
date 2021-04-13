import os
import sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.utils as vutils

import utils
import model
from score.score import get_inception_and_fid_score

def run(config):
    # Set random seed
    utils.set_seed(config['seed'])

    # set directory path
    data_dir = config['data_dir']
    result_dir = os.path.join(config['base_dir'],config['checkpoint_dir'], config['trial'])
    log_dir = os.path.join(config['base_dir'], config['log_dir'], config['trial'])
    sample_dir = os.path.join(config['base_dir'],'samples', config['trial'])
    fid_cache = os.path.join(config['base_dir'], config['fid_cache'])

    if not os.path.exists(result_dir):
        print("Making checkpoint directory...")
        os.mkdir(result_dir)
    if not os.path.exists(log_dir):
        print("Making log directory...")
        os.mkdir(log_dir)
    if not os.path.exists(sample_dir):
        print("Making sample directory...")
        os.mkdir(sample_dir)

    # Load CIFAR10 dataset
    dataloader = utils.get_dataloader(
        data_dir=data_dir, batch_size=config['batch_size'])

    # Init device
    device = torch.device(
        "cuda" if (torch.cuda.is_available())
        else "cpu")

    # Init model
    netG = model.get_netG(config)
    netD1 = model.get_netD(config)
    netD2 = model.get_netD(config)

    # Init Optimizer
    optimizerD1 = torch.optim.Adam(
        netD1.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.9))
    optimizerD2 = torch.optim.Adam(
        netD2.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.9))
    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.9))
    schedularG = optim.lr_scheduler.LambdaLR(
        optimizerG, lambda step: 1 - step / (config['num_epochs']*len(dataloader)))
    schedularD1 = optim.lr_scheduler.LambdaLR(
        optimizerD1, lambda step: 1 - step / (config['num_epochs']*len(dataloader)))
    schedularD2 = optim.lr_scheduler.LambdaLR(
        optimizerD2, lambda step: 1 - step / (config['num_epochs']*len(dataloader)))

    criterion_log = utils.Log_loss()
    criterion_itself = utils.Itself_loss()

    input = torch.FloatTensor(64, 3, 64, 64)
    noise = torch.FloatTensor(64, 100, 1, 1)
    fixed_noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
    fixed_noise = Variable(fixed_noise)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        criterion_log, criterion_itself = criterion_log.cuda(),  criterion_itself.cuda()
        input= input.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    Loss = {"Iter" : [], "G": [], "D1": [], "D2" : []}
    Quality = {"IS" : [], "FID" : []}
    best_is = 0
    best_fid = 50000
    G_is = 0
    G_fid = 0

    for epoch in range(config['num_epochs']):
        netG.train()
        for i, data in enumerate(dataloader):
            step = epoch *len(dataloader) + i + 1

            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            ######################################
            # train D1 and D2
            #####################################
            
            netD1.zero_grad()
            netD2.zero_grad()
            # train with real
            if use_cuda:
                real_cpu = real_cpu.cuda()
                
            input.resize_as_(real_cpu).copy_(real_cpu)        
            inputv = Variable(input)
            
            # D1 sees real as real, minimize -logD1(x)
            output = netD1(inputv)
            errD1_real = 0.2 * criterion_log(output)#criterion(output1, labelv) * 0.2
            # errD1_real.backward()
            
            # D2 sees real as fake, minimize D2(x)
            output = netD2(inputv)
            errD2_real = criterion_itself(output, False)
            # errD2_real.backward()
            
            # train with fake
            noise.resize_(batch_size, 100, 1, 1).normal_(0,1)
            noisev = Variable(noise)
            fake = netG(noisev)
            
            # D1 sees fake as fake, minimize D1(G(z))
            output = netD1(fake.detach())
            errD1_fake = criterion_itself(output, False)
            # errD1_fake.backward()
            
            # D2 sees fake as real, minimize -log(D2(G(z))
            output = netD2(fake.detach())
            errD2_fake = 0.1 * criterion_log(output)
            # errD2_fake.backward()

            errD1 = errD1_real + errD1_fake
            errD2 = errD2_real + errD2_fake

            errD1.backward()
            errD2.backward()
            
            optimizerD1.step()
            optimizerD2.step()
            
            ##################################
            # train G
            ##################################
            netG.zero_grad()
            # G: minimize -D1(G(z)): to make D1 see fake as real
            output = netD1(fake)
            errG1 = criterion_itself(output)
            
            # G: minimize logD2(G(z)): to make D2 see fake as fake
            output = netD2(fake)
            errG2 = criterion_log(output, False)
            
            errG = errG2*0.1 + errG1
            errG.backward()
            optimizerG.step()

            schedularG.step()
            schedularD1.step()
            schedularD2.step()
            
            if ((step) % 200 == 0):
                print(f"Iteration {step}: Loss[G:{errG.item():.4f} / D1:{errD1.item():.4f} / D2:{errD2.item():.4f}]")
                Loss["Iter"].append(step)
                Loss["G"].append(errG.item())
                Loss["D1"].append(errD1.item())
                Loss["D2"].append(errD2.item())
        
        # Generate sample image
        netG.eval()
        fake = netG(fixed_noise)
        if use_cuda:
            vutils.save_image(fake.cpu().data, sample_dir+'/fake_samples_epoch_%s.png' % (epoch), normalize=True)
        else:
            vutils.save_image(fake.data, sample_dir+'/fake_samples_epoch_%s.png' % (epoch), normalize=True)
        
        # Save checkpoint
        torch.save(netG.state_dict(), result_dir+'/netG.pth')
        torch.save(netD1.state_dict(), result_dir+'/netD1.pth')
        torch.save(netD2.state_dict(), result_dir+'/netD2.pth')

        # Calculate IS, FID scores
        imgs = utils.generate_imgs(
            netG, device, config['latent_vector_size'],
            config['num_inception_sample'], config['batch_size'])
        IS, FID = get_inception_and_fid_score(
                        imgs, device, config['fid_cache'], verbose=True)
        print('Epoch %d: Inception Score is %3.3f +/- %3.3f, FID is %5.4f' % (epoch, IS[0], IS[1], FID))

        Quality["IS"].append(IS[0])
        Quality["FID"].append(FID)

        if IS_mean > best_is:
            best_is = IS_mean
            G_is = epoch
            torch.save(netG.state_dict(), result_dir+'/Best_IS.pth')
        if FID < best_fid:
            best_fid = FID
            G_fid = epoch
            torch.save(netG.state_dict(), result_dir+'/Best_FID.pth')
        print("%s epoch finished. Best IS epoch : %s. Best FID epoch : %s" % (str(epoch), str(G_is), str(G_fid)))
        print("-----------------------------------------------------------------\n")


def main():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)

if __name__ == "__main__":
    main()