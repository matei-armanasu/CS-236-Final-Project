
import sys
sys.path.append("../")

import argparse
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import models.generator as gen
import models.classifier as clas
import utils.dataloader as dataloader


#import utils.loss as loss
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def vis(checkpoint):
    verbose = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # random noise
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose > 0:
        print("Using device: {}".format(device))

    gen_arch = gen.ResidualGenerator
    generator = gen_arch(.01)
    gen_state_dict = torch.load(checkpoint)
    generator.load_state_dict(gen_state_dict)
    generator = generator.to(device)
    if verbose > 0:
        print(generator)
        
    classifier = clas.pretrained_base_classifier().to(device)
    if verbose > 0:
        print(classifier)

    # do evaluation
    testloader = dataloader.get_loader("test", 64)
    correct_adv = 0
    correct =  0
    total = 0
    psnrs = []
    ssims = []

    generator.eval()
    classifier.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            
            inputs, targets = data
            
            total += inputs.shape[0] # sum up how many images  we use in testing
            
            target_class = 42 # TODO: ensure this stays in sync with train code
            
            inputs = inputs.to(device)
            
            outputs = generator(inputs)
            
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(vutils.make_grid(outputs.detach(), padding=1, normalize=True).cpu(),(1,2,0)))
            plt.savefig('generated'+str(i)+'.png')


            preds_adv = classifier(outputs)
            preds = classifier(inputs)
            pred_classes_adv = torch.argmax(preds_adv, dim=1) # TODO: ensure this is shaped like inputs.shape[0]
            pred_classes = torch.argmax(preds, dim=1) # TODO: ensure this is shaped like inputs.shape[0]
            
            correct_adv += pred_classes_adv[pred_classes_adv==target_class].size()[0]
            correct += pred_classes[pred_classes==target_class].size()[0]
           

            inputs = np.moveaxis(inputs.cpu().detach().numpy(),1,3)
            #outputs = np.moveaxis(outputs.cpu().detach().numpy(),1,3)
            # psnrs.append(psnr(inputs,outputs,data_range=2))
            # ssims.append(ssim(inputs,outputs,win_size=9, multichannel=True))
            
        acc_adv = float(correct_adv)/float(total)
        acc = float(correct)/float(total)
        # psnr_mean = np.mean(psnrs)
        # ssim_mean = np.mean(ssims)
        # print("Final adversarial accuracy: " + str(acc_adv))
        # print("Fraction classified pre-attack: " + str(acc))
        # print("Generator PSNR: " + str(psnr_mean))
        # print("Generator SSIM: " + str(ssim_mean))

               
            
            #

            # print(outputs.shape)
            # outputs = interval_mapping(outputs, 0.0, 1.0, 0, 255)
    #     with torch.no_grad():
    #         fake = generator(fixed_noise).detach().cpu()

        

    #     print("epoch done")

    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    # plt.savefig('loss'+'.png')


vis(checkpoint="../extra/2021-11-26/ResidualGenerator-50-50-FINAL.tar")