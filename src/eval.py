import argparse

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import sys
sys.path.append("../")
import models.generator as gen
import models.classifier as clas
import utils.dataloader as dataloader


def evaluate(generator, batch, verbose, checkpoint):
    verbose = verbose

    # load generator and classifier
    gen_arch = gen.BaseGenerator
    if generator == 1:
        gen_arch = gen.ResidualGenerator
    elif generator == 2:
        gen_arch = gen.MultiResidualGenerator
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose > 0:
        print("Using device: {}".format(device))

    generator = None
    if generator == 0:
        gen_arch = gen_arch()
    else: # need to provide initialization for resWeight
        generator = gen_arch(.01)
    gen_state_dict = torch.load(checkpoint)
    generator.load_state_dict(gen_state_dict)
    generator = generator.to(device)
    if verbose > 0:
        print(generator)
        
    classifier = clas.pretrained_efficient_net().to(device)
    if verbose > 0:
        print(classifier)

    # do evaluation
    testloader = dataloader.get_loader("test", args.batch)
    correct_adv = 0
    correct =  0
    total = 0
    psnrs = []
    ssims = []

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            
            inputs, targets = data
            
            total += inputs.shape[0] # sum up how many images  we use in testing
            
            target_class = 42 # TODO: ensure this stays in sync with train code
            
            inputs = inputs.to(device)
            
            outputs = generator(inputs)
            preds_adv = classifier(outputs)
            preds = classifier(inputs)
            pred_classes_adv = torch.argmax(preds_adv, dim=1) # TODO: ensure this is shaped like inputs.shape[0]
            pred_classes = torch.argmax(preds, dim=1) # TODO: ensure this is shaped like inputs.shape[0]
            
            correct_adv += pred_classes_adv[pred_classes_adv==target_class].size()[0]
            correct += pred_classes[pred_classes==target_class].size()[0]
            psnrs.append(psnr(inputs,outputs))
            ssims.append(ssim(inputs,outputs,win_size=10, channel_axis=1))
            
        acc_adv = float(correct_adv)/float(total)
        acc = float(correct)/float(total)
        psnr_mean = np.mean(psnrs)
        ssim_mean = np.mean(ssims)
        print("Final adversarial accuracy: " + str(acc_adv))
        print("Fraction classified pre-attack: " + str(acc))
        print("Generator PSNR: " + str(psnr_mean))
        print("Generator SSIM: " + str(ssim_mean))

if __name__ == '__main__':
    
    # parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('-gen', type=int, default=0) # 0 for BaseGenerator, 1 for ResidualGenerator, 2 for MultiResidualGenerator
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument('--verbose', action='count', default=0) #TODO: implement verbose output
    parser.add_argument('-checkpoint', type=str)

    args = parser.parse_args()
    
    evaluate(args.gen, args.batch, args.verbose, args.checkpoint)