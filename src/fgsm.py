import argparse

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import sys
sys.path.append("../")
import models.classifier as clas
import utils.dataloader as dataloader

# from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image - epsilon*sign_data_grad
    # Return the perturbed image
    return perturbed_image

def evaluate(epsilon, batch, verbose, checkpoint):
    verbose = verbose
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose > 0:
        print("Using device: {}".format(device))

    classifier = clas.pretrained_base_classifier().to(device)
    if verbose > 0:
        print(classifier)

    # do evaluation
    testloader = dataloader.get_loader("test", batch)
    correct_adv = 0
    correct =  0
    total = 0
    psnrs = []
    ssims = []

    for i, data in enumerate(testloader, 0):
        
        inputs, targets = data
        
        total += inputs.shape[0] # sum up how many images  we use in testing
        
        target_class = 42 
        
        inputs = inputs.to(device)
        inputs.requires_grad = True
        
        preds = classifier(inputs)
        pred_classes = torch.argmax(preds, dim=1)
        correct += pred_classes[pred_classes==target_class].size()[0]

        adv_target = torch.ones(inputs.shape[0],dtype=int).to(device)*target_class
        loss = F.nll_loss(preds, adv_target)
        classifier.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data
        outputs = fgsm_attack(inputs, epsilon, data_grad)
        preds_adv = classifier(outputs)
        
        pred_classes_adv = torch.argmax(preds_adv, dim=1) 
        
        
        correct_adv += pred_classes_adv[pred_classes_adv==target_class].size()[0]
        
        #print(pred_classes)
        #print(pred_classes_adv)

        inputs = np.moveaxis(inputs.cpu().detach().numpy(),1,3)
        outputs = np.moveaxis(outputs.cpu().detach().numpy(),1,3)
        psnrs.append(psnr(inputs,outputs,data_range=2))
        ssims.append(ssim(inputs,outputs,win_size=9, multichannel=True))

        if i == 0:
            torch.save(torch.from_numpy(outputs).cpu(),"fgsm_images_" + str(epsilon) + ".tar")
        
    acc_adv = float(correct_adv)/float(total)
    acc = float(correct)/float(total)
    psnr_mean = np.mean(psnrs)
    ssim_mean = np.mean(ssims)
    print("Final adversarial accuracy: " + str(acc_adv))
    print("Fraction classified pre-attack: " + str(acc))
    print("FGSM PSNR: " + str(psnr_mean))
    print("FGSM SSIM: " + str(ssim_mean))

if __name__ == '__main__':
    
    # parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('-eps', type=float, default=0.05) 
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument('--verbose', action='count', default=0) #TODO: implement verbose output
    parser.add_argument('-checkpoint', type=str)

    args = parser.parse_args()
    
    evaluate(args.eps, args.batch, args.verbose, args.checkpoint)
