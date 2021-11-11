"""
Code needs cleaning
"""
import argparse
import datetime
import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../")
import models.generator as gen
import models.classifier as clas
import utils.loss as loss
import utils.dataloader as dataloader
import utils.visual as visual

if __name__ == '__main__':
    # parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('-dest', type=str, default="/runs/"+str(datetime.date.today()))
    parser.add_argument('--verbose', action='count', default=0) #TODO: implement verbose output
    parser.add_argument('-batch', type=int, default=128)
    # TODO: add argument parsing to allow for loading generator from a model

    args = parser.parse_args()

    verbose = True if args.verbose > 0 else False
    dest = args.dest.split("/")
    dest = os.path.join(*dest)
    if not os.path.isdir(dest):
        os.makedirs(dest)
    
    #Tensorboard
    writer = SummaryWriter()        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print("Using device: {}".format(device))

    # generator = gen_arch().to(device)
 
    classifier = clas.pretrained_efficient_net().to(device)


    # collect data and set up optimizer
    testloader = dataloader.get_loader("test", args.batch)

        # Specify a path
    PATH = "BaseGenerator-500-194-checkpoint.tar"
    # Load
    generator = torch.load(PATH)

    generator.eval()
    classifier.eval()


    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            
            inputs, targets = data
            
            # total += inputs.shape[0] # sum up how many images  we use in testing
            
            target_class = 42 # TODO: ensure this stays in sync with train code
            
            inputs = inputs.to(device)
            
            outputs = generator(inputs)
            preds_fl = classifier(outputs)
            
            preds = classifier(inputs)
            pred_classes_fl = torch.argmax(preds_fl, dim=1) # TODO: ensure this is shaped like inputs.shape[0]
            pred_classes = torch.argmax(preds, dim=1) # TODO: ensure this is shaped like inputs.shape[0]

            # The code currently overwrites the images

            # create grid of images
            img_grid = torchvision.utils.make_grid(inputs.cpu())

            # show images
            visual.matplotlib_imshow(img_grid, one_channel=False)

            # write to tensorboard
            writer.add_image('Tiny Imagenet test set', img_grid)

            # create grid of images
            img_grid = torchvision.utils.make_grid(outputs.cpu())

            # show images
            visual.matplotlib_imshow(img_grid, one_channel=False)

            # write to tensorboard
            writer.add_image('Tiny Imagenet generated', img_grid)
