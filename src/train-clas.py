import argparse
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../")
import models.classifier as clas
import utils.dataloader as dataloader

from eval import evaluate

if __name__ == '__main__':
    # parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=3) # TODO: find good default
    parser.add_argument('-save', type=int, default=1) # TODO: maybe instead rely on # of epochs?
    parser.add_argument('-dest', type=str, default="/runs/"+str(datetime.date.today()))
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument('-lr', type=float, default=0.001) # TODO: search over space of hyperparameters
    parser.add_argument('--verbose', action='count', default=0) #TODO: implement verbose output
    parser.add_argument('-checkpoint', type=str, default=None)

    args = parser.parse_args()

    verbose = args.verbose
    dest = args.dest.split("/")
    dest = os.path.join(*dest)
    if not os.path.isdir(dest):
        os.makedirs(dest)
    
    #Tensorboard
    writer = SummaryWriter()

    # load classifier
        
    run_name_base = dest + os.sep  + clas.BaseClassifier.__name__ + "-" + str(args.epochs) + "-" # when saving finish with the current epoch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose > 0:
        print("Using device: {}".format(device))
        
    classifier = clas.BaseClassifier().to(device)
    if verbose > 0:
        print(classifier)


    # collect data and set up optimizer
    trainloader = dataloader.get_loader("train", args.batch)

    optimizer = optim.Adam(classifier.parameters(), args.lr) 
    loss_fn = nn.CrossEntropyLoss()


    # train the model
    for epoch in range(args.epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            if verbose > 0:
                print('epoch ' + str(epoch) + ', batch ' + str(i))


            inputs, targets = data # will we have targets from the loader?
                                # maybe start with a single target, e.g. 'gorilla'
            targets = targets.to(device)
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            if verbose > 1:
                print("data loaded")

            preds = classifier(inputs)
            
            if verbose > 1:
                print("images classified")

            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            if verbose > 1:
                print("backprop completed")

            writer.add_scalar("Loss/train", loss.item(), epoch)

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        
        if epoch % args.save == 0: # save generator model
            run_name = run_name_base + str(epoch) + "-checkpoint.tar"
            torch.save(classifier.state_dict(), run_name)


    # save final model
    run_name = run_name_base + str(args.epochs) + "-FINAL.tar"
    torch.save(classifier.state_dict(), run_name)
    print('Finished Training')
    
    # do final evaluation (on training set because we  don't have test labels for some reason???)
    correct =  0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            
            inputs, targets = data
            
            total += inputs.shape[0] # sum up how many images  we use in testing
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            preds = classifier(inputs)
            pred_classes = torch.argmax(preds, dim=1) # TODO: ensure this is shaped like inputs.shape[0]
            
            correct += (pred_classes == targets).sum().item()
            
        acc = float(correct)/float(total)
        print("Fraction classified correctly: " + str(acc))