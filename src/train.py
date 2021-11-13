import argparse
import datetime
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../")
import models.generator as gen
import models.classifier as clas
import utils.loss as loss
import utils.dataloader as dataloader

if __name__ == '__main__':
    # parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('-gen', type=int, default=0) # 0 for BaseGenerator, 1 for ResidualGenerator, 2 for MultiResidualGenerator
    parser.add_argument('-epochs', type=int, default=3) # TODO: find good default
    parser.add_argument('-save', type=int, default=1) # TODO: maybe instead rely on # of epochs?
    parser.add_argument('-dest', type=str, default="/runs/"+str(datetime.date.today()))
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument('-lr', type=float, default=0.001) # TODO: search over space of hyperparameters
    parser.add_argument('-beta', type=float, default=0.0001) # TODO: search over space of hyperparameters
    parser.add_argument('--verbose', action='count', default=0) #TODO: implement verbose output
    # TODO: add argument parsing to allow for loading generator from a model

    args = parser.parse_args()

    verbose = args.verbose
    dest = args.dest.split("/")
    dest = os.path.join(*dest)
    if not os.path.isdir(dest):
        os.makedirs(dest)
    
    #Tensorboard
    writer = SummaryWriter()

    # load generator andd classifier
    gen_arch = gen.BaseGenerator
    if args.gen == 1:
        gen_arch = gen.ResidualGenerator
    elif args.gen == 2:
        gen_arch = gen.MultiResidualGenerator
        
    run_name_base = gen_arch.__name__ + "-" + str(args.epochs) + "-" # when saving finish with the current epoch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose > 0:
        print("Using device: {}".format(device))

    generator = None
    if args.gen == 0:
        gen_arch = gen_arch().to(device)
    else: # need to provide initialization for resWeight
        generator = gen_arch(.01).to(device)
    if verbose > 0:
        print(generator)
        
    classifier = clas.pretrained_efficient_net().to(device)
    if verbose > 0:
        print(classifier)


    # collect data and set up optimizer
    trainloader = dataloader.get_loader("train", args.batch)

    optimizer = optim.Adam(generator.parameters(), args.lr) 
    loss_fn = loss.generator_loss


    # train the model
    for epoch in range(args.epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            if verbose > 0:
                print('epoch ' + str(epoch) + ', batch ' + str(i))


            inputs, targets = data # will we have targets from the loader?
                                # maybe start with a single target, e.g. 'gorilla'
            targets = torch.zeros(200)
            targets[42] = 1 # arbitrarily selected, TODO: make this variable
            targets = targets.to(device)
            
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            if verbose > 1:
                print("data loaded")
            # forward + backward + optimize
            outputs = generator(inputs)

            if verbose > 1:
                print("images generated")

            preds = classifier(outputs)
            
            if verbose > 1:
                print("images classified")

            loss = loss_fn(inputs, outputs, preds, targets, args.beta)
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
            torch.save(generator, run_name)


    # do final evaluation (possibly abstract this to a method?)
    testloader = dataloader.get_loader("test", args.batch)
    correct_adv = 0
    correct =  0
    total = 0

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
            
        acc_adv = float(correct_adv)/float(total)
        acc = float(correct)/float(total)
        print("Final adversarial accuracy: " + str(acc_adv))
        print("Fraction classified pre-attack: " + str(acc))

    # save final model
    run_name = run_name_base + str(args.epochs) + "-FINAL.tar"
    torch.save(generator, run_name)
    print('Finished Training')
