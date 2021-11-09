import argparse
import datetime
import os

import torch
import torch.optim as optim

import models.generator as gen
import utils.loss as loss

parser = argparse.ArgumentParser()
parser.add_argument('-gen', type=int, default=0) # 0 for BaseGenerator, 1 for ResidualGenerator, 2 for MultiResidualGenerator
parser.add_argument('-epochs', type=int, default=10) # TODO: find good default
parser.add_argument('-save', type=int, default=5) # TODO: maybe instead rely on # of epochs?
parser.add_argument('-dest', type=str, default="/runs/"+str(datetime.date.today()))
parser.add_argument('-lr', type=float, default=0.001) # TODO: search over space of hyperparameters
parser.add_argument('-beta', type=float, default=0.0001) # TODO: search over space of hyperparameters
parser.add_argument('--verbose', action='count', default=0) #TODO: implement verbose output
# TODO: add argument parsin to allow for loading from a model

args = parser.parse_args()

verbose = True if args.verbose > 0 else False
dest = args.dest.split("/")
dest = os.path.join(*dest)
if not os.path.isdir(dest):
    os.makedirs(dest)
    
gen_arch = gen.BaseGenerator
if args.gen == 1:
    gen_arch = gen.ResidualGenerator
elif args.gen == 2:
    gen_arch = gen.MultiResidualGenerator
    
run_name_base = gen_arch.__name__ + "-" + str(args.epochs) + "-" # when saving finish with the current epoch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if verbose:
    print("Using device: {}".format(device))

generator = gen_arch().to(device)
if verbose:
    print(generator)
    
'''
TODO: put in classifier loading code here
      remember to move it to device and freeze weights
'''
classifier = None


'''
TODO: put in dataloader code here
      remember to  move data to device
'''
trainloader = None

optimizer = optim.Adam(generator.parameters(), args.lr) 
loss_fn = loss.generator_loss

for epoch in range(args.epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, targets = data # will we have targets from the loader?
                               # maybe start with a single target, e.g. 'gorilla'

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = generator(inputs)
        preds = classifier(outputs)
        
        loss = loss_fn(inputs, outputs, preds, targets, args.beta)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
    if epoch % args.save == 0: # save generator model
        run_name = run_name_base + str(epoch) + "-checkpoint.tar"
        torch.save(generator, run_name)

'''
TODO: add final evaluataion code
'''

# save final model
run_name = run_name_base + str(args.epoch) + "-FINAL.tar"
torch.save(generator, run_name)
print('Finished Training')