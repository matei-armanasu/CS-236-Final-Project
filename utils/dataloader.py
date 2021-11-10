import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_loader(which_loader, batch = 100, data_dir = "../Imagenet"):
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])
    }   
    
    # num_workers = {
    #     'train' : 100,
    #     'val'   : 0,
    #     'test'  : 0
    # }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                        for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val', 'test']}
    return dataloaders[which_loader]
