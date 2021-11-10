"""
Some parts of code adapted from https://github.com/tjmoon0104/pytorch-tiny-imagenet
"""

import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
os.environ['TORCH_HOME'] = 'models\\resnet' #setting the environment variable

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

def main():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('-d','--data_path', type=str, default="Imagenet",help='Enter data path')
    args = parser.parse_args()

    data_dir = args.data_path
    # num_workers = {
    #     'train' : 100,
    #     'val'   : 0,
    #     'test'  : 0
    # }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                        for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    print(len(dataloaders["train"]))
    print(len(dataloaders["val"]))
    print(len(dataloaders["test"]))
    

    # print(next(iter(dataloaders["train"])))
    # print(next(iter(dataloaders["val"])))
    # print(next(iter(dataloaders["test"])))


    #Load Resnet18
    model_ft = models.resnet18()
    #Finetune Final few layers to adjust for tiny imagenet input
    model_ft.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model_ft.maxpool = nn.Sequential()
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    model_ft.fc.out_features = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # Loading weights from pretrained model
    pretrained_dict = torch.load('checkpoints/resnet18-f37072fd.pth')
    model_ft_dict = model_ft.state_dict()
    first_layer_weight = model_ft_dict['conv1.weight']
    pretrained_dict = {b[0]:b[1] for a,b in zip(model_ft_dict.items(), pretrained_dict.items()) if a[1].size() == b[1].size()}
    model_ft_dict.update(pretrained_dict) 
    model_ft.load_state_dict(model_ft_dict)
    print(model_ft.state_dict())
    

if __name__ == "__main__":
    main()