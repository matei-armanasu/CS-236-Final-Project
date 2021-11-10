import torch
import torch.nn as nn
import torchvision.models as models

def pretrained_resnet_18():
    model_ft = models.resnet18()
    #Finetune Final few layers to adjust for tiny imagenet input
    model_ft.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model_ft.maxpool = nn.Sequential()
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)

    # Loading weights from pretrained model
    pretrained_dict = torch.load('checkpoints/resnet18-f37072fd.pth')
    model_ft_dict = model_ft.state_dict()
    first_layer_weight = model_ft_dict['conv1.weight']
    pretrained_dict = {b[0]:b[1] for a,b in zip(model_ft_dict.items(), pretrained_dict.items()) if a[1].size() == b[1].size()}
    model_ft_dict.update(pretrained_dict) 
    model_ft.load_state_dict(model_ft_dict)
    
    model_ft.fc = nn.Linear(512,200)

    return model_ft

# TODO: Sean -- add method to pull in pretrained mixmo classifier

# TODO: Sean -- add method to pull in pretrained congruency classifier
