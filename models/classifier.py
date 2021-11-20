import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.c1 = nn.Conv2d(3, 32, 5, padding = 2, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lr1 = nn.LeakyReLU()
        
        self.c2 = nn.Conv2d(32, 64, 5, padding = 0, bias = False) # down to 60x60
        self.bn2 = nn.BatchNorm2d(64)
        self.lr2 = nn.LeakyReLU()
        
        self.c3 = nn.Conv2d(64, 128, 5, padding = 0, bias = False) # down to 56x56
        self.bn3 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2) # down to 28x28
        self.lr3 = nn.LeakyReLU()

        self.c4 = nn.Conv2d(128, 128, 5, padding = 0, bias = False) # down to 24x24
        self.bn4 = nn.BatchNorm2d(128)
        self.lr4 = nn.LeakyReLU()

        self.c5 = nn.Conv2d(128, 256, 5, padding = 0, bias = False) # down to 20x20
        self.bn5 = nn.BatchNorm2d(256)
        self.lr5 = nn.LeakyReLU()
        
        self.c_final = nn.Conv2d(256, 256, 5, padding = 0, bias = False) # down to 16x16
        self.bn_final = nn.BatchNorm2d(256)
        self.lr_final = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(65536,1000)
        self.activ1 = nn.LeakyReLU()
        
        self.fc2 = nn.Linear(1000,200) # final scores
    
    def forward(self, x):
        x = self.lr1(self.bn1(self.c1(x)))
        x = self.lr2(self.bn2(self.c2(x)))
        x = self.lr3(self.pool1(self.bn3(self.c3(x))))
        x = self.lr4(self.bn4(self.c4(x)))
        x = self.lr5(self.bn5(self.c5(x)))
        y = self.lr_final(self.bn_final(self.c_final(x)))
        y = torch.flatten(y,1)
        y = self.activ1(self.fc1(y))
        scores = self.fc2(y)
        
        return scores

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

def pretrained_efficient_net():
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=200)
    return model
