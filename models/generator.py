import torch
import torch.nn as nn

class BaseGenerator(nn.Module):
    '''
    Initial generator implementation, made to allow for modifications and extensions
    on the core architecture.
    '''
    def __init__(self):
        super(BaseGenerator, self).__init__()
        self.c1 = nn.Conv2d(3, 32, 5, padding = 2, bias = False) # TODO: do we need 5x5 convolutions, or will 3x3 be enough?
        self.bn1 = nn.BatchNorm2d(32)
        self.lr1 = nn.LeakyReLU()
        
        self.c2 = nn.Conv2d(32, 64, 5, padding = 2, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lr2 = nn.LeakyReLU()
        
        self.c3 = nn.Conv2d(64, 128, 5, padding = 2, bias = False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lr3 = nn.LeakyReLU()

        self.c4 = nn.Conv2d(128, 64, 5, padding = 2, bias = False)
        self.bn4 = nn.BatchNorm2d(64)
        self.lr4 = nn.LeakyReLU()

        self.c5 = nn.Conv2d(64, 32, 5, padding = 2, bias = False) # TODO: do we need more layers? Will fewer work?
        self.bn5 = nn.BatchNorm2d(32)
        self.lr5 = nn.LeakyReLU()
        
        self.c_final = nn.Conv2d(32, 3, 5, padding = 2, bias = False)
        self.activ_final = nn.Tanh() # output in range [-1,1]. TODO: ensure this matches our
                                     # preprocessing setup
    
    def forward(self, x):
        x = self.lr1(self.bn1(self.c1(x)))
        x = self.lr2(self.bn2(self.c2(x)))
        x = self.lr3(self.bn3(self.c3(x)))
        x = self.lr4(self.bn4(self.c4(x)))
        x = self.lr5(self.bn5(self.c5(x)))
        y = self.activ_final(self.c_final(x))
        
        return y

class ResidualGenerator(BaseGenerator):  
    '''
    Modified generator implementation, introducing a resnet-style component. 
    Also introduces a learnable parameter alpha defining how much of the output
    should be added to the initial input. Extends BaseGenerator, so changes to the 
    convolutional parameters like stride, kernel size, filters, padding, and bias
    will automatically be propagated.
    '''
    def __init__(self, resWeight):
        super(ResidualGenerator, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([resWeight]))
        
    def forward(self, x):
        out = self.lr1(self.bn1(self.c1(x)))
        out = self.lr2(self.bn2(self.c2(out)))
        out = self.lr3(self.bn3(self.c3(out)))
        out = self.lr4(self.bn4(self.c4(out)))
        out = self.lr5(self.bn5(self.c5(out)))
        y = torch.mul(self.alpha, self.activ_final(self.c_final(out))) + x
        
        return y
    
class MultiResidualGenerator(BaseGenerator):  
    '''
    Modified generator implementation,  introducing many resnet-style components. 
    Introduces a learnable parameter alpha defining how much of the output
    should be added to the initial input. The convolutional layers are also split into
    ResNet-style residual blocks for every two convolutions. Extends BaseGenerator,
    so changes to the  convolutional parameters like stride, kernel size, filters,
    padding, and bias will automatically be propagated.
    '''
    def __init__(self, resWeight):
        super(MultiResidualGenerator, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([resWeight]))
        
    def forward(self, x):
        out = self.lr1(self.bn1(self.c1(x)))
        out = self.lr2(self.bn2(self.c2(out))) + x # residual block 1
        out1 = self.lr3(self.bn3(self.c3(out)))
        out1 = self.lr4(self.bn4(self.c4(out1))) + out # residual block 2
        out = self.lr5(self.bn5(self.c5(out1)))
        y = torch.mul(self.alpha, self.activ_final(self.c_final(out))) + x # combined activation
        
        return y
