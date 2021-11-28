import torch
import torch.nn as nn

class AffineReLU(torch.nn.Module):
    def __init__(self, m=1, c=0):
        """
        In the constructor we instantiate a relu on affine value
        """
        super(AffineReLU, self).__init__()
        self.relu = torch.nn.ReLU()
        self.m = m
        self.c = c

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. 
        """
        x = self.relu(self.m*x + self.c)
        return x
        
        
class Thresholder(nn.Module):
    def __init__(self, b=0, eps=1e-6):
        super(Thresholder, self).__init__()
        self.arelu1     = AffineReLU(c = -(b+eps))
        self.arelu2     = AffineReLU(m = -1000, c = 1)
        self.arelu3     = AffineReLU(m = -1000, c = 1)    

    def forward(self, x):        
        x = self.arelu1(x)
        x = self.arelu2(x)
        x = self.arelu3(x)        
        return x 
