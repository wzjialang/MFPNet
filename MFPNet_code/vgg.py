import torch
import torch.nn as nn
from torchvision import models

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_4 = nn.Sequential()
        self.to_relu_4_4 = nn.Sequential()
        self.to_relu_5_4 = nn.Sequential()
        # conv -1
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 18):
            self.to_relu_3_4.add_module(str(x), features[x])
        for x in range(18, 27):
            self.to_relu_4_4.add_module(str(x), features[x])
        for x in range(27, 36):
            self.to_relu_5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_4(h)
        h_relu_3_4 = h
        h = self.to_relu_4_4(h)
        h_relu_4_4 = h
        h = self.to_relu_5_4(h)
        h_relu_5_4 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_4, h_relu_4_4, h_relu_5_4)
        return out