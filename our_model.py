import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import matplotlib.pyplot as plt 
import pandas as pd
#import seaborn as sn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys, os 
from os.path import dirname, join, abspath
from sklearn.model_selection import train_test_split
# from feature_block import auto_linear
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Decoupling_Block2(nn.Module):
    def __init__(self, in_channels, out_channels, mode = 'max'):
        super(Decoupling_Block2,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = [1,2], stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size = [2,1], stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace = True)
        if(mode == 'avg'):
            self.pooling = nn.AvgPool2d(kernel_size = 4, stride =1)
        else:
            self.pooling = nn.MaxPool2d(kernel_size = 4, stride =1)

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.pooling(out)
        out = self.relu2(out)
        return out


class Decoupling_Block1(nn.Module):
    def __init__(self, in_channels, out_channels, mode = 'max'):
        super(Decoupling_Block1,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = [2,1], stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size = [1,2], stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace = True)
        if(mode == 'avg'):
            self.pooling = nn.AvgPool2d(kernel_size = 4, stride =1)
        else:
            self.pooling = nn.MaxPool2d(kernel_size = 4, stride =1)

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.pooling(out)
        out = self.relu2(out)
        return out

class Combi_Block(nn.Module):
    def __init__(self):
        super(Combi_Block, self).__init__()
        self.branch1 = Decoupling_Block1(64,64)
        self.branch2 = Decoupling_Block2(64,64)
        self.alpha = Variable(torch.rand(1),requires_grad = True).float().to(device)
        self.beta = Variable(torch.rand(1),requires_grad = True).float().to(device)

    def forward(self,x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = h1*self.alpha+ h2*self.beta
        return h + x

class startingBlock(nn.Module):
    def __init__(self):
        super(startingBlock, self).__init__()
        self.branch1 = Decoupling_Block1(8,64)
        self.branch2 = Decoupling_Block2(8,64)
        self.alpha = Variable(torch.rand(1),requires_grad = True).float().to(device)
        self.beta = Variable(torch.rand(1),requires_grad = True).float().to(device)

    def forward(self,x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = h1*self.alpha+ h2*self.beta
        return h


class trans_combi_block(nn.Module):
    def __init__(self, in_size, out_size,pooling1, pooling2):
        super(trans_combi_block,self).__init__()
        self.alpha = Variable(torch.rand(1), requires_grad=True).float().to(device)
        self.beta = Variable(torch.rand(1), requires_grad=True).float().to(device)
        self.trans_branch1 = trans_decoupling_block(in_size, out_size, pooling1, pooling2)
        self.trans_branch2 = trans_decoupling_block2(in_size, out_size, pooling1, pooling2)
    def forward(self, x):
        h1 = x / self.alpha
        h2 = x / self.beta
        out = self.trans_branch1(h1) + self.trans_branch2(h2)
        return out
class trans_decoupling_block(nn.Module):
    def __init__(self, in_size, out_size, pooling1, pooling2):
        super(trans_decoupling_block, self).__init__()
        self.transconv1 = nn.ConvTranspose2d(in_size, in_size, 1, 2, pooling1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.relu1  = nn.ReLU()
        self.transconv2 = nn.ConvTranspose2d(in_size, out_size,1,2,pooling2,bias= False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        # print("1, input: ",x.shape)
        out = self.transconv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # print("1 [1,2]: ",out.shape)
        out = self.transconv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # print("1 [2,1]: ",out.shape)
        return out

class trans_decoupling_block2(nn.Module):
    def __init__(self, in_size, out_size,pooling1, pooling2):
        super(trans_decoupling_block2,self).__init__()
        self.transconv1 = nn.ConvTranspose2d(in_size, in_size, 1, 2, pooling1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.relu1  = nn.ReLU()
        self.transconv2 = nn.ConvTranspose2d(in_size, out_size,1,2,pooling2,bias= False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        # print("2, input: ",x.shape)
        out = self.transconv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # print("2, [2,1]: ",out.shape)
        out = self.transconv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # print("2, [1,2]: ",out.shape)
        return out

class simple_CNN(nn.Module):
    def __init__(self):
        super(simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, 2)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.conv3 = nn.Conv2d(32, 32, 2)
        self.fc1 = nn.Linear(32 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 400)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.reshape(out.size(0), 20, 20)
        return out

class new_CNN(nn.Module):
    def __init__(self):
        super(new_CNN, self).__init__()

        self.sblock = startingBlock()
        self.layers1  = self._make_layer(1,32)
        self.layers2  = self._make_layer(1,32)
        self.layers3  = self._make_layer(1,32)
        self.avgpool = nn.AvgPool2d(5)
        self.maxpool = nn.MaxPool2d(5)

        self.trans_layer1 = trans_combi_block(64,64,1,1)
        self.trans_layer2 = trans_combi_block(64,64,1,1)
        self.transconv1 = nn.ConvTranspose2d(64, 1, 1, 1, 6, bias=False)
        self.linear_mapping = nn.Linear(23*23, 20*20)
        self.linear_relu = nn.ReLU()
        self.linear_out = nn.Linear(20*20, 20*20)

        
        self.trans_layer3 = trans_combi_block(64,64,2,1)

        self.linear = nn.Linear(89, 128)
        # self.drop1 = nn.Dropout(0.5)
        self.relul = nn.ReLU()
        self.linear1 = nn.Linear(128, 256)
        # self.drop2 = nn.Dropout(0.5)
        self.relul1 = nn.ReLU()
        self.out_l = nn.Linear(256, 25)
        self.out_p = nn.Linear(256,1)


    def _make_layer(self, layer_count, channels):
        return nn.Sequential(Combi_Block(),
            *[Combi_Block() for _ in range(layer_count -1)])

    def forward(self, x):
        out1 = self.sblock(x)
        out2 = self.layers1(out1)
        out3 = self.layers2(out2)
        out4 = self.layers3(out3)

        out4_pool= self.maxpool(out4)
        out5 = self.trans_layer1(out4)
        out6 = self.trans_layer2(out5)
        out7 = self.transconv1(out6)

        linear_out = self.linear_relu(self.linear_mapping(out))
        linear_out = self.linear_out(linear_out)
        linear_out = linear_out.reshape(-1, 20, 20)
        return linear_out


