# Copyright (c) 2022, 2026 Mahdi Kazemi Moghaddam
# SPDX-License-Identifier: Apache-2.0
# Public-release preparation and portability changes: 2026-11-19.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


def conv1x3(in_channels, out_channels, stride=1):
    """
    Conv 1x3 without changing the input shape except for the number of channels
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    """
    Conv 3x3 across the spatial and temporal axes
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, stride), padding=(0, 1), bias=False)

class BasicBlockNoRes(nn.Module):
    """Residual (convolution => [BN] => ReLU/ LeakyReLU) * 2"""
    def __init__(self, in_channels=200, mid_channels=50, stride=1, downsample=None):
        super().__init__()
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels*4, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(mid_channels*4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out)

        return out
    
class ConvInput1D(nn.Module):
    """(convolution => [BN] => ReLU/ LeakyReLU) * 2"""

    def __init__(self, in_channels=180, out_channels=100, mid_channels=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 100), stride=(1, 20), padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class BasicBlock(nn.Module):
    """Residual (convolution => [BN] => ReLU/ LeakyReLU) * 2"""
    def __init__(self, in_channels=200, mid_channels=50, stride=1, downsample=None):
        super().__init__()
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels*4, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(mid_channels*4),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CNN1D(nn.Module):
    def __init__(self, in_ch=6, size="small", out_feature=False):
        super().__init__()
        self.out_feature = out_feature
        if size == "small":
            self.inconv = ConvInput1D(in_channels=in_ch, out_channels=64)
            self.bb1 = BasicBlockNoRes(in_channels=64, mid_channels=16)
            self.bb2 = BasicBlockNoRes(in_channels=64, mid_channels=8, stride=2)
            self.bb3 = BasicBlockNoRes(in_channels=32, mid_channels=8, stride=2)
            #self.drop = nn.Dropout2d(0.5) 
            #self.final_conv = conv3x3(in_channels=100, out_channels=100)
            self.final_conv = conv1x3(in_channels=32, out_channels=32)
            #self.fc = nn.Linear(100*7, 3) # When input is 600 timesteps only
            self.fc = nn.Linear(512, 3) # When input is 1300 timesteps 
        elif size == "large":
            self.inconv = ConvInput1D(in_channels=in_ch, out_channels=64)
            self.bb1 = BasicBlockNoRes(in_channels=64, mid_channels=32)
            self.bb2 = BasicBlockNoRes(in_channels=128, mid_channels=32, stride=2)
            self.bb3 = BasicBlockNoRes(in_channels=128, mid_channels=8, stride=2)
            #self.drop = nn.Dropout2d(0.5) 
            #self.final_conv = conv3x3(in_channels=100, out_channels=100)
            self.final_conv = conv1x3(in_channels=32, out_channels=32)
            #self.fc = nn.Linear(100*7, 3) # When input is 600 timesteps only
            self.fc = nn.Linear(512, 3) # When input is 1300 timesteps 
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.inconv(x)
        x = self.bb1(x)
        x = self.bb2(x)
        x = self.bb3(x)
        x = self.final_conv(x)
        if self.out_feature:
            features = x.view(batch_size, -1)
            x = self.fc(x.view(batch_size, -1))
            
            return x, features
        else:
            #x = self.drop(x)
            #print(x.view(batch_size, -1).shape)
            x = self.fc(x.view(batch_size, -1))
        
            return x    
    
    
