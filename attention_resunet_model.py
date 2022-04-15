import os
import torchvision
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn


class Attention_ResUnet(torch.nn.Module):
    def __init__(self):
        super(Attention_ResUnet, self).__init__()
        
        self.conv1 = ConvLayer(4, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)
        self.res7 = ResidualBlock(128)
        self.res8 = ResidualBlock(128)
        self.res9 = ResidualBlock(128)
        self.res10 = ResidualBlock(128)
        self.res11 = ResidualBlock(128)
        self.res12 = ResidualBlock(128)
        self.res13 = ResidualBlock(128)
        self.res14 = ResidualBlock(128)
        self.res15 = ResidualBlock(128)
        self.res16 = ResidualBlock(128)
        
        self.deconv1 = UpsampleConvLayer(128*2, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64*2, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32*2, 3, kernel_size=9, stride=1)

        self.relu = torch.nn.ReLU()
        
        # Attentions
        self.att1 = AttentionBlock(128, 128, 64)
        self.att2 = AttentionBlock(64, 64, 32)
        self.att3 = AttentionBlock(32, 32, 16)

    
    def forward(self, X):
        o1 = self.relu(self.conv1(X))
        o2 = self.relu(self.conv2(o1))
        o3 = self.relu(self.conv3(o2))

        y = self.res1(o3)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)
        y = self.res10(y)
        y = self.res11(y)
        y = self.res12(y)
        y = self.res13(y)
        y = self.res14(y)
        y = self.res15(y)
        y = self.res16(y)
        
        o3 = self.att1(y, o3)
        in1 = torch.cat((y, o3), 1)
        y = self.relu(self.deconv1(in1))

        o2 = self.att2(y, o2)
        in2 = torch.cat((y, o2), 1)
        y = self.relu(self.deconv2(in2))

        o1 = self.att3(y, o1)
        in3 = torch.cat((y, o1), 1)
        y = self.deconv3(in3)
        
        return y



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out



class ResidualBlock(torch.nn.Module):
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out



class UpsampleConvLayer(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
    


class AttentionBlock(torch.nn.Module):
    
    def __init__(self, Fg, Fl, Fint):
        super(AttentionBlock, self).__init__()               
        self.Wg = nn.Sequential(ConvLayer(Fg, Fint, kernel_size=3, stride=1), nn.BatchNorm2d(Fint))
        self.Wx = nn.Sequential(ConvLayer(Fl, Fint, kernel_size=3, stride=1), nn.BatchNorm2d(Fint))
        self.relu = nn.ReLU()
        self.psi = nn.Sequential(ConvLayer(Fint, 1, kernel_size=3, stride=1), nn.BatchNorm2d(1), nn.Sigmoid())
        
    def forward(self, g, x):
        g1 = self.Wg(g)   
        x1 = self.Wx(x)
        y = self.relu(g1 + x1)  
        y = self.psi(y)
        return torch.mul(y, x)
