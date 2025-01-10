import torch
import torch.nn as nn
import torch.functional as F
import math
import os

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.25),
        )
        
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = Convolution(in_channels, out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        p = self.maxpool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBlock, self).__init__()
        self.out_channels = in_channels//2
        self.upconv = nn.ConvTranspose1d(in_channels, self.out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.out_channels),
            nn.LeakyReLU(0.25),
        )
        
    def forward(self, x, skip):

        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x

class UNet_1D(nn.Module):
    def __init__(self):
        super(UNet_1D, self).__init__()
        self.encoder_block1 = EncoderBlock(1, 16)
        self.encoder_block2 = EncoderBlock(16, 32)
        self.encoder_block3 = EncoderBlock(32, 64)
        self.encoder_block4 = EncoderBlock(64, 128)
        
        self.bottleneck = Convolution(128, 256)
        
        self.decoder_block1 = DecoderBlock(256)
        self.decoder_block2 = DecoderBlock(128)
        self.decoder_block3 = DecoderBlock(64)
        self.decoder_block4 = DecoderBlock(32)
        
        self.outconv = nn.Conv1d(16, 1, kernel_size=1)
        
    def forward(self, x):
        skip1, p1 = self.encoder_block1(x)
        skip2, p2 = self.encoder_block2(p1)
        skip3, p3 = self.encoder_block3(p2)
        skip4, p4 = self.encoder_block4(p3)

        x = self.bottleneck(p4)
        
        x = self.decoder_block1(x, skip4)
        x = self.decoder_block2(x, skip3)
        x = self.decoder_block3(x, skip2)
        x = self.decoder_block4(x, skip1)
        
        x = self.outconv(x)
        
        return torch.softmax(x, dim=1)
    
    def initialize_weights_uniform(self, lower_bound=-0.1, upper_bound=0.1):
        for param in self.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, lower_bound, upper_bound)