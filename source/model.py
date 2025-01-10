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
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.25),
        )
        
    def forward(self, x, skip):

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.upconv(x)
        x = self.conv(x)
        return x

class UNet_1D(nn.Module):
    def __init__(self):
        super(UNet_1D, self).__init__()
        self.encoder_block1 = EncoderBlock(1, 16)
        self.encoder_block2 = EncoderBlock(16, 16)
        self.encoder_block3 = EncoderBlock(16, 32)
        self.encoder_block4 = EncoderBlock(32, 32)
        self.encoder_block5 = EncoderBlock(32, 64)
        self.encoder_block6 = EncoderBlock(64, 64)
        
        
        self.decoder_block1 = DecoderBlock(64, 64)
        self.decoder_block2 = DecoderBlock(128, 32)
        self.decoder_block3 = DecoderBlock(64, 32)
        self.decoder_block4 = DecoderBlock(64, 16)
        self.decoder_block5 = DecoderBlock(32, 16)
        
        self.outconv = nn.ConvTranspose1d(32, 1, kernel_size=2, stride=2)
        
    def forward(self, x):
        _, p1 = self.encoder_block1(x)
        _, p2 = self.encoder_block2(p1)
        _, p3 = self.encoder_block3(p2)
        _, p4 = self.encoder_block4(p3)
        _, p5 = self.encoder_block5(p4)
        _, p6 = self.encoder_block6(p5)
        
        x = self.decoder_block1(p6, None)
        x = self.decoder_block2(x, p5)
        x = self.decoder_block3(x, p4)
        x = self.decoder_block4(x, p3)
        x = self.decoder_block5(x, p2)
        
        x = self.outconv(torch.cat([x, p1], dim=1))
        
        return torch.sigmoid(x)
    
    def initialize_weights_uniform(self, lower_bound=-0.1, upper_bound=0.1):
        for param in self.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, lower_bound, upper_bound)