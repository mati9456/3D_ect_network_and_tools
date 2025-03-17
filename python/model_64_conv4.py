import torch
import torch.nn as nn
import torch.nn.functional as f

kernelSize = 4
stride = 2
padding = 1
outerPadding = 0

class ECT_conv_net_3d(nn.Module):
    def __init__(self, s_f=8, leak=0.2):
        super(ECT_conv_net_3d, self).__init__()  
        self.fc0 = nn.Linear(496, 512, bias=True)

        self.leak = leak
        self.sq = nn.Sequential(
            nn.ConvTranspose3d(1, s_f*16, kernel_size=3, stride=1, padding=1, bias=False), # make more channels
            nn.BatchNorm3d(s_f*16),
            nn.LeakyReLU(leak, True),
            nn.ConvTranspose3d(s_f*16, s_f*8, kernel_size=kernelSize, stride=stride, padding=padding, output_padding=outerPadding, bias=False), # Upsample to 16x16
            nn.BatchNorm3d(s_f*8),
            nn.LeakyReLU(leak, True),
            nn.ConvTranspose3d(s_f*8, s_f*4, kernel_size=kernelSize, stride=stride, padding=padding, output_padding=outerPadding, bias=False), # Upsample to 32x32
            nn.BatchNorm3d(s_f*4),
            nn.LeakyReLU(leak, True),
            nn.ConvTranspose3d(s_f*4, s_f*2, kernel_size=kernelSize, stride=stride, padding=padding, output_padding=outerPadding, bias=False), # Upsample to 64x64
            nn.BatchNorm3d(2*s_f),
            nn.LeakyReLU(leak, True),
            nn.ConvTranspose3d(2*s_f, s_f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(s_f),
            nn.LeakyReLU(leak, True),

            nn.ConvTranspose3d(s_f, 1, kernel_size=3, stride=1, padding=1, bias=True), 
            #nn.Tanh()
        )
        self.relu = nn.LeakyReLU(leak, True)

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(x)

        x = x.view(-1, 1, 8, 8, 8)
        x = self.sq(x)

        return x