# Copyright (C) 2025 Riley K
# License: GNU AGPL v3

import torch
import torch.nn as nn
from __init__ import device

GRID_SIZE = 8

class multi_digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation=nn.ReLU()

        # # Number 1 (This is 17.809 billion flops).
        # # Downsampling and upsampling with 1 in-between CNN layer for relevant height layers,
        # # and with residual to dimensions 32 x 32
        # # Downsampling
        # self.conv_level_0_downsample_0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # self.conv_level_0_downsample_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        # self.pool_level_0 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_1_downsample_0 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # self.conv_level_1_downsample_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.pool_level_1 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_2_downsample_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv_level_2_downsample_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.pool_level_2 = nn.MaxPool2d(kernel_size=2)

        # # 32 x 32
        # # Downsampling
        # self.conv_level_3_downsample_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv_level_3_downsample_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.pool_level_3 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_3_across_0 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_3_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_3_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # # 1 x 1 Linear Convolutions
        # self.conv_1x1_linear_level_3_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        # self.conv_1x1_linear_level_3_1 = nn.Conv2d(in_channels=256, out_channels=12, kernel_size=1, padding=0)
        
        # # 16 x 16
        # # Downsampling
        # self.conv_level_4_downsample_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv_level_4_downsample_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.pool_level_4 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_4_across_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_4_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_4_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.level_4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # # 8 x 8
        # # Downsampling
        # self.conv_level_5_downsample_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_5_downsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.pool_level_5 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_5_across_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_5_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_5_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.level_5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # # 4 x 4
        # # Downsampling
        # self.conv_level_6_downsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_6_downsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # # Upsampling
        # self.level_6_upsample = nn.Upsample(scale_factor=2, mode='nearest')


        # Number 2 (This is 13.563 billion flops. It was 13.458 billion flops 
        # before adding projections for residuals. Projections added .105 billion flops.)
        # Downsampling and upsampling with 1 in-between CNN layer for relevant height layers,
        # and with residuals to dimensions 32 x 32
        # Downsampling
        # self.conv_level_0_downsample_0 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        # self.conv_level_0_downsample_1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        # self.pool_level_0 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_1_downsample_0 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # self.conv_level_1_downsample_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        # self.conv_projection_downsample_level_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, padding=0)
        # self.pool_level_1 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_2_downsample_0 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # self.conv_level_2_downsample_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.conv_projection_downsample_level_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0)
        # self.pool_level_2 = nn.MaxPool2d(kernel_size=2)

        # # 32 x 32
        # # Downsampling
        # self.conv_level_3_downsample_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv_level_3_downsample_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv_projection_downsample_level_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0)
        # self.pool_level_3 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_3_across_0 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_3_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_3_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # # 1 x 1 Linear Convolutions
        # self.conv_1x1_linear_level_3_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        # self.conv_1x1_linear_level_3_1 = nn.Conv2d(in_channels=256, out_channels=12, kernel_size=1, padding=0)
        
        # # 16 x 16
        # # Downsampling
        # self.conv_level_4_downsample_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv_level_4_downsample_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.conv_projection_downsample_level_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0)
        # self.pool_level_4 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_4_across_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_projection_upsample_level_4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        # self.conv_level_4_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_4_upsample_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        # self.level_4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # # 8 x 8
        # # Downsampling
        # self.conv_level_5_downsample_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv_level_5_downsample_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.conv_projection_downsample_level_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0)
        # self.pool_level_5 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_5_across_0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_projection_upsample_level_5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        # self.conv_level_5_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_5_upsample_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        # self.level_5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # # 4 x 4
        # # Downsampling
        # self.conv_level_6_downsample_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_6_downsample_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        # # Upsampling
        # self.level_6_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Number 3 (This is 796.426 million FLOPS).
        # Downsampling and upsampling with 1 in-between CNN layer for relevant height layers,
        # and with residuals to dimensions 8 x 8
        # Downsampling
        self.conv_level_0_downsample_0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv_level_0_downsample_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool_level_0 = nn.MaxPool2d(kernel_size=2)

        self.conv_level_1_downsample_0 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_level_1_downsample_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_projection_downsample_level_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0)
        self.pool_level_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_level_2_downsample_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_level_2_downsample_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_projection_downsample_level_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0)
        self.pool_level_2 = nn.MaxPool2d(kernel_size=2)

        # 8 x 8
        # Downsampling
        self.conv_level_3_downsample_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_level_3_downsample_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_projection_downsample_level_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0)
        self.pool_level_3 = nn.MaxPool2d(kernel_size=2)
        # Across
        self.conv_level_3_across_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # Upsampling
        self.conv_level_3_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_level_3_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # 1 x 1 Linear Convolutions
        self.conv_1x1_linear_level_3_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        self.conv_1x1_linear_level_3_1 = nn.Conv2d(in_channels=256, out_channels=13, kernel_size=1, padding=0)
        
        # 4 x 4
        # Downsampling
        self.conv_level_4_downsample_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_level_4_downsample_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_projection_downsample_level_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0)
        self.pool_level_4 = nn.MaxPool2d(kernel_size=2)
        # Across
        self.conv_level_4_across_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # Upsampling
        self.conv_projection_upsample_level_4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        self.conv_level_4_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_level_4_upsample_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.level_4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,x):
            # Number 1
            # x = x
            # x = self.conv_level_0_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_0_downsample_1(x)
            # x = self.activation(x)
            # x = self.pool_level_0(x)
            # # Level 1 downsample
            # x = self.conv_level_1_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_1_downsample_1(x)
            # x = self.activation(x)
            # x = self.pool_level_1(x)
            # # Level 2 downsample
            # x = self.conv_level_2_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_2_downsample_1(x)
            # x = self.activation(x)
            # x = self.pool_level_2(x)
            # # Level 3 downsample
            # x = self.conv_level_3_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_3_downsample_1(x)
            # x = self.activation(x)

            # x_3 = x

            # x = self.pool_level_3(x)
            # # Level 4 downsample
            # x = self.conv_level_4_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_4_downsample_1(x)
            # x = self.activation(x)

            # x_2 = x

            # x = self.pool_level_4(x)
            # # Level 5 downsample
            # x = self.conv_level_5_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_5_downsample_1(x)
            # x = self.activation(x)

            # x_1 = x

            # x = self.pool_level_5(x)
            # # Level 6 downsample
            # x = self.conv_level_6_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_6_downsample_1(x)
            # x = self.activation(x)
            # # Level 5 across
            # x_1 = self.conv_level_5_across_0(x_1)
            # x_1 = self.activation(x_1)
            # # Level 5 upsample and sum
            # x = self.level_6_upsample(x)
            # x = x + x_1
            # x = self.conv_level_5_upsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_5_upsample_1(x)
            # x = self.activation(x)
            # # Level 4 across
            # x_2 = self.conv_level_4_across_0(x_2)
            # x_2 = self.activation(x_2)
            # # Level 4 upsample and sum
            # x = self.level_5_upsample(x)
            # x = x + x_2
            # x = self.conv_level_4_upsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_4_upsample_1(x)
            # x = self.activation(x)
            # #Level 3 across
            # x_3 = self.conv_level_3_across_0(x_3)
            # x_3 = self.activation(x_3)
            # # Level 3 upsample and sum
            # x = self.level_4_upsample(x)
            # x = x + x_3
            # x = self.conv_level_3_upsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_3_upsample_1(x)
            # x = self.activation(x)
            # # Level 3 1x1 linear convolutions
            # x = self.conv_1x1_linear_level_3_0(x)
            # x = self.activation(x)
            # x = self.conv_1x1_linear_level_3_1(x)




            # Number 2
            # Level 0 downsample
            # x = self.conv_level_0_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_0_downsample_1(x)
            # x = self.activation(x)
            # x = self.pool_level_0(x)
            # # Level 1 downsample
            # x_initial = x
            # x = self.conv_level_1_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_1_downsample_1(x)
            # x_projection = self.conv_projection_downsample_level_1(x_initial)
            # x = x + x_projection
            # x = self.activation(x)
            # x = self.pool_level_1(x)
            # # Level 2 downsample
            # x_initial = x
            # x = self.conv_level_2_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_2_downsample_1(x)
            # x_projection = self.conv_projection_downsample_level_2(x_initial)
            # x = x + x_projection
            # x = self.activation(x)
            # x = self.pool_level_2(x)
            # # Level 3 downsample
            # x_initial = x
            # x = self.conv_level_3_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_3_downsample_1(x)
            # x_projection = self.conv_projection_downsample_level_3(x_initial)
            # x = x + x_projection
            # x = self.activation(x)

            # x_level_3 = x

            # x = self.pool_level_3(x)
            # # Level 4 downsample
            # x_initial = x
            # x = self.conv_level_4_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_4_downsample_1(x)
            # x_projection = self.conv_projection_downsample_level_4(x_initial)
            # x = x + x_projection
            # x = self.activation(x)

            # x_level_4 = x

            # x = self.pool_level_4(x)
            # # Level 5 downsample
            # x_initial = x
            # x = self.conv_level_5_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_5_downsample_1(x)
            # x_projection = self.conv_projection_downsample_level_5(x_initial)
            # x = x + x_projection
            # x = self.activation(x)

            # x_level_5 = x

            # x = self.pool_level_5(x)
            # # Level 6 downsample
            # x_initial = x
            # x = self.conv_level_6_downsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_6_downsample_1(x)
            # x = x + x_initial
            # x = self.activation(x)
            # # Level 5 across
            # x_level_5 = self.conv_level_5_across_0(x_level_5)
            # x_level_5 = self.activation(x_level_5)
            # # Level 5 upsample and concatenation
            # x = self.level_6_upsample(x)
            # x = torch.cat([x_level_5,x], dim=1)
            # x_initial = x
            # x = self.conv_level_5_upsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_5_upsample_1(x)
            # x_projection = self.conv_projection_upsample_level_5(x_initial)
            # x = x + x_projection
            # x = self.activation(x)
            # # Level 4 across
            # x_level_4 = self.conv_level_4_across_0(x_level_4)
            # x_level_4 = self.activation(x_level_4)
            # # Level 4 upsample and concatenation
            # x = self.level_5_upsample(x)
            # x = torch.cat([x_level_4,x], dim=1)
            # x_initial = x
            # x = self.conv_level_4_upsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_4_upsample_1(x)
            # x_projection = self.conv_projection_upsample_level_4(x_initial)
            # x = x + x_projection
            # x = self.activation(x)
            # #Level 3 across
            # x_level_3 = self.conv_level_3_across_0(x_level_3)
            # x_level_3 = self.activation(x_level_3)
            # # Level 3 upsample and concatenation
            # x = self.level_4_upsample(x)
            # x = torch.cat([x_level_3,x], dim=1)
            # x_initial = x
            # x = self.conv_level_3_upsample_0(x)
            # x = self.activation(x)
            # x = self.conv_level_3_upsample_1(x)
            # x = x + x_initial
            # x = self.activation(x)
            # # Level 3 1x1 linear convolutions
            # x = self.conv_1x1_linear_level_3_0(x)
            # x = self.activation(x)
            # x = self.conv_1x1_linear_level_3_1(x)

            # Number 3
            # Level 0 downsample
            x = self.conv_level_0_downsample_0(x)
            x = self.activation(x)
            x = self.conv_level_0_downsample_1(x)
            x = self.activation(x)
            x = self.pool_level_0(x)

            # Level 1 downsample
            x_initial = x
            x = self.conv_level_1_downsample_0(x)
            x = self.activation(x)
            x = self.conv_level_1_downsample_1(x)
            x_projection = self.conv_projection_downsample_level_1(x_initial)
            x = x + x_projection
            x = self.activation(x)
            x = self.pool_level_1(x)

            # Level 2 downsample
            x_initial = x
            x = self.conv_level_2_downsample_0(x)
            x = self.activation(x)
            x = self.conv_level_2_downsample_1(x)
            x_projection = self.conv_projection_downsample_level_2(x_initial)
            x = x + x_projection
            x = self.activation(x)
            x = self.pool_level_2(x)
            # Level 3 downsample
            x_initial = x
            x = self.conv_level_3_downsample_0(x)
            x = self.activation(x)
            x = self.conv_level_3_downsample_1(x)
            x_projection = self.conv_projection_downsample_level_3(x_initial)
            x = x + x_projection
            x = self.activation(x)

            x_level_3 = x
            
            x = self.pool_level_3(x)
            # Level 4 downsample
            x_initial = x
            x = self.conv_level_4_downsample_0(x)
            x = self.activation(x)
            x = self.conv_level_4_downsample_1(x)
            x_projection = self.conv_projection_downsample_level_4(x_initial)
            x = x + x_projection
            x = self.activation(x)

            #Level 3 across
            x_level_3 = self.conv_level_3_across_0(x_level_3)
            x_level_3 = self.activation(x_level_3)
            # Level 3 upsample and concatenation
            x = self.level_4_upsample(x)
            x = torch.cat([x_level_3,x], dim=1)
            x_initial = x
            x = self.conv_level_3_upsample_0(x)
            x = self.activation(x)
            x = self.conv_level_3_upsample_1(x)
            x = x + x_initial
            x = self.activation(x)
            # Level 3 1x1 linear convolutions
            x = self.conv_1x1_linear_level_3_0(x)
            x = self.activation(x)
            x = self.conv_1x1_linear_level_3_1(x)

            return x
