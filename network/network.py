import torch.nn as nn
from __init__ import device

GRID_SIZE = 28

class multi_digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation=nn.ReLU()
        
        # Number 1
        # Linear, no upsampling
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv11 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # self.conv12 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # self.conv13 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
        # self.conv14 = nn.Conv2d(in_channels=512, out_channels=12, kernel_size=1, padding=0)
        
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=2)
        # self.pool5 = nn.MaxPool2d(kernel_size=2)

        # Number 2 (This was 108.605 billion flops!)
        # # Downsampling and upsampling and across with residual to dimensions 28 x 28
        # # Downsampling
        # self.conv_level_0_downsample_0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # self.conv_level_0_downsample_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.pool_level_0 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_1_downsample_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv_level_1_downsample_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.pool_level_1 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_2_downsample_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv_level_2_downsample_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.pool_level_2 = nn.MaxPool2d(kernel_size=2)

        # # 28 x 28 
        # # Downsampling
        # self.conv_level_3_downsample_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv_level_3_downsample_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.pool_level_3 = nn.MaxPool2d(kernel_size=2)
        # #Across
        # self.conv_level_3_across_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_3_across_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_3_across_2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_3_across_3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_3_across_4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_3_across_5 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_3_upsample_0 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_3_upsample_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # # 1 x 1 Convolutions
        # self.conv_1x1_linear_level_3_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
        # self.conv_1x1_linear_level_3_1 = nn.Conv2d(in_channels=512, out_channels=12, kernel_size=1, padding=0)
        
        # # 14 x 14
        # # Downsampling
        # self.conv_level_4_downsample_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_4_downsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.pool_level_4 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_4_across_0 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_4_across_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_4_upsample_0 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_4_upsample_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.level_4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # # 7 x 7
        # # Across
        # self.conv_level_5_across_0 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_5_across_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # # Upsampling
        # self.level_5_upsample = nn.Upsample(scale_factor=2, mode='nearest')


        
        # Number 3 (This is 50 billion flops).
        # Downsampling and upsampling with 1 in-between CNN layer for relevant height layers,
        # and with residual to dimensions 28 x 28
        # Downsampling
        # self.conv_level_0_downsample_0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # self.conv_level_0_downsample_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.pool_level_0 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_1_downsample_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv_level_1_downsample_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.pool_level_1 = nn.MaxPool2d(kernel_size=2)

        # self.conv_level_2_downsample_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv_level_2_downsample_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.pool_level_2 = nn.MaxPool2d(kernel_size=2)

        # # 28 x 28
        # # Downsampling
        # self.conv_level_3_downsample_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv_level_3_downsample_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.pool_level_3 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_3_across_0 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_3_upsample_0 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_3_upsample_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # # 1 x 1 Linear Convolutions
        # self.conv_1x1_linear_level_3_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
        # self.conv_1x1_linear_level_3_1 = nn.Conv2d(in_channels=512, out_channels=12, kernel_size=1, padding=0)
        
        # # 14 x 14
        # # Downsampling
        # self.conv_level_4_downsample_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_4_downsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.pool_level_4 = nn.MaxPool2d(kernel_size=2)
        # # Across
        # self.conv_level_4_across_0 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # # Upsampling
        # self.conv_level_4_upsample_0 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_4_upsample_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.level_4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # # 7 x 7
        # # Downsampling
        # self.conv_level_5_downsample_0 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # self.conv_level_5_downsample_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        # self.level_5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # # Number 4 (This is 12.826 billion flops).
        # # Downsampling and upsampling with 1 in-between CNN layer for relevant height layers,
        # # and with residual to dimensions 28 x 28
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

        # # 28 x 28
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
        
        # # 14 x 14
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
        # # 7 x 7
        # # Downsampling
        # self.conv_level_5_downsample_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv_level_5_downsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.level_5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Number 5 (This is 17.809 billion flops).
        # Downsampling and upsampling with 1 in-between CNN layer for relevant height layers,
        # and with residual to dimensions 32 x 32
        # Downsampling
        self.conv_level_0_downsample_0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv_level_0_downsample_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool_level_0 = nn.MaxPool2d(kernel_size=2)

        self.conv_level_1_downsample_0 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_level_1_downsample_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool_level_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_level_2_downsample_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_level_2_downsample_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool_level_2 = nn.MaxPool2d(kernel_size=2)

        # 32 x 32
        # Downsampling
        self.conv_level_3_downsample_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_level_3_downsample_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool_level_3 = nn.MaxPool2d(kernel_size=2)
        # Across
        self.conv_level_3_across_0 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)
        # Upsampling
        self.conv_level_3_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_level_3_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # 1 x 1 Linear Convolutions
        self.conv_1x1_linear_level_3_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        self.conv_1x1_linear_level_3_1 = nn.Conv2d(in_channels=256, out_channels=12, kernel_size=1, padding=0)
        
        # 16 x 16
        # Downsampling
        self.conv_level_4_downsample_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_level_4_downsample_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool_level_4 = nn.MaxPool2d(kernel_size=2)
        # Across
        self.conv_level_4_across_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # Upsampling
        self.conv_level_4_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_level_4_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.level_4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 8 x 8
        # Downsampling
        self.conv_level_5_downsample_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_level_5_downsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool_level_5 = nn.MaxPool2d(kernel_size=2)
        # Across
        self.conv_level_5_across_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # Upsampling
        self.conv_level_5_upsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_level_5_upsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.level_5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # 4 x 4
        # Downsampling
        self.conv_level_6_downsample_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_level_6_downsample_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # Upsampling
        self.level_6_upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,x):
            # Number 1
            # Linear, no upsampling
            # x = self.conv1(x)
            # x = self.activation(x)
            # x = self.conv2(x)
            # x = self.activation(x)
            # x = self.pool1(x)
            # x = self.conv3(x)
            # x = self.activation(x)
            # x = self.conv4(x)
            # x = self.activation(x)
            # x = self.pool2(x)
            # x = self.conv5(x)
            # x = self.activation(x)
            # x = self.conv6(x)
            # x = self.activation(x)
            # x = self.pool3(x)
            # x = self.conv7(x)
            # x = self.activation(x)
            # x = self.conv8(x)
            # x = self.activation(x)
            # x = self.pool4(x)
            # x = self.conv9(x)
            # x = self.activation(x)
            # x = self.conv10(x)
            # x = self.activation(x)
            # x = self.pool5(x)
            # x = self.conv11(x)
            # x = self.activation(x)
            # x = self.conv12(x)
            # x = self.activation(x)

            # x = self.conv13(x)
            # x = self.activation(x)
            # x = self.conv14(x)

            # Number 2
            # Downsampling and upsampling with residual to dimensions 28 x 28
            # Level 0 downsample
            # x_0 = x
            # x_0 = self.conv_level_0_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_0_downsample_1(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.pool_level_0(x_0)
            # # Level 1 downsample
            # x_0 = self.conv_level_1_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_1_downsample_1(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.pool_level_1(x_0)
            # # Level 2 downsample
            # x_0 = self.conv_level_2_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_2_downsample_1(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.pool_level_2(x_0)
            # # Level 3 downsample
            # x_0 = self.conv_level_3_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_3_downsample_1(x_0)
            # x_0 = self.activation(x_0)

            # x_2 = x_0

            # x_0 = self.pool_level_3(x_0)
            # # Level 4 downsample
            # x_0 = self.conv_level_4_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_4_downsample_1(x_0)
            # x_0 = self.activation(x_0)

            # x_1 = x_0

            # x_0 = self.pool_level_4(x_0)
            # # Level 5 across
            # x_0 = self.conv_level_5_across_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_5_across_1(x_0)
            # x_0 = self.activation(x_0)
            # # Level 4 across
            # x_1 = self.conv_level_4_across_0(x_1)
            # x_1 = self.activation(x_1)
            # x_1 = self.conv_level_4_across_1(x_1)
            # x_1 = self.activation(x_1)
            # # Level 4 upsample and sum
            # x_0 = self.level_5_upsample(x_0)
            # x_0 = x_0 + x_1
            # x_0 = self.conv_level_4_upsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_4_upsample_1(x_0)
            # x_0 = self.activation(x_0)
            # # Level 3 across
            # x_2 = self.conv_level_3_across_0(x_2)
            # x_2 = self.activation(x_2)
            # x_2 = self.conv_level_3_across_1(x_2)
            # x_2 = self.activation(x_2)
            # x_2 = self.conv_level_3_across_2(x_2)
            # x_2 = self.activation(x_2)
            # x_2 = self.conv_level_3_across_3(x_2)
            # x_2 = self.activation(x_2)
            # x_2 = self.conv_level_3_across_4(x_2)
            # x_2 = self.activation(x_2)
            # x_2 = self.conv_level_3_across_5(x_2)
            # x_2 = self.activation(x_2)
            # # Level 3 upsample and sum
            # x_0 = self.level_4_upsample(x_0)
            # x_0 = x_0 + x_2
            # x_0 = self.conv_level_3_upsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_3_upsample_1(x_0)
            # x_0 = self.activation(x_0)
            # # Level 3 1x1 convolutions
            # x_0 = self.conv_1x1_linear_level_3_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_1x1_linear_level_3_1(x_0)

            # Number 3
            # Downsampling and upsampling with residual to dimensions 28 x 28
            # Level 0 downsample
            # x_0 = x
            # x_0 = self.conv_level_0_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_0_downsample_1(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.pool_level_0(x_0)
            # # Level 1 downsample
            # x_0 = self.conv_level_1_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_1_downsample_1(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.pool_level_1(x_0)
            # # Level 2 downsample
            # x_0 = self.conv_level_2_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_2_downsample_1(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.pool_level_2(x_0)
            # # Level 3 downsample
            # x_0 = self.conv_level_3_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_3_downsample_1(x_0)
            # x_0 = self.activation(x_0)

            # x_2 = x_0

            # x_0 = self.pool_level_3(x_0)
            # # Level 4 downsample
            # x_0 = self.conv_level_4_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_4_downsample_1(x_0)
            # x_0 = self.activation(x_0)

            # x_1 = x_0

            # x_0 = self.pool_level_4(x_0)
            # # Level 5 downsample
            # x_0 = self.conv_level_5_downsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_5_downsample_1(x_0)
            # x_0 = self.activation(x_0)
            # # Level 4 across
            # x_1 = self.conv_level_4_across_0(x_1)
            # x_1 = self.activation(x_1)
            # # Level 4 upsample and sum
            # x_0 = self.level_5_upsample(x_0)
            # x_0 = x_0 + x_1
            # x_0 = self.conv_level_4_upsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_4_upsample_1(x_0)
            # x_0 = self.activation(x_0)
            # #Level 3 across
            # x_2 = self.conv_level_3_across_0(x_2)
            # x_2 = self.activation(x_2)
            # # Level 3 upsample and sum
            # x_0 = self.level_4_upsample(x_0)
            # x_0 = x_0 + x_2
            # x_0 = self.conv_level_3_upsample_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_level_3_upsample_1(x_0)
            # x_0 = self.activation(x_0)
            # # Level 3 1x1 linear convolutions
            # x_0 = self.conv_1x1_linear_level_3_0(x_0)
            # x_0 = self.activation(x_0)
            # x_0 = self.conv_1x1_linear_level_3_1(x_0)

            # Number 5
            x_0 = x
            x_0 = self.conv_level_0_downsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_0_downsample_1(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.pool_level_0(x_0)
            # Level 1 downsample
            x_0 = self.conv_level_1_downsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_1_downsample_1(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.pool_level_1(x_0)
            # Level 2 downsample
            x_0 = self.conv_level_2_downsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_2_downsample_1(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.pool_level_2(x_0)
            # Level 3 downsample
            x_0 = self.conv_level_3_downsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_3_downsample_1(x_0)
            x_0 = self.activation(x_0)

            x_3 = x_0

            x_0 = self.pool_level_3(x_0)
            # Level 4 downsample
            x_0 = self.conv_level_4_downsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_4_downsample_1(x_0)
            x_0 = self.activation(x_0)

            x_2 = x_0

            x_0 = self.pool_level_4(x_0)
            # Level 5 downsample
            x_0 = self.conv_level_5_downsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_5_downsample_1(x_0)
            x_0 = self.activation(x_0)

            x_1 = x_0

            x_0 = self.pool_level_5(x_0)
            # Level 6 downsample
            x_0 = self.conv_level_6_downsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_6_downsample_1(x_0)
            x_0 = self.activation(x_0)
            # Level 5 across
            x_1 = self.conv_level_5_across_0(x_1)
            x_1 = self.activation(x_1)
            # Level 5 upsample and sum
            x_0 = self.level_6_upsample(x_0)
            x_0 = x_0 + x_1
            x_0 = self.conv_level_5_upsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_5_upsample_1(x_0)
            x_0 = self.activation(x_0)
            # Level 4 across
            x_2 = self.conv_level_4_across_0(x_2)
            x_2 = self.activation(x_2)
            # Level 4 upsample and sum
            x_0 = self.level_5_upsample(x_0)
            x_0 = x_0 + x_2
            x_0 = self.conv_level_4_upsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_4_upsample_1(x_0)
            x_0 = self.activation(x_0)
            #Level 3 across
            x_3 = self.conv_level_3_across_0(x_3)
            x_3 = self.activation(x_3)
            # Level 3 upsample and sum
            x_0 = self.level_4_upsample(x_0)
            x_0 = x_0 + x_3
            x_0 = self.conv_level_3_upsample_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_level_3_upsample_1(x_0)
            x_0 = self.activation(x_0)
            # Level 3 1x1 linear convolutions
            x_0 = self.conv_1x1_linear_level_3_0(x_0)
            x_0 = self.activation(x_0)
            x_0 = self.conv_1x1_linear_level_3_1(x_0)

            return x_0