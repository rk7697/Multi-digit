import torch
import torch.nn as nn


class multi_digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation=nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=3)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(4*4*40,50)
        self.linear2 = nn.Linear(50,10)

    def forward(self,x):
            x = self.conv1(x)
            x = self.activation(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.activation(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.activation(x)
            x = self.pool3(x)

            # Flatten tensor by
            # reshaping to (batch_size , -1)
            batch_size = x.shape[0]
            x=x.reshape(batch_size,-1)

            x=self.linear1(x)
            x = self.activation(x)
            x=self.linear2(x)
            return x