import torch.nn as nn

class multi_digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation=nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=50, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(3*3*50,50)
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
            x = self.conv4(x)
            x = self.activation(x)
            x = self.pool4(x)
            x = self.conv5(x)
            x = self.activation(x)
            x = self.pool5(x)

            # Flatten tensor by
            # reshaping to (batch_size , -1)
            batch_size = x.shape[0]
            x=x.reshape(batch_size,-1)

            x=self.linear1(x)
            x = self.activation(x)
            x=self.linear2(x)
            return x