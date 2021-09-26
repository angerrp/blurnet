import torch.nn as nn


class BlurNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.full4 = nn.Linear(128 * 35 * 35, 512)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)

        self.full5 = nn.Linear(512, 2)
        self.soft5 = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 128 * 35 * 35)
        x = self.full4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.full5(x)
        x = self.soft5(x)

        return x