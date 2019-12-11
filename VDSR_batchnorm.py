import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, no_filter=64):
        super(block, self).__init__()
        self.conv = nn.Conv2d(in_channels=no_filter, out_channels=no_filter, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(no_filter)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        input = self.conv(input)
        input = self.bn(input)
        output = self.relu(input)
        return output

class vdsr_bn(nn.Module):
    def __init__(self, depth=18):
        super(vdsr_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deepconv = nn.ModuleList([block(64) for i in range(depth)])
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        for Block in self.deepconv:
            x = Block(x)
        output = self.conv2(x) + input
        return output