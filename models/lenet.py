import torch
import torch.nn as nn
class LeNet(nn.Module):
    """
    LeNet model with all the layers defined, ready to be trained from scratch
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(3136 , 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        out=self.layer1(x)
        out=self.batchnorm1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out=self.layer2(out)
        out=self.batchnorm2(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
