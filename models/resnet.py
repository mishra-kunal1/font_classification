import torch
from torchvision import models
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class CustomResNet:
    """
    Custom ResNet model with the desired number of output classes
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_model(self):
        seq_model = models.resnet18(pretrained=True)
        for param in seq_model.parameters():
            param.requires_grad = False
        num_ftrs = seq_model.fc.in_features
        seq_model.fc = nn.Linear(num_ftrs, self.num_classes)
        return seq_model
