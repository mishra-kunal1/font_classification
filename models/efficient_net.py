import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


class CustomEfficientNet:
    """
    Custom EfficientNet model with the desired number of output classes
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_model(self):
        # Load the EfficientNet model with the desired weights
        weights = models.EfficientNet_B1_Weights.DEFAULT 
        seq_model = models.efficientnet_b1(weights=weights)
        
        # Freezing the feature extraction layers
        for param in seq_model.parameters():
            param.requires_grad = False
        
        # Modifying the classifier to match the number of output classes
        seq_model.classifier[1]=nn.Linear(seq_model.classifier[1].in_features, out_features=10, bias=True)
        return seq_model