import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import torchvision.models as models

import torch.nn as nn
import torchvision.models as models

class TumorResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):  # Add pretrained argument here
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)  # Use pretrained argument here
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Example Check in model.py
class TumorResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Check that your layers (convolutional, pooling, and final fully connected layer) 
        # match the saved model exactly.
        # ...
        self.fc = nn.Linear(in_features, num_classes) # num_classes must be 2

class TumorResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load ResNet50 with ImageNet weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Replace FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
