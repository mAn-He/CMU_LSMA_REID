import torch
import torch.nn as nn
from torchvision.models import resnet101
import torchvision.transforms as T
import torch.nn.functional as F 
from config import CONFIG
import torchvision
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, num_classes=None, use_head=True):
        """
        ResNet Feature Extractor with an optional classification head.
        
        Args:
            num_classes (int, optional): Number of classes for classification. Defaults to None.
            use_head (bool, optional): Whether to include a classification head. Defaults to True.
        """
        super(ResNetFeatureExtractor, self).__init__()
        self.use_head = use_head  # Whether to include the classification head
        
        # Initialize the ResNet backbone
        backbone_name = CONFIG["backbone"]  # Load the specified ResNet model
        self.resnet = getattr(torchvision.models, backbone_name)(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the default FC layer

        # Initialize the classification head if `use_head` is True
        if use_head and num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = None  # No head for contrastive learning

        # Add resizing to ensure input compatibility
        # self.resize = T.Resize((224, 224))

    def forward(self, x):
        # Resize input to match ResNet's expected dimensions
        # x = self.resize(x)
        features = self.resnet(x)  # Extract features using ResNet backbone
        features = F.normalize(features, p=2, dim=1)
        if self.use_head and self.classifier is not None:
            logits = self.classifier(features)  # Pass through classification head
            return features, logits
        else:
            return features  # Return only features for contrastive learning
