import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=15):
    """
    Create and return a modified ResNet-18 model adapted for a specified number of classes.

    Args:
        num_classes (int): Number of classes in the dataset, default is 15 (Aerial_Landscapes has 15 classes)

    Returns:
        model: Modified ResNet-18 model for classification
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_mobilenetv3_small(num_classes=15):
    """
    Create and return a modified MobileNetV3-Small model adapted for a specified number of classes.

    Args:
        num_classes (int): Number of classes in the dataset, default is 15 (Aerial_Landscapes has 15 classes)

    Returns:
        model: Modified MobileNetV3-Small model for classification
    """
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model