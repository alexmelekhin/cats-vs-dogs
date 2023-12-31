from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def load_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, num_classes)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model
