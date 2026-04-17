import torch.nn as nn
from torchvision import models


def get_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all pretrained layers first
        for param in model.parameters():
            param.requires_grad = False

        # Replace final classifier
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze pretrained layers
        for param in model.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unsupported model name. Use 'resnet50' or 'mobilenet_v2'.")

    return model