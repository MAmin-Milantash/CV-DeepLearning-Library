import torch
from pretrained_models import load_model

def freeze_model_layers(model):
    """Freeze all layers of the model to use as feature extractor."""
    for param in model.parameters():
        param.requires_grad = False
    return model

def prepare_feature_extractor(model_name='resnet18', num_classes=10):
    model = load_model(model_name=model_name, pretrained=True, num_classes=num_classes)
    model = freeze_model_layers(model)
    return model
