import torch
from pretrained_models import load_model

def unfreeze_last_layers(model, num_layers_to_unfreeze=2):
    """
    Unfreeze the last `num_layers_to_unfreeze` layers of the model for fine-tuning.
    """
    # Flatten model layers (works for Sequential models)
    layers = list(model.children())
    for layer in layers[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    return model

def prepare_fine_tuning(model_name='resnet18', num_classes=10, num_layers_to_unfreeze=2):
    model = load_model(model_name=model_name, pretrained=True, num_classes=num_classes)
    model = unfreeze_last_layers(model, num_layers_to_unfreeze=num_layers_to_unfreeze)
    return model
