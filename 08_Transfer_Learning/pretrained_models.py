import torch
import torchvision.models as models

def load_model(model_name='resnet18', pretrained=True, num_classes=None):
    """
    Load a pre-trained model from torchvision.
    
    Args:
        model_name (str): Name of the model ('resnet18', 'vgg16', 'efficientnet_b0', etc.)
        pretrained (bool): Load ImageNet pre-trained weights
        num_classes (int, optional): If specified, replace the final layer for new task
    
    Returns:
        model (torch.nn.Module): Loaded model
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if num_classes:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        if num_classes:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        if num_classes:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return model
