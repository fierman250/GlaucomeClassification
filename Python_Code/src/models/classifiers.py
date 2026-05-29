import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=2, pretrained=True):
    """
    Instantiates one of the 5 requested CNN architectures with pre-trained ImageNet weights,
    and replaces the final classification head to output `num_classes` (default 2 for Glaucoma).
    
    Supported model_name: 'alexnet', 'vgg16', 'densenet121', 'resnet18', 'mobilenet_v2'
    """
    model_name = model_name.lower()
    
    # We use the updated weights parameter if available, else pretrained=True
    # For PyTorch > 0.13, weights=models.<Model>_Weights.DEFAULT is preferred
    # To keep compatibility, we'll use pretrained=True which is still supported in many versions,
    # or handle the weights parameter if needed. pretrained=True works fine in PyTorch 2.5.1
    
    if model_name == 'alexnet':
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        model = models.alexnet(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'densenet121':
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        weights = models.MobileNetV2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Model {model_name} is not supported. Choose from 'alexnet', 'vgg16', 'densenet121', 'resnet18', 'mobilenet_v2'.")
        
    return model

if __name__ == "__main__":
    # Quick test to ensure all models instantiate properly and have correct output shape
    models_to_test = ['alexnet', 'vgg16', 'densenet121', 'resnet18', 'mobilenet_v2']
    dummy_input = torch.randn(1, 3, 224, 224)
    
    for name in models_to_test:
        print(f"Testing {name}...")
        model = get_model(name)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape} (Expected: [1, 2])")
