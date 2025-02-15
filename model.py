import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    VGG16_Weights, MobileNet_V2_Weights, Inception_V3_Weights,
    ViT_B_16_Weights, AlexNet_Weights
)

class BasicModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            # Error-prone layer: No padding, might cause size mismatch
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=0),  #1
            nn.BatchNorm2d(64),                                     #2
            nn.LeakyReLU(0.1),                                      #3
            
            # Potential dimension collapse point
            nn.Conv2d(64, 128, kernel_size=3, stride=2),           #4 (no padding)
            nn.Dropout2d(0.3),                                      #5
            nn.ReLU(),                                              #6
            
            # Over-aggressive downsampling
            nn.Conv2d(128, 256, kernel_size=3, stride=2),          #7 (no padding)
            nn.BatchNorm2d(256),                                    #8
            nn.ReLU(),                                              #9
            
            # Final layer with risky kernel size
            nn.Conv2d(256, 512, kernel_size=7, stride=1),          #10
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Potential dimension mismatch in classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),    # 512 assumes successful forward pass
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

MODEL_CONFIG = {
    # ResNet Family
    "resnet18": (torchvision.models.resnet18, ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet34": (torchvision.models.resnet34, ResNet34_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (torchvision.models.resnet50, ResNet50_Weights.IMAGENET1K_V1, "fc"),
    "resnet101": (torchvision.models.resnet101, ResNet101_Weights.IMAGENET1K_V1, "fc"),
    
    # Other Models
    "vgg16": (torchvision.models.vgg16, VGG16_Weights.IMAGENET1K_V1, "classifier.6"),
    "mobilenetv2": (torchvision.models.mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V2, "classifier.1"),
    "vit": (torchvision.models.vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, "heads.head"),
    "alexnet": (torchvision.models.alexnet, AlexNet_Weights.IMAGENET1K_V1, "classifier.6"),
    "basic": (BasicModel, None, None)
}

def get_model(model_name, num_classes=4):
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(MODEL_CONFIG.keys())}")
    
    model_fn, weights, layer_key = MODEL_CONFIG[model_name]
    
    if model_name == "basic":
        return model_fn(num_classes=num_classes)
    
    model = model_fn(weights=weights)
    
    # Modify final layer
    if layer_key:
        layer_components = layer_key.split('.')
        current_layer = model
        for component in layer_components[:-1]:
            # Handle sequential indices
            if component.isdigit():
                current_layer = current_layer[int(component)]
            else:
                current_layer = getattr(current_layer, component)
                
        final_layer_name = layer_components[-1]
        
        # Handle index for final component
        if final_layer_name.isdigit():
            in_features = current_layer[int(final_layer_name)].in_features
            current_layer[int(final_layer_name)] = nn.Linear(in_features, num_classes)
        else:
            in_features = getattr(current_layer, final_layer_name).in_features
            setattr(current_layer, final_layer_name, nn.Linear(in_features, num_classes))
    
    return model