import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class CustomGradCAM:
    """
    Custom implementation of Grad-CAM to avoid external dependencies like pytorch-gradcam,
    ensuring stability in the SimTR environment.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t the activation
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates the Grad-CAM heatmap.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        target = output[0, target_class]
        
        # Backward pass
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients to get weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to cam
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, target_class, output[0, target_class].item()

def overlay_cam_on_image(img_path, cam, alpha=0.5):
    """
    Overlays the CAM heatmap on the original image.
    img_path: path to original image.
    cam: 2D numpy array containing the CAM (normalized [0,1]).
    """
    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize cam to match image size
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # Convert cam to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

def get_target_layer(model, model_name):
    """
    Returns the appropriate target layer for Grad-CAM depending on the architecture.
    """
    model_name = model_name.lower()
    if model_name == 'alexnet' or model_name == 'vgg16':
        return model.features[-1]
    elif model_name == 'densenet121':
        return model.features.denseblock4.denselayer16.conv2
    elif model_name == 'resnet18':
        return model.layer4[-1].conv2
    elif model_name == 'mobilenet_v2':
        return model.features[-1]
    else:
        raise ValueError("Unsupported model architecture for Grad-CAM target layer mapping.")

# For standalone testing
if __name__ == "__main__":
    from src.models.classifiers import get_model
    from src.eda.preprocessing import preprocess_image
    
    # Test Grad-CAM with a dummy image and model if available
    dummy_model = get_model('resnet18', num_classes=2, pretrained=True)
    target_layer = get_target_layer(dummy_model, 'resnet18')
    cam_extractor = CustomGradCAM(dummy_model, target_layer)
    print("Grad-CAM initialized successfully.")
