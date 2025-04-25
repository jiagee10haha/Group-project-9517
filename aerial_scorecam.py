from pytorch_grad_cam import ScoreCAM
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_scorecam_heatmap(model, image, target_layer, target_class):
    """
    Generate a Score-CAM heatmap to visualize the model's attention on an input image.

    Args:
        model: Trained model (e.g., ResNet-18)
        image (torch.Tensor): Input image tensor, shape (1, channels, height, width)
        target_layer: Target layer for CAM computation (e.g., model.layer4[-1])
        target_class (int): Target class index for the heatmap

    Returns:
        heatmap (numpy.ndarray): Score-CAM heatmap showing the model's attention
    """
    cam = ScoreCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    heatmap = cam(input_tensor=image, targets=targets)[0]
    return heatmap

def get_batch_scorecam_heatmaps(model, images, target_layer, target_classes):
    """
    Generate Score-CAM heatmaps for a batch of images.

    Args:
        model: Trained model
        images (torch.Tensor): Batch of input images, shape (batch_size, channels, height, width)
        target_layer: Target layer for CAM computation
        target_classes (list): List of target class indices for each image

    Returns:
        heatmaps (list): List of heatmaps for each image
    """
    heatmaps = []
    for i in range(images.size(0)):
        image = images[i].unsqueeze(0)
        target_class = target_classes[i]
        heatmap = get_scorecam_heatmap(model, image, target_layer, target_class)
        heatmaps.append(heatmap)
    return heatmaps

def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    """
    Overlay the heatmap on the original image.

    Args:
        image (torch.Tensor): Original image tensor, shape (channels, height, width)
        heatmap (numpy.ndarray): Heatmap to overlay
        alpha (float): Transparency factor for the heatmap

    Returns:
        overlaid_image (numpy.ndarray): Image with heatmap overlaid
    """
    # Denormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # Denormalize
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    image = (image * 255).astype(np.uint8)  # Scale to 0-255

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)

    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on the image
    overlaid_image = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid_image