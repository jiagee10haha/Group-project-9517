import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def apply_occlusion(image, occlusion_size=32):
    """
    Apply a random occlusion to an image to test model robustness.

    Args:
        image (torch.Tensor): Input image tensor, shape (channels, height, width) or (1, channels, height, width)
        occlusion_size (int): Size of the occlusion square, default is 32 pixels

    Returns:
        torch.Tensor: Image with the occlusion applied, occluded area filled with 0 (black)
    """
    if image.dim() == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.dim() != 3:
        raise ValueError(f"Expected image with shape (channels, height, width), but got {image.shape}")
    
    h, w = image.shape[1], image.shape[2]
    occlusion_size = min(occlusion_size, h, w)
    if occlusion_size <= 0:
        return image
    
    x = np.random.randint(0, w - occlusion_size + 1)
    y = np.random.randint(0, h - occlusion_size + 1)
    occluded_image = image.clone()
    occluded_image[:, y:y+occlusion_size, x:x+occlusion_size] = 0
    return occluded_image

def evaluate_with_occlusion(model, test_loader, occlusion_size=32, device='cuda', class_names=None, train_loss=None):
    """
    Evaluate the model's performance on occluded images and generate metrics and visualizations.

    Args:
        model: Trained model (e.g., ResNet-18)
        test_loader (DataLoader): DataLoader for the test set
        occlusion_size (int): Size of the occlusion square, default is 32 pixels
        device (str): Device to run the model on ('cuda' or 'cpu')
        class_names (list): List of class names for visualization, default is None
        train_loss (float): Final training loss of the model

    Returns:
        accuracy (float): Accuracy on occluded images (percentage)
        f1 (float): Weighted F1 score
        precision (float): Weighted precision
        recall (float): Weighted recall
        weighted_iou (float): Weighted IoU score
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            occluded_inputs = torch.stack([apply_occlusion(inp, occlusion_size) for inp in inputs])
            outputs = model(occluded_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    
    weighted_iou = 0.0
    if class_names is not None:
        cm = confusion_matrix(all_labels, all_preds)
        cr = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        
        # Calculate IoU for each class
        iou_per_class = []
        for i in range(len(class_names)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            iou_per_class.append(iou)
        
        # Calculate weighted IoU
        supports = [cr[cls]['support'] for cls in class_names]
        total_support = sum(supports)
        weighted_iou = sum(iou * support / total_support for iou, support in zip(iou_per_class, supports))
        
        # Add IoU and Train loss to the report
        for i, cls in enumerate(class_names):
            cr[cls]['iou'] = iou_per_class[i]
            cr[cls]['train_loss'] = train_loss if train_loss is not None else 0.0
        
        # Convert to DataFrame and select desired columns
        df_cr = pd.DataFrame(cr).T
        df_cr = df_cr[['precision', 'recall', 'f1-score', 'support', 'iou', 'train_loss']]
        
        print(f"Classification Report for Occlusion Test (Size {occlusion_size}):")
        print(df_cr.to_string())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Occlusion Test - Confusion Matrix (Size {occlusion_size})")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    return acc, f1, prec, rec, weighted_iou