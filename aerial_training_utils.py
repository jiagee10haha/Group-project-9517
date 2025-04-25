import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm.notebook import tqdm
import pandas as pd

def cutmix(inputs, labels, alpha=1.0):
    """
    Apply CutMix data augmentation: randomly crop and mix two images and their labels.

    Args:
        inputs (torch.Tensor): Batch of input images, shape (batch_size, channels, height, width)
        labels (torch.Tensor): Batch of labels, shape (batch_size,)
        alpha (float): Parameter for Beta distribution to control mixing ratio, default is 1.0

    Returns:
        mixed_inputs (torch.Tensor): Mixed batch of images
        labels_a (torch.Tensor): Original labels
        labels_b (torch.Tensor): Labels of the mixed portion
        lam (float): Mixing ratio for the original image
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    
    index = torch.randperm(batch_size).to(inputs.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
    
    return inputs, labels, labels[index], lam

def rand_bbox(size, lam):
    """
    Generate random bounding box coordinates for CutMix.

    Args:
        size (tuple): Image dimensions, format (batch_size, channels, height, width)
        lam (float): Proportion of the area to crop (0 to 1)

    Returns:
        bbx1, bby1, bbx2, bby2 (int): Coordinates of the top-left (bbx1, bby1) and bottom-right (bbx2, bby2) corners
    """
    W = size[2]
    H = size[3]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def evaluate_model(model, data_loader, device, class_names=None, scenario="Baseline", train_loss=None):
    """
    Evaluate the model's performance on a given dataset and generate metrics and visualizations.

    Args:
        model: Trained model (e.g., ResNet-18)
        data_loader (DataLoader): DataLoader for the dataset
        device (str): Device to run the model on ('cuda' or 'cpu')
        class_names (list): List of class names for visualization, default is None
        scenario (str): Evaluation scenario (e.g., "Baseline" or "Occlusion Test")
        train_loss (float): Final training loss of the model

    Returns:
        acc (float): Accuracy (percentage)
        f1 (float): Weighted F1 score
        prec (float): Weighted precision
        rec (float): Weighted recall
        weighted_iou (float): Weighted IoU score
        all_preds (list): List of predicted labels
        all_labels (list): List of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    
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
        
        print(f"Classification Report ({scenario}):")
        print(df_cr.to_string())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{scenario} - Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    return acc, f1, prec, rec, weighted_iou, all_preds, all_labels

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=5, use_cutmix=False, cutmix_prob=0.5):
    """
    Train a deep learning model with optional CutMix augmentation, evaluate performance per epoch, and display progress.

    Args:
        model: Model to train (e.g., ResNet-18)
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (e.g., Adam)
        device (str): Device to run the model on ('cuda' or 'cpu')
        epochs (int): Number of training epochs, default is 5
        use_cutmix (bool): Whether to apply CutMix augmentation, default is False
        cutmix_prob (float): Probability of applying CutMix, default is 0.5

    Returns:
        best_acc (float): Best test accuracy achieved
        training_time (float): Total training time in seconds
        final_train_loss (float): Final training loss
    """
    model.to(device)
    best_acc = 0
    start_time = time.time()
    final_train_loss = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_cutmix and np.random.rand() < cutmix_prob:
                mixed_inputs, labels_a, labels_b, lam = cutmix(inputs, labels)
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_acc = 100 * correct_train / total_train
            
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1), train_acc=train_acc)
        
        final_train_loss = running_loss / len(train_loader)
        # Evaluate on training and test sets to get IoU
        train_acc, _, _, _, train_iou, _, _ = evaluate_model(model, train_loader, device)
        test_acc, _, _, _, test_iou, _, _ = evaluate_model(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Print metrics including IoU
        print(f"Epoch {epoch+1}/{epochs}, Train loss: {final_train_loss:.4f}, Train Accuracy: {train_acc:.2f}, Train IoU: {train_iou:.3f}, Test Accuracy: {test_acc:.2f}, Test IoU: {test_iou:.3f}")
    
    training_time = time.time() - start_time
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    
    return best_acc, training_time, final_train_loss