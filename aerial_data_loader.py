import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

def get_data_loaders(data_dir, batch_size=64, train_ratio=0.8, sample_ratio=1, imbalance=False):
    """
    Load the Aerial_Landscapes dataset and perform stratified sampling to create DataLoader objects for training and testing.

    Args:
        data_dir (str): Path to the dataset root directory, e.g., '/content/drive/MyDrive/Aerial_Landscapes'
        batch_size (int): Number of samples per batch, default is 64
        train_ratio (float): Proportion of the dataset to use for training, default is 0.8 (80% train, 20% test)
        sample_ratio (float): Proportion of samples to use from each class, default is 1 (100%)
        imbalance (bool): Whether to create an imbalanced training set, default is False

    Returns:
        train_loader1 (DataLoader): DataLoader for training set with augmentation strategy 1
        train_loader2 (DataLoader): DataLoader for training set with augmentation strategy 2
        test_loader (DataLoader): DataLoader for test set
    """
    # Augmentation strategy 1: Random flips, crops, and color adjustments to improve generalization
    transform1 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Augmentation strategy 2: Random rotation, paired with CutMix in training
    transform2 = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test set preprocessing: Resize and normalize, no random transformations for consistency
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the entire dataset using ImageFolder, automatically identifying 15 classes
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Extract labels for stratified sampling
    labels = np.array([label for _, label in full_dataset.samples])
    
    # Sample a proportion of the data from each class
    classes = np.unique(labels)
    selected_indices = []
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        num_samples = int(len(cls_indices) * sample_ratio)
        selected_cls_indices = np.random.choice(cls_indices, num_samples, replace=False)
        selected_indices.extend(selected_cls_indices)
    
    # Create a subset of the dataset
    selected_dataset = Subset(full_dataset, selected_indices)
    selected_labels = labels[selected_indices]
    
    # Perform stratified train-test split to ensure balanced class distribution
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=42
    )
    
    if imbalance:
        # Create imbalanced training set
        train_labels = labels[train_indices]
        unique, counts = np.unique(train_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Define imbalance ratios (example: linear decrease)
        imbalance_ratios = {cls: 1 - (cls / (len(classes) - 1)) for cls in classes}
        
        # Sample training indices based on imbalance ratios
        imbalanced_train_indices = []
        for cls in classes:
            cls_indices = np.where(train_labels == cls)[0]
            num_samples = int(len(cls_indices) * imbalance_ratios[cls])
            selected_cls_indices = np.random.choice(cls_indices, num_samples, replace=False)
            imbalanced_train_indices.extend(selected_cls_indices)
        
        train_indices = [train_indices[i] for i in imbalanced_train_indices]
    
    # Create training and test subsets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create two versions of the training set with different augmentations
    train_dataset1 = datasets.ImageFolder(root=data_dir, transform=transform1)
    train_dataset2 = datasets.ImageFolder(root=data_dir, transform=transform2)
    
    # Filter training sets with the selected indices
    train_dataset1 = Subset(train_dataset1, train_indices)
    train_dataset2 = Subset(train_dataset2, train_indices)
    
    # Apply preprocessing to the test set
    test_dataset.dataset.transform = test_transform
    
    # Create DataLoader objects for batch processing
    train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader1, train_loader2, test_loader