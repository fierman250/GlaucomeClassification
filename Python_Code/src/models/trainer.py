import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import gc

# Import from refactored modules
from src.eda.preprocessing import setup_logger
from src.models.classifiers import get_model
from src.data_processing.dataset import get_processed_dataset_stats, preload_to_device, RAMCachedDataset, balance_classes_on_gpu

logger = logging.getLogger('GlaucomaPipeline')
if not logger.handlers:
    logger = setup_logger()

# ==============================================================================
# METRICS AND PLOTTING
# ==============================================================================
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        tn, fp, fn, tp = (cm[0,0], 0, 0, 0) if y_true[0] == 0 else (0, 0, 0, cm[0,0])
    else:
        tn, fp, fn, tp = cm.ravel()
        
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    sensitivity = tp / (tp + fn + 1e-7) # Recall
    specificity = tn / (tn + fp + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-7)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'auc': roc_auc,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=['NEGATIVE', 'POSITIVE'], yticklabels=['NEGATIVE', 'POSITIVE'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    # Initialize AMP GradScaler (modern PyTorch syntax)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Data is already on GPU VRAM — no transfer needed
            optimizer.zero_grad()
            
            # Use AMP autocast (modern PyTorch syntax)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_pred, y_prob = [], [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Data is already on GPU VRAM — no transfer needed
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            import copy
            best_model_weights = copy.deepcopy(model.state_dict())
            
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Data is already on GPU VRAM
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    final_metrics = calculate_metrics(y_true, y_pred, y_prob)
    return model, final_metrics


