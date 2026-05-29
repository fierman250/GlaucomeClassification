# %% [markdown]
# # Glaucoma Classification - Interactive Training
# Use this script to run the training process step-by-step. 
# In VS Code, you can click "Run Cell" above each `#%%` block to execute it and inspect variables.

# %%
# ==============================================================================
# STEP 1: Configuration & Imports
# ==============================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import copy
import logging

# Import modular components
from src.eda.preprocessing import setup_logger
from src.data_processing.dataset import get_processed_dataset_stats, preload_to_device, RAMCachedDataset, balance_classes_on_gpu
from src.models.classifiers import get_model
from src.models.trainer import calculate_metrics, plot_confusion_matrix, plot_roc_curve

# Configuration
PROCESSED_DATA_DIR = r"C:\Users\laimm\Processed_Datasets\original"
RESULTS_DIR = "results/interactive_run"
MODEL_NAME = "resnet18"  # Options: 'alexnet', 'vgg16', 'densenet121', 'resnet18', 'mobilenet_v2'
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

os.makedirs(RESULTS_DIR, exist_ok=True)
logger = setup_logger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Configuration set. Using device: {device}")

# %%
# ==============================================================================
# STEP 2: Load Dataset Metadata
# ==============================================================================
print(f"Scanning for preprocessed datasets in: {PROCESSED_DATA_DIR}")
df_train = get_processed_dataset_stats(PROCESSED_DATA_DIR, split='train')

if df_train.empty:
    print(f"ERROR: No data found in {PROCESSED_DATA_DIR}. Make sure offline processing has run.")
else:
    print(f"Total images found: {len(df_train)}")
    print("\nClass Distribution:")
    print(df_train['label'].value_counts().rename({0: "NEGATIVE", 1: "POSITIVE"}))

# %%
# ==============================================================================
# STEP 3: VRAM Preloading (Load dataset directly to GPU memory)
# ==============================================================================
# This step eliminates disk I/O bottlenecks during training
all_images, all_labels = preload_to_device(df_train, logger, device)

print(f"\nImages loaded into VRAM. Shape: {all_images.shape}")
print(f"Labels loaded into VRAM. Shape: {all_labels.shape}")

# %%
# ==============================================================================
# STEP 4: Data Splitting & GPU Oversampling
# ==============================================================================
# We use a standard 80/20 train/val split for interactive training (instead of K-Fold)
print("Splitting data into 80% Training and 20% Validation...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
y_cpu = all_labels.cpu().numpy()

train_idx, val_idx = next(sss.split(all_images, y_cpu))

# Slicing tensors dynamically in VRAM
train_images = all_images[train_idx]
train_labels = all_labels[train_idx]
val_images = all_images[val_idx]
val_labels = all_labels[val_idx]

print(f"Before balancing - Train Positive: {(train_labels == 1).sum().item()}, Train Negative: {(train_labels == 0).sum().item()}")

# Balance classes (Oversampling) - DISABLED due to already balanced dataset
# train_images, train_labels = balance_classes_on_gpu(train_images, train_labels, device, logger)
# print(f"After balancing  - Train Positive: {(train_labels == 1).sum().item()}, Train Negative: {(train_labels == 0).sum().item()}")

# Create DataLoaders
train_dataset = RAMCachedDataset(train_images, train_labels, augment=True)
val_dataset = RAMCachedDataset(val_images, val_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("DataLoaders are ready!")

# %%
# ==============================================================================
# STEP 5: Model Initialization
# ==============================================================================
model = get_model(MODEL_NAME, num_classes=2, pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda')

print(f"{MODEL_NAME.upper()} model initialized and moved to {device}.")

# %%
# ==============================================================================
# STEP 6: Interactive Training Loop
# ==============================================================================
# You can run this block and observe the loss decrease epoch by epoch.
print(f"Starting training for {EPOCHS} epochs...")

best_val_loss = float('inf')
best_model_weights = None

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())

# Load best weights back into model
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
print("Training Complete. Best model weights restored.")

# Plot Learning Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, marker='o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), val_accuracies, marker='o', label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
curve_path = os.path.join(RESULTS_DIR, "learning_curves.png")
plt.savefig(curve_path)
print(f"Learning curves saved to {curve_path}")

try:
    from PIL import Image
    from IPython.display import display
    display(Image.open(curve_path))
except ImportError:
    pass

# %%
# ==============================================================================
# STEP 7: Evaluation & Metrics
# ==============================================================================
print("Evaluating best model on Validation Set...")
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

# Calculate Metrics
metrics = calculate_metrics(y_true, y_pred, y_prob)

print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"AUC:         {metrics['auc']:.4f}")

# Plot Confusion Matrix
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plot_confusion_matrix(metrics['cm'], "Validation Confusion Matrix", cm_path)

# Plot ROC Curve
roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], "Validation ROC Curve", roc_path)

print(f"Plots saved to {RESULTS_DIR}")

# Display plots inline if running in Jupyter/Interactive Window
try:
    from PIL import Image
    from IPython.display import display
    display(Image.open(cm_path))
    display(Image.open(roc_path))
except ImportError:
    print("Could not display images inline (IPython not found). Check the results directory.")

# %%
# ==============================================================================
# STEP 8: Save Model Weights
# ==============================================================================
save_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_best_weights.pth")
torch.save(model.state_dict(), save_path)
print(f"Model successfully saved to: {save_path}")

# %%
# ==============================================================================
# STEP 9: Final Evaluation on Dedicated Test Set
# ==============================================================================
print("\n--- Final Test Set Evaluation ---")
df_test = get_processed_dataset_stats(PROCESSED_DATA_DIR, split='test')

if df_test.empty:
    print(f"ERROR: No test data found in {PROCESSED_DATA_DIR}\\test.")
else:
    print(f"Total test images found: {len(df_test)}")
    print(df_test['label'].value_counts().rename({0: "NEGATIVE", 1: "POSITIVE"}))
    
    # Preload Test Set
    test_images, test_labels = preload_to_device(df_test, logger, device)
    test_dataset = RAMCachedDataset(test_images, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("\nRunning inference on Test Set...")
    model.eval()
    y_true_test, y_pred_test, y_prob_test = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())
            y_prob_test.extend(probs.cpu().numpy())
            
    # Calculate Metrics
    test_metrics = calculate_metrics(y_true_test, y_pred_test, y_prob_test)
    
    print("\n[TEST SET METRICS]")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"AUC:         {test_metrics['auc']:.4f}")
    
    # Plot Test Confusion Matrix
    test_cm_path = os.path.join(RESULTS_DIR, "test_confusion_matrix.png")
    plot_confusion_matrix(test_metrics['cm'], "Test Set Confusion Matrix", test_cm_path)
    print(f"Test Confusion Matrix saved to {test_cm_path}")
    
    try:
        display(Image.open(test_cm_path))
    except ImportError:
        pass

