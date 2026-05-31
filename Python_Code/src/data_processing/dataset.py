import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# DATASET DEFINITION (Offline Preprocessed)
# ==============================================================================
def get_processed_dataset_stats(processed_dir, split='train'):
    data = []
    split_dir = os.path.join(processed_dir, split)
    if not os.path.exists(split_dir):
        return pd.DataFrame()
        
    for label_str in ['POSITIVE', 'NEGATIVE']:
        label_dir = os.path.join(split_dir, label_str)
        if not os.path.exists(label_dir): continue
        for f in os.listdir(label_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                label = 1 if label_str == "POSITIVE" else 0
                data.append({'filepath': os.path.join(label_dir, f), 'label': label})
    return pd.DataFrame(data)

# ==============================================================================
# RAM PRE-LOADING (eliminates disk I/O during training)
# ==============================================================================
def preload_to_device(df, logger, device):
    """
    Pre-loads all images from disk directly into GPU VRAM as normalized tensors.
    Called once per preprocessing method. All 5 models then train directly from VRAM.
    Zero disk I/O and zero CPU-GPU transfer during training.
    """
    to_tensor_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    labels = []
    logger.info(f"Pre-loading {len(df)} images from disk into CPU RAM...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        img = Image.open(row['filepath']).convert('RGB')
        img_tensor = to_tensor_normalize(img)
        images.append(img_tensor)
        labels.append(row['label'])
    
    images_tensor = torch.stack(images)          # [N, 3, 224, 224] on CPU
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    cpu_gb = images_tensor.nbytes / 1e9
    logger.info(f"Moving {cpu_gb:.2f} GB of image data to {device} VRAM...")
    
    # Move entire dataset to GPU VRAM — batches will be VRAM slices (900 GB/s internal bandwidth)
    images_tensor = images_tensor.to(device)
    labels_tensor = labels_tensor.to(device)
    
    logger.info(f"Data loaded to {device} VRAM successfully. Shape: {images_tensor.shape}")
    return images_tensor, labels_tensor

class RAMCachedDataset(Dataset):
    """
    Serves images directly from GPU VRAM tensors — zero disk I/O, zero CPU-GPU transfer.
    Augmentation transforms run on GPU tensors automatically.
    """
    def __init__(self, images_tensor, labels_tensor, augment=False):
        self.images = images_tensor  # GPU tensor
        self.labels = labels_tensor  # GPU tensor
        
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]    # Slice from GPU VRAM tensor — nanoseconds
        label = self.labels[idx]
        
        if self.aug_transform:
            img = self.aug_transform(img)  # Transforms run on GPU tensor
            
        return img, label

# ==============================================================================
# GPU OVERSAMPLING
# ==============================================================================
def balance_classes_on_gpu(train_images, train_labels, device, logger):
    """
    Performs random oversampling of the minority class directly on the GPU.
    """
    pos_indices = (train_labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (train_labels == 0).nonzero(as_tuple=True)[0]
    
    num_pos = len(pos_indices)
    num_neg = len(neg_indices)
    
    if num_neg > num_pos:
        logger.info(f"Oversampling train split: POSITIVE ({num_pos}) to match NEGATIVE ({num_neg})")
        extra_indices = torch.randint(0, num_pos, (num_neg - num_pos,), device=device)
        sampled_pos_indices = pos_indices[extra_indices]
        
        resampled_indices = torch.cat([pos_indices, neg_indices, sampled_pos_indices])
        shuffled_indices = resampled_indices[torch.randperm(len(resampled_indices))]
        
        return train_images[shuffled_indices], train_labels[shuffled_indices]
        
    elif num_pos > num_neg:
        logger.info(f"Oversampling train split: NEGATIVE ({num_neg}) to match POSITIVE ({num_pos})")
        extra_indices = torch.randint(0, num_neg, (num_pos - num_neg,), device=device)
        sampled_neg_indices = neg_indices[extra_indices]
        
        resampled_indices = torch.cat([pos_indices, neg_indices, sampled_neg_indices])
        shuffled_indices = resampled_indices[torch.randperm(len(resampled_indices))]
        
        return train_images[shuffled_indices], train_labels[shuffled_indices]
        
    return train_images, train_labels
