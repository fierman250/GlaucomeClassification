import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from PIL import Image

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def setup_logger():
    logger = logging.getLogger('GlaucomaPipeline')
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if not logger.handlers:
        # Console Handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        
        # File Handler
        f_handler = logging.FileHandler('glaucoma_pipeline.log')
        f_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

logger = setup_logger()

# ==============================================================================
# EDA (Exploratory Data Analysis)
# ==============================================================================
def get_dataset_stats(root_dir):
    """
    Scans the directory structure and computes class balance statistics.
    Returns a pandas dataframe of all found images and their labels.
    """
    logger.info(f"Scanning dataset directory: {root_dir}")
    data = []
    
    # Train set
    train_dir = os.path.join(root_dir, "train")
    if os.path.isdir(train_dir):
        for dataset in os.listdir(train_dir):
            dataset_path = os.path.join(train_dir, dataset)
            if not os.path.isdir(dataset_path): continue
            
            for root, _, files in os.walk(dataset_path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                        file_path = os.path.join(root, f)
                        # Determine label
                        path_parts = root.replace(dataset_path, "").split(os.sep)
                        dirname = os.path.basename(root)
                        is_pos = any("POSITIVE" in part for part in path_parts) or "POSITIVE" in dirname
                        is_neg = any("NEGATIVE" in part for part in path_parts) or "NEGATIVE" in dirname
                        
                        if is_pos:
                            label = "POSITIVE"
                        elif is_neg:
                            label = "NEGATIVE"
                        else:
                            continue # Skip if unable to determine
                        
                        data.append({'filepath': file_path, 'label': label, 'split': 'train', 'dataset': dataset})

    # Test set
    test_dir = os.path.join(root_dir, "test")
    if os.path.isdir(test_dir):
        for cls in ['POSITIVE', 'NEGATIVE']:
            cls_path = os.path.join(test_dir, cls)
            if not os.path.isdir(cls_path): continue
            for dataset in os.listdir(cls_path):
                dataset_path = os.path.join(cls_path, dataset)
                if not os.path.isdir(dataset_path): continue
                for root, _, files in os.walk(dataset_path):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                            file_path = os.path.join(root, f)
                            data.append({'filepath': file_path, 'label': cls, 'split': 'test', 'dataset': dataset})
    
    df = pd.DataFrame(data)
    logger.info(f"Total images found: {len(df)}")
    if len(df) > 0:
        logger.info(f"Class Distribution: \n{df['label'].value_counts()}")
    return df

def plot_class_distribution(df, save_path="eda_class_distribution.png"):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label', hue='split')
    plt.title('Class Distribution (Train vs Test)')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved class distribution plot to {save_path}")

# ==============================================================================
# PREPROCESSING METHODS
# ==============================================================================
# These functions expect a numpy array (RGB image, HxWxC) or a PIL Image
# and return a PIL image (since torchvision transforms work well with PIL)

def apply_original(image):
    """
    Baseline: just resize, no enhancements.
    Assumes image is PIL Image or RGB numpy array.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.resize((224, 224))

def apply_he(image):
    """
    Histogram Equalization on L channel of LAB space.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Equalize the L channel
    l_eq = cv2.equalizeHist(l)
    
    # Merge back and convert to RGB
    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(result).resize((224, 224))

def apply_gamma(image, gamma=1.2):
    """
    Gamma Correction (Power-law transformation).
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    result = cv2.LUT(image, table)
    
    return Image.fromarray(result).resize((224, 224))

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on grayscale image.
    Matches MATLAB's rgb2gray + adapthisteq.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_clahe = clahe.apply(gray)
    
    # Convert back to 3-channel RGB (so it is compatible with CNNs expecting RGB)
    result = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(result).resize((224, 224))

def preprocess_image(filepath, method="original"):
    """
    Loads an image from filepath and applies the requested preprocessing method.
    """
    try:
        # Load image as RGB
        img = cv2.imread(filepath)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {filepath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if method == "original":
            return apply_original(img)
        elif method == "he":
            return apply_he(img)
        elif method == "gamma":
            return apply_gamma(img)
        elif method == "clahe":
            return apply_clahe(img)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
            
    except Exception as e:
        logger.error(f"Error processing {filepath} with {method}: {str(e)}")
        # Return a black 224x224 image as fallback
        return Image.new("RGB", (224, 224))

if __name__ == "__main__":
    logger.info("Running EDA and Preprocessing test...")
    # Example usage / test run
    root_data = r"C:\Users\laimm\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Datasets Sources\RefA1 - Fundus Glaucoma Detection Data"
    df = get_dataset_stats(root_data)
    if len(df) > 0:
        plot_class_distribution(df, os.path.join(os.path.dirname(__file__), "eda_class_distribution.png"))
        logger.info("EDA completed successfully.")
