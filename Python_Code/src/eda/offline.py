import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
from src.eda.preprocessing import get_dataset_stats, apply_original, apply_he, apply_gamma, apply_clahe, setup_logger

logger = setup_logger()

def process_and_save_dataset():
    root_data_dir = r"C:\Users\laimm\Processed_Datasets\Datasets"
    out_dir = r"C:\Users\laimm\Processed_Datasets"
    
    logger.info("Scanning original dataset for offline preprocessing...")
    df = get_dataset_stats(root_data_dir)
    if df.empty:
        logger.error("No images found.")
        return
        
    methods = {
        'original': apply_original,
        'he': apply_he,
        'gamma': apply_gamma,
        'clahe': apply_clahe
    }
    
    for method, func in methods.items():
        logger.info(f"--- Processing method: {method} ---")
        method_out_dir = os.path.join(out_dir, method)
        
        # Using tqdm for progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {method}"):
            filepath = row['filepath']
            label = row['label']
            split = row['split']
            dataset = row['dataset']
            
            # Format: Processed_Datasets/<method>/<split>/<label>/<dataset>_<filename>
            basename = os.path.basename(filepath)
            new_filename = f"{dataset}_{basename}"
            # Ensure it is saved as jpg or png
            if new_filename.lower().endswith(('.tif', '.tiff', '.bmp')):
                new_filename = os.path.splitext(new_filename)[0] + '.jpg'
                
            save_path_dir = os.path.join(method_out_dir, split, label)
            os.makedirs(save_path_dir, exist_ok=True)
            
            save_path = os.path.join(save_path_dir, new_filename)
            
            if not os.path.exists(save_path):
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        processed_img = func(img)
                        # Save image
                        processed_img.save(save_path)
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {e}")
                    
    logger.info("Offline preprocessing completed successfully!")

if __name__ == "__main__":
    process_and_save_dataset()
