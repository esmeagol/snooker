import os
import random
import shutil
from tqdm import tqdm

def split_dataset(input_dir, output_base, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        input_dir (str): Directory containing all the frames
        output_base (str): Base directory for output datasets
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
    """
    # Validate ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)
    
    # Get all frame files
    frame_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(frame_files)
    
    # Calculate split indices
    total = len(frame_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split the files
    splits_files = {
        'train': frame_files[:train_end],
        'val': frame_files[train_end:val_end],
        'test': frame_files[val_end:]
    }
    
    # Copy files to respective directories
    for split, files in splits_files.items():
        print(f"Processing {split} set ({len(files)} images)...")
        for file in tqdm(files, desc=f"Copying {split}"):
            src_path = os.path.join(input_dir, file)
            dst_path = os.path.join(output_base, split, 'images', file)
            shutil.copy2(src_path, dst_path)
    
    print("\nDataset split complete!")
    for split in splits:
        count = len(os.listdir(os.path.join(output_base, split, 'images')))
        print(f"{split}: {count} images")

if __name__ == "__main__":
    # Define paths
    input_dir = "data/raw_frames"
    output_base = "data/yolo_dataset"
    
    # Split the dataset
    split_dataset(input_dir, output_base, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
