import os
import shutil
from pathlib import Path
import yaml

def prepare_dataset(source_dir, target_dir):
    """
    Prepare the dataset for YOLOv8 training by organizing it into the required directory structure.
    
    Args:
        source_dir (str): Path to the source dataset directory
        target_dir (str): Path to the target directory where the dataset will be prepared
    """
    # Create target directory structure
    target_dir = Path(target_dir)
    (target_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (target_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (target_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (target_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Copy training images and labels
    print("Copying training data...")
    train_img_src = Path(source_dir) / 'train' / 'images'
    train_lbl_src = Path(source_dir) / 'train' / 'labels'
    
    for img_file in train_img_src.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy2(img_file, target_dir / 'images' / 'train' / img_file.name)
    
    for lbl_file in train_lbl_src.glob('*'):
        if lbl_file.suffix.lower() == '.txt':
            shutil.copy2(lbl_file, target_dir / 'labels' / 'train' / lbl_file.name)
    
    # Copy validation images and labels
    print("Copying validation data...")
    val_img_src = Path(source_dir) / 'valid' / 'images'
    val_lbl_src = Path(source_dir) / 'valid' / 'labels'
    
    for img_file in val_img_src.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy2(img_file, target_dir / 'images' / 'val' / img_file.name)
    
    for lbl_file in val_lbl_src.glob('*'):
        if lbl_file.suffix.lower() == '.txt':
            shutil.copy2(lbl_file, target_dir / 'labels' / 'val' / lbl_file.name)
    
    # Create dataset YAML file
    dataset_yaml = {
        'path': str(target_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'black',
            1: 'blue',
            2: 'brown',
            3: 'green',
            4: 'pink',
            5: 'red',
            6: 'white_cueball',
            7: 'yellow'
        },
        'nc': 8
    }
    
    with open(target_dir / 'snooker_dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"\nDataset prepared at: {target_dir}")
    print(f"Dataset YAML file: {target_dir / 'snooker_dataset.yaml'}")
    print("\nYou can now start training with:")
    print(f"yolo task=detect mode=train model=yolov8n.pt data={target_dir / 'snooker_dataset.yaml'}")

if __name__ == "__main__":
    source_dataset = "/Users/abhinavrai/Downloads/snooker images/ball.v2i.yolov8"
    target_dataset = "data/snooker_dataset"
    
    prepare_dataset(source_dataset, target_dataset)
