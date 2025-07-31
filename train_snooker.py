from ultralytics import YOLO
import os

def train_model():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')  # Using YOLOv8n (nano) for faster training, can use yolov8s/m/l/x for better accuracy
    
    # Check if MPS (Apple Silicon GPU) is available
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")
    
    # Training configuration
    args = {
        'data': 'training_data/snooker_dataset/snooker_dataset.yaml',
        'epochs': 50,  # Reduced epochs for stability
        'imgsz': 640,
        'batch': 4,  # Reduced batch size for MPS stability
        'device': 'cpu',  # Using CPU for stability
        'workers': 4,  # Reduced workers for stability
        'project': 'snooker_detection',
        'name': 'yolov8n_snooker',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',  # Using Adam for better stability
        'seed': 42,
        'lr0': 0.001,  # Reduced learning rate
        'lrf': 0.01,
        'momentum': 0.9,  # Reduced momentum
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.5,  # Reduced warmup momentum
        'warmup_bias_lr': 0.1,
        'box': 5.0,  # Reduced box loss gain
        'cls': 0.5,
        'dfl': 1.0,  # Reduced DFL gain
        'hsv_h': 0.01,  # Reduced HSV augmentation
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 0.0,  # Disabled rotation
        'translate': 0.1,
        'scale': 0.2,  # Reduced scale
        'flipud': 0.0,  # Disabled flip up-down
        'fliplr': 0.5,
        'mosaic': 0.0,  # Disabled mosaic for stability
        'mixup': 0.0,  # Disabled mixup
        'copy_paste': 0.0,  # Disabled copy-paste
    }
    
    print("Training with the following configuration:")
    for k, v in args.items():
        print(f"  {k}: {v}")
    
    # Start training
    results = model.train(**args)

if __name__ == "__main__":
    print("Starting YOLOv8 training for snooker ball detection...")
    print("Training data: training_data/snooker_dataset")
    print("Model will be saved in: runs/detect/yolov8n_snooker")
    print("\nTraining parameters:")
    print("- Epochs: 100")
    print("- Image size: 640x640")
    print("- Batch size: 16")
    print("- Device: GPU (if available)")
    
    train_model()
    print("\nTraining complete! Check the 'runs/detect/yolov8n_snooker' directory for results.")
    print("You can use the trained model with: model = YOLO('runs/detect/yolov8n_snooker/weights/best.pt')")
