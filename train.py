from ultralytics import YOLO
import os

def train_model():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')  # You can also use yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # Train the model
    results = model.train(
        data='snooker_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='0',  # Use '0' for GPU or 'cpu' for CPU
        workers=8,
        project='snooker_detection',
        name='yolov8n_snooker',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        seed=42,
        close_mosaic=10,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        fl_gamma=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )

if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs('runs', exist_ok=True)
    
    print("Starting YOLOv8 training...")
    train_model()
    print("Training complete! Check the 'runs' directory for results.")
