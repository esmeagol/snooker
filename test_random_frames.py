import os
import random
import glob
import cv2
import gc
from ultralytics import YOLO
from pathlib import Path
import torch

# Configuration
MODEL_PATH = './snooker_detection/yolov8n_snooker/weights/best.pt'
SAMPLE_SIZE = 3  # Reduced number of frames to test
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_DIR = 'inference_results'

# Colors for different classes
COLORS = {
    'black': (0, 0, 0),
    'blue': (255, 0, 0),
    'brown': (42, 42, 165),
    'green': (0, 255, 0),
    'pink': (203, 192, 255),
    'red': (0, 0, 255),
    'white_cueball': (255, 255, 255),
    'yellow': (0, 255, 255)
}

def get_random_frames(directory, sample_size):
    """Get a list of random image files from the directory."""
    image_files = glob.glob(os.path.join(directory, '*.jpg'))
    return random.sample(image_files, min(sample_size, len(image_files)))

def process_single_image(model, img_path):
    """Process a single image and return the result."""
    try:
        # Read the image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error: Could not read image {img_path}")
            return None
            
        # Run inference with reduced size for memory efficiency
        results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=640)
        
        # Process results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Draw detections
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[cls_id]
                color = COLORS.get(cls_name, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = f"{cls_name} {conf:.2f}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
                
                # Put text on image
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Set model to evaluation mode
    model.model.eval()
    
    # Get random frames
    frames_dir = '/Users/abhinavrai/Playground/snooker/data/raw_frames'
    print(f"Sampling {SAMPLE_SIZE} random frames from {frames_dir}...")
    random_frames = get_random_frames(frames_dir, SAMPLE_SIZE)
    
    # Process each frame one at a time
    for i, img_path in enumerate(random_frames):
        print(f"\nProcessing image {i+1}/{len(random_frames)}: {os.path.basename(img_path)}")
        
        # Process the image
        result_frame = process_single_image(model, img_path)
        if result_frame is None:
            continue
        
        # Save the result
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(output_path, result_frame)
        print(f"Saved result to {output_path}")
        
        # Display the result
        cv2.imshow('Inference Result', result_frame)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break
            
        # Clear memory
        del result_frame
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    cv2.destroyAllWindows()
    print("\nDone! Check the 'inference_results' directory for the output images.")

if __name__ == "__main__":
    main()
