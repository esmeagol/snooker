import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on images or videos')
    parser.add_argument('--source', type=str, default='0', help='Path to input image/video or 0 for webcam')
    parser.add_argument('--model', type=str, default='runs/detect/yolov8n_snooker/weights/best.pt', 
                        help='Path to the trained model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save the output')
    parser.add_argument('--output-dir', type=str, default='inference_results', help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if source is webcam
    if args.source == '0':
        args.source = 0
        is_webcam = True
    else:
        is_webcam = False
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Class names
    class_names = ['black', 'blue', 'brown', 'green', 'pink', 'red', 'white_cueball', 'yellow']
    
    # Colors for different classes
    colors = {
        'red': (0, 0, 255),        # Red
        'yellow': (0, 255, 255),   # Yellow
        'green': (0, 255, 0),      # Green
        'brown': (42, 42, 165),    # Brown
        'blue': (255, 0, 0),       # Blue
        'pink': (203, 192, 255),   # Pink
        'black': (0, 0, 0),        # Black
        'white_cueball': (255, 255, 255)  # White
    }
    
    # Process the source
    if isinstance(args.source, str) and args.source.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Single image
        image = cv2.imread(args.source)
        if image is None:
            print(f"Error: Could not read image {args.source}")
            return
            
        # Run inference
        results = model(image, conf=args.conf)
        
        # Process results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = class_names[cls_id]
                
                # Draw bounding box
                color = colors.get(cls_name, (0, 255, 0))  # Default to green if class not in colors
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = f"{cls_name} {conf:.2f}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
                
                # Put text on image
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Show or save the result
        if args.save:
            output_path = os.path.join(args.output_dir, os.path.basename(args.source))
            cv2.imwrite(output_path, image)
            print(f"Result saved to {output_path}")
        
        # Show the image
        cv2.imshow('Inference Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        # Video or webcam
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if saving
        if args.save and not is_webcam:
            output_path = os.path.join(args.output_dir, f"detected_{os.path.basename(args.source)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Press 'q' to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            results = model(frame, conf=args.conf)
            
            # Process results
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confs, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = class_names[cls_id]
                    
                    # Draw bounding box
                    color = colors.get(cls_name, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"{cls_name} {conf:.2f}"
                    
                    # Draw label background
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
                    
                    # Put text on frame
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save the frame if needed
            if args.save and not is_webcam:
                out.write(frame)
            
            # Show the frame
            cv2.imshow('Inference', frame)
            
            # Break the loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if args.save and not is_webcam:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
