""# Snooker Ball Detection with YOLOv8

This notebook demonstrates how to use the trained YOLOv8 model to detect snooker balls in images and videos.
"""

# 1. Install Dependencies
# !pip install ultralytics opencv-python-headless matplotlib ipywidgets

# 2. Import Required Libraries
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import display, HTML
from IPython.display import Image as IPImage
from ipywidgets import interact, widgets, Layout

# 3. Load the Trained Model
# Path to the trained model
MODEL_PATH = './snooker_detection/yolov8n_snooker/weights/best.pt'

# Load the model
model = YOLO(MODEL_PATH)

# Set model to evaluation mode
model.model.eval()

# Class names
class_names = ['black', 'blue', 'brown', 'green', 'pink', 'red', 'white_cueball', 'yellow']

# Colors for different classes
colors = {
    'black': (0, 0, 0),
    'blue': (255, 0, 0),
    'brown': (42, 42, 165),
    'green': (0, 255, 0),
    'pink': (203, 192, 255),
    'red': (0, 0, 255),
    'white_cueball': (255, 255, 255),
    'yellow': (0, 255, 255)
}

print("Model loaded successfully!")
print(f"Model classes: {class_names}")

# 4. Helper Functions for Visualization
def plot_image(image, bgr=False):
    """Display an image in the notebook."""
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def draw_detections(image, results, conf_threshold=0.5):
    """Draw bounding boxes and labels on the image."""
    # Make a copy of the image
    img = image.copy()
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confs, class_ids):
            if conf < conf_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            cls_name = class_names[cls_id]
            color = colors.get(cls_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{cls_name} {conf:.2f}"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
            
            # Put text on image
            cv2.putText(img, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

# 5. Run Inference on an Image
def detect_image(image_path, conf_threshold=0.5):
    """Run detection on a single image and display results."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run inference
    results = model(image, conf=conf_threshold)
    
    # Draw detections
    result_image = draw_detections(image, results, conf_threshold)
    
    # Display results
    plot_image(result_image, bgr=True)
    
    # Return results for further processing
    return results

# 6. Run Inference on a Video
def detect_video(video_path, output_path='output.mp4', conf_threshold=0.5, max_frames=None):
    """Run detection on a video and save the results."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit the number of frames if specified
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frame by frame
    frame_count = 0
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)
        
        # Draw detections
        result_frame = draw_detections(frame, results, conf_threshold)
        
        # Write the frame
        out.write(result_frame)
        
        # Display progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}")

# 7. Interactive Demo
def interactive_demo():
    """Create an interactive demo for image detection."""
    # Create widgets
    image_path_widget = widgets.Text(
        value='path/to/your/image.jpg',
        placeholder='Enter image path',
        description='Image Path:',
        layout=Layout(width='80%')
    )
    
    conf_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=0.9,
        step=0.05,
        description='Confidence:',
        continuous_update=False
    )
    
    detect_button = widgets.Button(description='Detect Objects')
    output = widgets.Output()
    
    def on_detect_clicked(b):
        with output:
            output.clear_output()
            image_path = image_path_widget.value
            if os.path.exists(image_path):
                print(f"Processing {image_path}...")
                results = detect_image(image_path, conf_slider.value)
                if results is not None:
                    print("Detection complete!")
            else:
                print(f"Error: File not found - {image_path}")
    
    detect_button.on_click(on_detect_clicked)
    
    # Display widgets
    display(widgets.VBox([
        widgets.HTML("<h3>Snooker Ball Detection Demo</h3>"),
        image_path_widget,
        conf_slider,
        detect_button,
        output
    ]))

# 8. Example Usage
if __name__ == "__main__":
    # Example 1: Detect objects in an image
    # detect_image('path/to/your/image.jpg', conf_threshold=0.5)
    
    # Example 2: Process a video
    # detect_video('path/to/your/video.mp4', 'output.mp4', conf_threshold=0.5, max_frames=100)
    
    # Example 3: Run interactive demo (in Jupyter Notebook)
    # interactive_demo()
    print("Ready to run detection! Uncomment the example code above to get started.")

# To convert this script to a Jupyter Notebook:
# 1. Install jupytext: pip install jupytext
# 2. Convert the script: jupytext --to notebook snooker_detection_demo.py
# 3. Open the resulting .ipynb file in Jupyter Notebook/Lab
