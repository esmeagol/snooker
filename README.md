# Snooker Ball Detection with YOLOv8

This project demonstrates how to train a YOLOv8 model to detect snooker balls and players in videos.

## Project Structure

```
snooker/
├── data/                   # Data directory
│   ├── raw_videos/         # Downloaded YouTube videos
│   ├── processed_videos/   # Preprocessed videos
│   ├── raw_frames/         # Extracted video frames
│   ├── annotated_frames/   # Labeled frames with annotations
│   └── yolo_dataset/       # YOLO-formatted dataset
│       ├── train/          # Training set
│       │   ├── images/     # Training images
│       │   └── labels/     # Training labels
│       ├── val/            # Validation set
│       │   ├── images/     # Validation images
│       │   └── labels/     # Validation labels
│       └── test/           # Test set
│           ├── images/     # Test images
│           └── labels/     # Test labels
├── snooker_pipeline/       # Training data generation pipeline
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── pipeline.py        # Main pipeline implementation
│   ├── video_processor.py # Video processing utilities
│   └── example_config.yaml # Example configuration
├── videos/                 # Input videos
├── extract_frames.py       # Script to extract frames from videos
├── split_dataset.py        # Script to split dataset
├── snooker_dataset.yaml    # YOLO dataset configuration
└── train.py               # Training script
```

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install ultralytics opencv-python-headless tqdm
   ```

## Usage

1. **Extract Frames from Videos**
   Place your snooker videos in the `videos` directory and run:
   ```bash
   python extract_frames.py
   ```
   This will extract frames from all videos in the `videos` directory and save them to `data/raw_frames`.

2. **Label the Frames**
   Use a tool like LabelImg to annotate the frames. Install it with:
   ```bash
   pip install labelImg
   labelImg
   ```
   - Set the output format to YOLO
   - Open the `data/raw_frames` directory
   - Save annotations to `data/annotated_frames`

3. **Split the Dataset**
   Run the following command to split the dataset into training, validation, and test sets:
   ```bash
   python split_dataset.py
   ```

4. **Train the Model**
   Start training with:
   ```bash
   python train.py
   ```
   The model will be saved in the `runs` directory.

## Training on Custom Data

1. Update the class names in `snooker_dataset.yaml`
2. Modify the training parameters in `train.py` as needed
3. Adjust the frame extraction interval in `extract_frames.py` if needed

## Model Evaluation

After training, you can evaluate the model using the test set:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/yolov8n_snooker/weights/best.pt')

# Evaluate on test set
metrics = model.val()
print(metrics)
```

## Inference on New Videos

To run inference on new videos:

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('runs/detect/yolov8n_snooker/weights/best.pt')

# Run inference on a video
results = model('path/to/your/video.mp4', save=True)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
