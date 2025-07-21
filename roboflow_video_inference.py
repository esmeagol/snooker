#!/usr/bin/env python3
"""
Roboflow Video Inference Script
Deploy multiple Roboflow models on video files with real-time processing and annotation output.
"""

import argparse
import cv2
import os
import sys
import time
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

try:
    from inference import get_model
    import supervision as sv
except ImportError as e:
    print(f"Error: Missing required dependencies. Install with:")
    print("pip install inference supervision")
    sys.exit(1)

# RT-DETR local inference import
try:
    from rtdetr_inference import RTDETRInference
except ImportError:
    RTDETRInference = None

class RoboflowVideoProcessor:
    def __init__(self, model_names=None, confidence=0.5):
        """Initialize the video processor with models and settings."""
        self.model_names = model_names or [
            "snooker-vision-5iuwr-afgep/1",  # Pocket detection model
            "snookers-gkqap/1"  # Ball detection model
        ]
        self.confidence = confidence
        self.models = []
        self.annotators = []
        self.label_annotators = []
        self.pocket_positions = None
        self.detected_pockets = None  # Store detected pocket positions
        self.homography = None
        self.table_img = None
        self.ball_assets = {}
        self.pockets_detected = False  # Flag to track if pockets have been detected
        self.rtdetr_inferencer = None
        
        # Load table and ball assets
        self._load_assets()
        self._define_pocket_positions()

    def _load_assets(self):
        """Load table image and ball assets."""
        # Load table image
        table_img_path = '/Users/abhinavrai/Playground/snooker/game_info/snooker table markings.png'
        if os.path.exists(table_img_path):
            self.table_img = cv2.imread(table_img_path)
            if self.table_img is not None:
                print(f"Loaded table image: {table_img_path}")
        
        # Load ball assets
        ball_colors = ['red', 'yellow', 'green', 'brown', 'blue', 'pink', 'black']
        for color in ball_colors:
            ball_path = f'/Users/abhinavrai/Playground/snooker/game_info/Snooker_ball_{color}.png'
            if os.path.exists(ball_path):
                img = cv2.imread(ball_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Resize ball to a reasonable size (adjust as needed)
                    ball_size = 40
                    img = cv2.resize(img, (ball_size, ball_size))
                    self.ball_assets[color] = img
                    print(f"Loaded ball asset: {color}")

    def _define_pocket_positions(self):
        """Define the pocket positions in the table image.
        These are the target points for the perspective transform.
        Order: [bottom_right, bottom_middle, bottom_left, top_left, top_middle, top_right]
        """
        if self.table_img is None:
            return
            
        h, w = self.table_img.shape[:2]
        
        # Define pocket positions as percentage of image dimensions
        # These values may need adjustment based on your specific table image
        x_margin = 0.05 * w
        y_margin = 0.05 * h
        
        self.pocket_positions = np.array([
            [w - x_margin, h - y_margin],  # Bottom right
            [w // 2, h - y_margin],        # Bottom middle
            [x_margin, h - y_margin],      # Bottom left
            [x_margin, y_margin],          # Top left
            [w // 2, y_margin],            # Top middle
            [w - x_margin, y_margin]       # Top right
        ], dtype=np.float32)
        
        # Draw the pocket positions on the table image for debugging
        self.table_img = self.table_img.copy()
        for i, (px, py) in enumerate(self.pocket_positions):
            cv2.circle(self.table_img, (int(px), int(py)), 10, (0, 0, 255), 2)
            cv2.putText(self.table_img, f'P{i+1}', (int(px) - 15, int(py) - 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def _update_homography(self, pocket_centers):
        """Update the homography matrix based on detected pocket centers.
        
        Args:
            pocket_centers: List of detected pocket centers
            
        Returns:
            bool: True if homography was updated successfully, False otherwise
        """
        if len(pocket_centers) != 6 or self.pocket_positions is None:
            return False
            
        # Sort pocket centers in the same order as pocket_positions
        # Sort by y-coordinate first, then x-coordinate for consistent ordering
        pocket_centers = np.array(sorted(pocket_centers, key=lambda x: (x[1], x[0])))
        
        # Store the detected pockets for future use
        self.detected_pockets = pocket_centers
        self.pockets_detected = True
        
        # Calculate homography using RANSAC for robustness to outliers
        self.homography, _ = cv2.findHomography(pocket_centers, self.pocket_positions, cv2.RANSAC, 5.0)
        return self.homography is not None

    def _draw_table_overlay(self, frame, all_detections=None):
        """Draw the table overlay with balls and pockets.
        
        Args:
            frame: The video frame (unused, kept for compatibility)
            all_detections: List of detections from all models
            
        Returns:
            The table overlay with balls and pockets drawn on it
        """
        # Create a copy of the table image
        overlay = self.table_img.copy()
        
        # Draw pockets on the overlay if detected
        if hasattr(self, 'detected_pockets') and self.detected_pockets is not None:
            for i, (px, py) in enumerate(self.detected_pockets):
                # Draw a circle for each pocket
                cv2.circle(overlay, (int(px), int(py)), 10, (0, 0, 255), 2)
                # Add pocket number
                cv2.putText(overlay, f'P{i+1}', (int(px)-15, int(py)-15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if all_detections is not None and self.homography is not None:
            for detections in all_detections:
                if detections is not None and hasattr(detections, 'xyxy') and hasattr(detections, 'class_name'):
                    for i in range(len(detections.xyxy)):
                        # Skip if not a ball detection
                        if i >= len(detections.class_name) or detections.class_name[i] not in self.ball_assets:
                            continue
                            
                        # Get ball center
                        x1, y1, x2, y2 = detections.xyxy[i]
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        
                        # Transform to table coordinates
                        point = np.array([[[x_center, y_center]]], dtype=np.float32)
                        transformed_point = cv2.perspectiveTransform(point, self.homography)
                        tx, ty = transformed_point[0][0]
                        
                        # Get ball color
                        color = detections.class_name[i]
                        
                        # Get ball asset if available
                        if color in self.ball_assets:
                            ball_img = self.ball_assets[color]
                            ball_h, ball_w = ball_img.shape[:2]
                            
                            # Calculate position to center the ball
                            x = int(tx - ball_w // 2)
                            y = int(ty - ball_h // 2)
                            
                            # Draw ball with transparency
                            if ball_img.shape[2] == 4:  # Has alpha channel
                                alpha_s = ball_img[:, :, 3] / 255.0
                                alpha_l = 1.0 - alpha_s
                                
                                for c in range(0, 3):
                                    try:
                                        overlay[y:y+ball_h, x:x+ball_w, c] = (
                                            alpha_s * ball_img[:, :, c] + 
                                            alpha_l * overlay[y:y+ball_h, x:x+ball_w, c]
                                        )
                                    except:
                                        # Handle out of bounds errors
                                        pass
                            else:
                                try:
                                    overlay[y:y+ball_h, x:x+ball_w] = ball_img
                                except:
                                    pass
        
        return overlay

    def load_models(self):
        """Load multiple Roboflow and/or local RT-DETR models."""
        try:
            # Define colors for different models
            colors = [
                sv.Color.from_hex("#FF0000"),  # Red
                sv.Color.from_hex("#00FF00"),  # Green
                sv.Color.from_hex("#0000FF"),  # Blue
                sv.Color.from_hex("#FFFF00"),  # Yellow
                sv.Color.from_hex("#FF00FF"),  # Magenta
            ]
            
            for i, model_name in enumerate(self.model_names):
                if model_name == 'rtdetr-local':
                    if RTDETRInference is None:
                        raise ImportError("RTDETRInference could not be imported. Make sure rtdetr_inference.py exists.")
                    self.rtdetr_inferencer = RTDETRInference()
                    self.models.append('rtdetr-local')
                    print(f"✓ Local RT-DETR loaded from /Users/abhinavrai/Downloads/rt-detr")
                else:
                    # Load model using inference library
                    model = get_model(model_id=model_name)
                    self.models.append(model)
                    print(f"✓ Model loaded: {model_name}")
                # Create annotators for each model
                box_annotator = sv.BoxAnnotator(
                    color=colors[i % len(colors)],
                    thickness=2
                )
                label_annotator = sv.LabelAnnotator(
                    color=colors[i % len(colors)],
                    text_padding=2,   # Less padding
                    text_position=sv.Position.TOP_LEFT  # Consistent label position
                )
                self.annotators.append(box_annotator)
                self.label_annotators.append(label_annotator)
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def process_video(self, video_path, output_path=None, show_preview=True, save_results=True):
        """Process a video file with the loaded models."""
        if not self.models:
            print("Error: No models loaded")
            return
            
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {frame_width}x{frame_height} @ {fps:.2f}fps, {frame_count} frames")
        
        # Initialize video writer if saving results
        if save_results:
            if output_path is None:
                base_name = os.path.splitext(video_path)[0]
                output_path = f"{base_name}_annotated.mp4"
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize FPS counter
        fps_counter = 0
        fps_start_time = time.time()
        frame_num = 0
        start_time = time.time()
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference on all models
            all_detections = []
            for model, model_name in zip(self.models, self.model_names):
                # Skip pocket detection model if we've already detected pockets
                if 'vision' in model_name.lower() and self.pockets_detected:
                    all_detections.append(None)
                    continue
                
                if model_name == 'rtdetr-local':
                    results = self.rtdetr_inferencer.infer(frame, threshold=self.confidence)
                    # Convert RT-DETR output to sv.Detections
                    if results is not None and 'boxes' in results:
                        xyxy = []
                        confidence = []
                        class_id = []
                        for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
                            x1, y1, x2, y2 = box
                            xyxy.append([x1, y1, x2, y2])
                            confidence.append(score)
                            class_id.append(label)
                        detections = sv.Detections(
                            xyxy=np.array(xyxy),
                            confidence=np.array(confidence),
                            class_id=np.array(class_id)
                        )
                    else:
                        detections = sv.Detections.empty()
                else:
                    results = model.infer(frame, confidence=self.confidence)[0]
                    detections = sv.Detections.from_inference(results)
                all_detections.append(detections)
                
                # Process pocket detections if this is the pocket model and we haven't detected pockets yet
                if 'vision' in model_name.lower() and not self.pockets_detected and len(detections) > 0:
                    pocket_centers = []
                    for bbox, confidence in zip(detections.xyxy, detections.confidence):
                        if confidence >= self.confidence:
                            x1, y1, x2, y2 = bbox
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            pocket_centers.append([cx, cy])
                    
                    if len(pocket_centers) == 6:
                        if self._update_homography(np.array(pocket_centers, dtype=np.float32)):
                            print("✅ Successfully detected all 6 pockets and calculated homography")
                            print("ℹ️ Pocket detection will be disabled for remaining frames")
            
            # Annotate the frame
            annotated_frame = frame.copy()
            for detections, box_annotator, label_annotator, model_name in zip(
                all_detections, self.annotators, self.label_annotators, self.model_names
            ):
                if detections is None:
                    continue  # Skip if we're not running this model anymore
                    
                if len(detections) > 0:
                    # Annotate frame with bounding boxes
                    annotated_frame = box_annotator.annotate(
                        scene=annotated_frame.copy(),
                        detections=detections
                    )
                    
                    # Add labels with confidence scores
                    labels = [
                        f"{model_name.split('/')[-2]}: {score:.2f}"
                        for score in detections.confidence
                    ]
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                    )
            
            # Draw table overlay if homography is available
            if self.homography is not None:
                # Get the table overlay with ball positions
                table_overlay = self._draw_table_overlay(frame, all_detections)
                
                # Resize both frames to be half the width of the output
                h, w = frame.shape[:2]
                new_width = w // 2
                new_height = int(h * (new_width / w))
                
                # Resize the original frame
                resized_frame = cv2.resize(annotated_frame, (new_width, new_height))
                
                # Draw pocket annotations on the video frame
                if self.detected_pockets is not None and self.homography is not None:
                    # Transform pocket positions back to video frame coordinates
                    inv_homography = np.linalg.inv(self.homography)
                    for i, (px, py) in enumerate(self.detected_pockets):
                        # Transform pocket position to video frame coordinates
                        point = np.array([[[px, py]]], dtype=np.float32)
                        transformed_point = cv2.perspectiveTransform(point, inv_homography)
                        x, y = transformed_point[0][0]
                        
                        # Draw a circle at the pocket position
                        cv2.circle(resized_frame, (int(x * new_width / w), int(y * new_height / h)), 
                                 8, (0, 0, 255), 2)
                        # Add pocket number
                        cv2.putText(resized_frame, f'P{i+1}', 
                                  (int(x * new_width / w) - 20, int(y * new_height / h) - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Resize the table overlay to match the height of the video
                table_overlay_resized = cv2.resize(table_overlay, (new_width, new_height))
                
                # Create a new canvas that's wide enough for both frames side by side
                combined = np.zeros((new_height, w, 3), dtype=np.uint8)
                
                # Place the original frame on the left
                combined[:, :new_width] = resized_frame
                
                # Place the table overlay on the right
                if len(table_overlay_resized.shape) == 3 and table_overlay_resized.shape[2] == 4:  # RGBA
                    # Extract the alpha channel and create a mask
                    alpha = table_overlay_resized[:, :, 3] / 255.0
                    alpha = cv2.merge([alpha, alpha, alpha])
                    
                    # Blend the overlay with the black background
                    overlay_rgb = table_overlay_resized[:, :, :3]
                    combined[:, new_width:] = (1 - alpha) * combined[:, new_width:].astype(float) + \
                                            alpha * overlay_rgb.astype(float)
                    combined = combined.astype(np.uint8)
                else:  # RGB or single channel
                    combined[:, new_width:] = table_overlay_resized
                
                # Update the annotated frame to be the combined view
                annotated_frame = combined
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:  # Update FPS every second
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                
                # Print progress
                progress = (frame_num / frame_count) * 100
                status = "(detecting pockets...)" if not self.pockets_detected else "(tracking balls)"
                print(f"Progress: {progress:.1f}% ({frame_num}/{frame_count}) - {current_fps:.1f} fps {status}")
            
            # Display the annotated frame
            if show_preview:
                cv2.imshow('Snooker Analysis', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user.")
                    break
                    
            # Save the annotated frame
            if save_results:
                out.write(annotated_frame)
            
            frame_num += 1
            
            # Early exit if we've processed enough frames for testing
            # if frame_num > 100:
            #     print("\nEarly exit for testing")
            #     break
        
        # Release resources
        cap.release()
        if save_results:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
            
        print("\nProcessing complete!")
        print(f"Processed {frame_num} frames in {time.time() - start_time:.1f}s")
        if frame_num > 0:
            print(f"Average FPS: {frame_num / (time.time() - start_time):.1f}")
        if save_results:
            print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Deploy multiple Roboflow models on video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python roboflow_video.py video.mp4 --models snooker-vision-5iuwr-afgep/1 snookers-gkqap/1
  python roboflow_video.py video.mp4 --models snooker-vision-5iuwr-afgep/1 snookers-gkqap/1 --preview
  python roboflow_video.py video.mp4 --models snooker-vision-5iuwr-afgep/1 snookers-gkqap/1 --save-results
        """
    )
    
    parser.add_argument(
        'video_path',
        help='Path to the input video file'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['snooker-vision-5iuwr-afgep/1', 'snookers-gkqap/1'],
        help='Model IDs to run (default: both snooker models)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output video file path (default: input_annotated.mp4)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show real-time preview window'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detection results to JSON file'
    )
    
    args = parser.parse_args()
    
    print(f"Running {len(args.models)} models:")
    for i, model_name in enumerate(args.models):
        print(f"  {i+1}. {model_name}")
    
    # Initialize processor
    processor = RoboflowVideoProcessor(
        model_names=args.models,
        confidence=args.confidence
    )
    
    # Load models
    processor.load_models()
    
    # Process video
    success = processor.process_video(
        video_path=args.video_path,
        output_path=args.output,
        show_preview=args.preview,
        save_results=args.save_results
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()