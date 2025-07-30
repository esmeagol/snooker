"""
Video processing utilities for the snooker training pipeline.
"""
import os
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def is_full_table_visible(frame: np.ndarray) -> bool:
    """Check if the frame contains a full view of the snooker table.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        bool: True if full table is visible, False otherwise
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define green color range for snooker table (adjust these values if needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Check if we have a quadrilateral (table)
    if len(approx) == 4:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate areas
        frame_area = frame.shape[0] * frame.shape[1]
        table_area = w * h
        
        # If table is less than 40% of frame, it's probably not a full table view
        if table_area / frame_area < 0.4:
            return False
            
        # Additional checks can be added here if needed
        return True
    
    return False

def process_video(input_path: str, output_path: str, target_fps: Optional[int] = None, max_frames: int = 120) -> bool:
    """Process a video to extract relevant frames with full table view.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save processed video
        target_fps: Target FPS for output video (None to keep original)
        max_frames: Maximum number of frames to extract from the video (default: 120)
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use target FPS if specified, otherwise use original
    output_fps = target_fps if target_fps is not None else fps
    
    logger.info(f"Processing video: {os.path.basename(input_path)}")
    logger.info(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
    logger.info(f"Output FPS: {output_fps:.2f}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        output_fps, 
        (width, height)
    )
    
    # Process each frame
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if we've reached the maximum number of frames
            if saved_count >= max_frames:
                logger.info(f"Reached maximum of {max_frames} frames. Stopping extraction.")
                break
                
            # Check if this frame shows a full table
            if is_full_table_visible(frame):
                out.write(frame)
                saved_count += 1
                
                # Log progress periodically
                if saved_count % 10 == 0:
                    logger.debug(f"Saved {saved_count}/{max_frames} frames from {frame_count}/{total_frames} processed")
                
            frame_count += 1
            
            # Log progress periodically for processed frames
            if frame_count % 100 == 0:
                logger.debug(f"Processed {frame_count}/{total_frames} frames, saved {saved_count}/{max_frames}")
                
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return False
    finally:
        # Release resources
        cap.release()
        out.release()
    
    logger.info(f"Processing complete. Saved {saved_count} frames out of {frame_count} "
               f"({saved_count/max(1, frame_count)*100:.1f}%)")
    
    return True
