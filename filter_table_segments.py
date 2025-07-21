import cv2
import numpy as np
import os
from tqdm import tqdm

def is_full_table_visible(frame):
    """
    Check if the frame contains a full view of the snooker table.
    Returns True if full table is visible, False otherwise.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define green color range for snooker table (adjust these values if needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
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
            
        # Check if table is near the edges (indicating it might be cropped)
        margin = 0.05  # 5% margin from edges
        if (x < frame.shape[1] * margin or 
            y < frame.shape[0] * margin or
            x + w > frame.shape[1] * (1 - margin) or
            y + h > frame.shape[0] * (1 - margin)):
            return False
            
        return True
    return False

def process_video(input_path, output_path):
    """
    Process the input video and save only frames with full table view to output video.
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {input_path}")
    print(f"FPS: {fps}, Total frames: {frame_count}, Resolution: {frame_width}x{frame_height}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define codec and create VideoWriter object with high quality settings
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec for better quality
    
    # Use a high quality bitrate (5 Mbps for 640x360 resolution)
    # The original bitrate of 260 bps is too low for video
    target_bitrate = 5000000  # 5 Mbps for good quality at 640x360
    
    # Create VideoWriter with quality settings
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        fps, 
        (frame_width, frame_height),
        isColor=True
    )
    
    # Try to set the quality (not all codecs support this)
    try:
        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)  # 100% quality
    except:
        pass  # Some codecs don't support quality setting
    
    # Get the original codec information
    original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"Original codec: {chr(original_fourcc&0xff)}{chr((original_fourcc>>8)&0xff)}{chr((original_fourcc>>16)&0xff)}{chr((original_fourcc>>24)&0xff)}")
    print(f"Using high quality bitrate: {target_bitrate} bps")
    
    # Process frames
    saved_frames = 0
    pbar = tqdm(total=frame_count, desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if is_full_table_visible(frame):
            out.write(frame)
            saved_frames += 1
            
        pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    pbar.close()
    
    print(f"\nProcessing complete!")
    print(f"Saved {saved_frames} out of {frame_count} frames ({saved_frames/frame_count*100:.1f}%)")
    print(f"Output saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    # Input and output paths
    input_video = "/Users/abhinavrai/Playground/snooker_data/videos/ROS-Frame-1.mp4"
    output_video = "/Users/abhinavrai/Playground/snooker_data/processed/filtered_table_view.mp4"
    
    # Process the video
    process_video(input_video, output_video)
