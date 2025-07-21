import os
import cv2
import numpy as np
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
    Maintains original video quality and codec settings.
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False, 0, 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get original codec information
    original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    original_fourcc_str = ''.join([chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"\nProcessing video: {os.path.basename(input_path)}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}, Frames: {frame_count}")
    print(f"Original codec: {original_fourcc_str}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define codec and create VideoWriter object with original codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to mp4v if original codec can't be used
    try:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    except:
        pass
    
    # Create VideoWriter with original settings
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        fps, 
        (frame_width, frame_height),
        isColor=True
    )
    
    # Process frames
    saved_frames = 0
    pbar = tqdm(total=frame_count, desc="Processing frames", unit="frame")
    
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
    
    # Calculate statistics
    saved_percent = (saved_frames / frame_count * 100) if frame_count > 0 else 0
    
    print(f"Processing complete!")
    print(f"Saved {saved_frames} out of {frame_count} frames ({saved_percent:.1f}%)")
    print(f"Output saved to: {output_path}")
    
    return True, saved_frames, frame_count

def main():
    # Input and output directories
    input_dir = "/Users/abhinavrai/Playground/snooker_data/step_0_yt_videos"
    output_dir = "/Users/abhinavrai/Playground/snooker_data/step_1_yt_videos_full_table_segments"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files in input directory
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) 
                  and f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video
    total_saved = 0
    total_frames = 0
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f"filtered_{video_file}")
        
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"\nSkipping {video_file} - output already exists")
            continue
            
        success, saved, total = process_video(input_path, output_path)
        if success:
            total_saved += saved
            total_frames += total
    
    # Print summary
    if total_frames > 0:
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total videos processed: {len(video_files)}")
        print(f"Total frames processed: {total_frames}")
        print(f"Total frames saved: {total_saved} ({total_saved/max(1, total_frames)*100:.1f}%)")
        print("Processing complete!")

if __name__ == "__main__":
    main()
