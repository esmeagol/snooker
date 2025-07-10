"""
Extract sample frames from a snooker video for testing.
"""

import os
import cv2
import numpy as np

def extract_sample_frames(video_path: str, output_dir: str, num_frames: int = 5) -> None:
    """
    Extract evenly spaced frames from a video file.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default: 5)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Extracting {num_frames} sample frames...")
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Extract and save frames
    for i, frame_idx in enumerate(frame_indices):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Generate output filename
            frame_time = frame_idx / fps
            filename = f"frame_{i+1:02d}_at_{frame_time:.1f}s.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Save the frame
            cv2.imwrite(output_path, frame)
            print(f"Saved: {filename} (frame {frame_idx})")
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    
    # Release the video capture object
    cap.release()
    print(f"\nExtracted {len(frame_indices)} frames to: {output_dir}")

if __name__ == "__main__":
    # Input video path
    video_path = "/Users/abhinavrai/Playground/snooker_data/videos/6-reds Snooker Shootout： Century Player Takes On League Teammate [KqDvy8UmmXo].mp4"
    
    # Output directory for frames
    output_dir = "test_data"
    
    # Extract frames
    extract_sample_frames(video_path, output_dir, num_frames=5)
