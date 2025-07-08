import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=30):
    """Extract frames from video file.
    
    Args:
        video_path (str): Path to video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract one frame every N frames
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
    
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames...")
    
    frame_count = 0
    saved_count = 0
    progress_bar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's the right interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
        progress_bar.update(1)
        
    cap.release()
    progress_bar.close()
    print(f"Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    videos_dir = "videos"
    output_dir = "data/raw_frames"
    
    # Process all video files in the videos directory
    for video_file in os.listdir(videos_dir):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(videos_dir, video_file)
            extract_frames(video_path, output_dir, frame_interval=30)  # Extract 1 frame per second (assuming 30fps)
