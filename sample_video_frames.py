import os
import cv2
import numpy as np
import random

# Input and output directories
INPUT_DIR = "/Users/abhinavrai/Playground/snooker_data/step_1_yt_videos_full_table_segments"
OUTPUT_DIR = "/Users/abhinavrai/Playground/snooker_data/step_2_training_data"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sample_count(frame_count):
    """
    Choose number of samples based on video length.
    10 for very short (< 5k frames), 25 for very long (> 40k frames),
    interpolate in between.
    """
    if frame_count <= 5000:
        return 10
    elif frame_count >= 40000:
        return 25
    else:
        # Linear interpolation
        return int(10 + (frame_count - 5000) * (15 / 35000))

def sample_frames_from_video(video_path, global_state):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"No frames in {video_path}")
        cap.release()
        return
    n_samples = get_sample_count(frame_count)
    sample_indices = sorted(random.sample(range(frame_count), min(n_samples, frame_count)))
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx} from {video_path}")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{global_state['img_idx']:03d}.jpg")
        cv2.imwrite(out_path, frame)
        global_state['img_idx'] += 1
    cap.release()
    print(f"Sampled {len(sample_indices)} frames from {video_path}")

def main():
    global_state = {'img_idx': 0}
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith('.mp4'):
            continue
        video_path = os.path.join(INPUT_DIR, fname)
        sample_frames_from_video(video_path, global_state)

if __name__ == "__main__":
    main()
