"""
Main pipeline module for the snooker training data generation.
"""
import os
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import logging
from tqdm import tqdm
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SnookerTrainingPipeline:
    """Main pipeline class for generating snooker training data."""
    
    def __init__(self, config: Dict):
        """Initialize the pipeline with configuration.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.config['raw_videos_dir'], exist_ok=True)
        os.makedirs(self.config['processed_videos_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Create train/val/test directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.config['output_dir'], split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.config['output_dir'], split, 'labels'), exist_ok=True)
    
    def download_videos(self, urls_file: str):
        """Download videos from YouTube using yt-dlp.
        
        Args:
            urls_file: Path to file containing YouTube URLs (one per line)
        """
        logger.info(f"Starting video download from {urls_file}")
        
        # Check if yt-dlp is installed
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("yt-dlp is not installed. Please install it with: pip install yt-dlp")
            raise
        
        # Read all URLs from the file
        try:
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading URLs file: {e}")
            raise
        
        # Create a temporary file to store successful downloads
        temp_file = f"{urls_file}.tmp"
        
        # Base command without the input file
        base_cmd = [
            'yt-dlp',
            '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
            '--merge-output-format', 'mp4',
            '--output', os.path.join(self.config['raw_videos_dir'], '%(id)s.%(ext)s'),
            '--download-archive', os.path.join(self.config['raw_videos_dir'], 'downloaded.txt'),
            '--no-warnings',
            '--no-playlist'
        ]
        
        # Add cookies if specified in config
        if 'cookies_file' in self.config and self.config['cookies_file']:
            base_cmd.extend(['--cookies', self.config['cookies_file']])
        
        # Process each URL individually
        success_count = 0
        failed_urls = []
        
        for url in urls:
            try:
                # Create a temporary file with just this URL
                with open(temp_file, 'w') as f:
                    f.write(f"{url}\n")
                
                # Build the command for this URL
                cmd = base_cmd.copy()
                cmd.extend(['-a', temp_file])
                
                # Run the command
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    success_count += 1
                    logger.debug(f"Successfully processed: {url}")
                else:
                    # If download fails, log the error and add to failed URLs
                    error_msg = result.stderr or "Unknown error"
                    logger.warning(f"Failed to download {url}: {error_msg}")
                    failed_urls.append(url)
            except Exception as e:
                logger.warning(f"Error processing {url}: {str(e)}")
                failed_urls.append(url)
        
        # Clean up temporary file
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file: {e}")
        
        # If there were any failures, update the URLs file
        if failed_urls:
            # Get the successful URLs (original list minus failed ones)
            successful_urls = [url for url in urls if url not in failed_urls]
            
            # Write the successful URLs back to the file
            try:
                with open(urls_file, 'w') as f:
                    f.write('\n'.join(successful_urls) + '\n')
                logger.warning(f"Removed {len(failed_urls)} failed URLs from {urls_file}")
            except Exception as e:
                logger.error(f"Error updating URLs file: {e}")
        
        # Log summary
        total = len(urls)
        failed = len(failed_urls)
        logger.info(f"Download completed: {success_count} succeeded, {failed} failed out of {total} URLs")
    
    def preprocess_videos(self):
        """Preprocess downloaded videos to extract relevant frames."""
        logger.info("Starting video preprocessing")
        
        # Get list of raw video files
        video_files = [f for f in os.listdir(self.config['raw_videos_dir']) 
                      if f.endswith(('.mp4', '.mkv', '.webm'))]
        
        if not video_files:
            logger.warning("No video files found in the raw videos directory")
            return
        
        # Process each video
        for video_file in tqdm(video_files, desc="Processing videos"):
            input_path = os.path.join(self.config['raw_videos_dir'], video_file)
            output_path = os.path.join(
                self.config['processed_videos_dir'], 
                f"processed_{os.path.splitext(video_file)[0]}.mp4"
            )
            
            # Skip if output already exists
            if os.path.exists(output_path):
                logger.debug(f"Skipping {video_file} - already processed")
                continue
                
            # Call the existing process_video function
            self._process_video(input_path, output_path)
    
    def _process_video(self, input_path: str, output_path: str):
        """Process a single video file to extract relevant frames using FFmpeg.
        
        This is a wrapper around the process_video function.
        """
        from .video_processor import process_video
        
        # Get video processing settings with defaults
        video_config = self.config.get('video_processing', {})
        target_fps = video_config.get('target_fps')
        debug = self.config.get('debug', False)
        
        # Get output resolution with defaults
        output_res = video_config.get('output_resolution', {})
        output_width = output_res.get('width', 640)
        output_height = output_res.get('height', 640)
        
        process_video(
            input_path=input_path, 
            output_path=output_path,
            target_fps=target_fps,
            output_width=output_width,
            output_height=output_height,
            debug=debug
        )
    
    def _get_split_ratios(self):
        """Get the train/val/test split ratios from config."""
        return (
            self.config.get('train_ratio', 0.7),
            self.config.get('val_ratio', 0.15),
            self.config.get('test_ratio', 0.15)
        )
        
    def _create_split_directories(self):
        """Create directories for train/val/test splits."""
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.config['output_dir'], split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.config['output_dir'], split, 'labels'), exist_ok=True)
    
    def extract_frames(self):
        """Extract frames from all videos into a temporary directory."""
        logger.info("Extracting frames from all videos")
        
        # Get list of processed video files
        video_files = [f for f in os.listdir(self.config['processed_videos_dir']) 
                      if f.endswith('.mp4')]
        
        if not video_files:
            logger.warning("No processed video files found")
            return
            
        # Create a temporary directory for all extracted frames
        temp_dir = os.path.join(self.config['output_dir'], '_temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get frame interval from config, default to 10 if not specified
        frame_interval = self.config.get('frame_interval', 10)
        logger.info(f"Using frame interval: {frame_interval}")
        
        max_frames = self.config.get('max_frames_per_video', 120)
        logger.info(f"Using max frames to extract per video: {max_frames}")
        
        # Extract frames from all videos
        for video_file in tqdm(video_files, desc="Extracting frames"):
            video_path = os.path.join(self.config['processed_videos_dir'], video_file)
            self._extract_frames_from_video(video_path, temp_dir, frame_interval, max_frames)
            
        # Store the temporary directory path for later use
        self.temp_frames_dir = temp_dir
        
    def split_dataset(self):
        """Split extracted frames into train/val/test sets."""
        if not hasattr(self, 'temp_frames_dir') or not os.path.exists(self.temp_frames_dir):
            logger.error("No frames found. Run extract_frames() first.")
            return
            
        logger.info("Splitting frames into train/val/test sets")
        
        # Get all frame files
        frame_files = [f for f in os.listdir(self.temp_frames_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not frame_files:
            logger.warning("No frame files found to split")
            return
            
        # Shuffle the frames
        random.shuffle(frame_files)
        
        # Get split ratios
        train_ratio, val_ratio, test_ratio = self._get_split_ratios()
        total = len(frame_files)
        
        # Calculate split indices
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        
        # Split the frames
        splits = {
            'train': frame_files[:train_end],
            'val': frame_files[train_end:val_end],
            'test': frame_files[val_end:]
        }
        
        # Create output directories
        self._create_split_directories()
        
        # Move frames to their respective split directories
        for split_name, files in splits.items():
            logger.info(f"Moving {len(files)} frames to {split_name} set")
            
            for frame_file in tqdm(files, desc=f"Moving {split_name} frames"):
                src = os.path.join(self.temp_frames_dir, frame_file)
                dst = os.path.join(self.config['output_dir'], split_name, 'images', frame_file)
                
                # Move the file
                shutil.move(src, dst)
        
        # Clean up the temporary directory
        shutil.rmtree(self.temp_frames_dir)
        logger.info("Temporary frame directory cleaned up")
    
    # Removed the old extract_frames method as it's been replaced with a new implementation
    
    def _extract_frames_from_video(self, video_path: str, output_dir: str, frame_interval: int = 10, max_frames: int = 120):
        """Extract frames from a video file.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save extracted frames
            frame_interval: Extract one frame every N frames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame if it's time to do so
            if frame_count % frame_interval == 0:
                frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
            frame_count += 1
            
            # Show progress
            if frame_count % 100 == 0:
                logger.debug(f"Processed {frame_count}/{total_video_frames} frames, saved {saved_count}")
        
        # Release resources
        cap.release()
        logger.debug(f"Extracted {saved_count} frames from {video_path}")

def run_pipeline(config: Dict, steps: List[str] = None):
    """Run the entire pipeline or specified steps.
    
    Args:
        config: Configuration dictionary
        steps: List of steps to run. If None, run all steps.
               Possible values: 'download', 'preprocess', 'extract', 'split'
    """
    # Default to all steps if none specified
    if steps is None:
        steps = ['download', 'preprocess', 'extract', 'split']
    
    # Initialize the pipeline
    pipeline = SnookerTrainingPipeline(config)
    
    # Run the specified steps
    if 'download' in steps and 'youtube_urls' in config:
        logger.info("=== Running download step ===")
        pipeline.download_videos(config['youtube_urls'])
    
    if 'preprocess' in steps:
        logger.info("=== Running preprocessing step ===")
        pipeline.preprocess_videos()
    
    if 'extract' in steps:
        logger.info("=== Running frame extraction step ===")
        pipeline.extract_frames()
    
    if 'split' in steps:
        logger.info("=== Running dataset split step ===")
        pipeline.split_dataset()
    
    logger.info("Pipeline execution completed")
