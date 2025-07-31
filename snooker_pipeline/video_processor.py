"""
Video processing utilities for the snooker training pipeline.
"""
import os
import logging
import subprocess
import math
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def run_ffmpeg_command(cmd: list) -> bool:
    """Run an FFmpeg command and handle the output."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg command failed with return code {result.returncode}")
            logger.error(f"FFmpeg stderr: {result.stderr}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error running FFmpeg command: {e}")
        return False

def get_video_info(input_path: str) -> Dict[str, Any]:
    """Get video information using FFprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
        '-of', 'json',
        input_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFprobe failed: {result.stderr}")
            return {}
            
        import json
        info = json.loads(result.stdout)
        
        if 'streams' not in info or not info['streams']:
            return {}
            
        stream = info['streams'][0]
        
        # Parse frame rate (can be in fraction form like '30000/1001')
        if 'r_frame_rate' in stream:
            num, denom = map(float, stream['r_frame_rate'].split('/'))
            fps = num / denom if denom != 0 else 0
        else:
            fps = 0
            
        return {
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
            'fps': fps,
            'duration': float(stream.get('duration', 0)),
            'total_frames': int(stream.get('nb_frames', 0))
        }
        
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {}

def process_video(
    input_path: str,
    output_path: str,
    target_fps: Optional[float] = None,
    output_width: int = 640,
    output_height: int = 640,
    debug: bool = False
) -> bool:
    """Process a video using FFmpeg to extract frames at specified intervals.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save processed video
        target_fps: Target FPS for frame extraction (None to keep original)
        output_width: Width of output video (default: 640)
        output_height: Height of output video (default: 640)
        debug: If True, enable debug logging
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False
    
    # Get video information
    video_info = get_video_info(input_path)
    if not video_info:
        logger.error(f"Could not get video info for: {input_path}")
        return False
    
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    total_frames = video_info['total_frames']
    
    # Use target FPS if specified, otherwise use original
    output_fps = target_fps if target_fps is not None else fps
    
    # Calculate how many frames to skip to achieve target FPS
    frame_interval = max(1, round(fps / output_fps)) if output_fps < fps else 1
    
    total_frames_to_process = math.ceil(total_frames / frame_interval)
    
    logger.info(f"Processing video: {os.path.basename(input_path)}")
    logger.info(f"Input: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    logger.info(f"Output: {output_width}x{output_height} @ {output_fps:.2f}fps, max {total_frames_to_process} frames")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', input_path,
        '-vf', f'fps={output_fps},scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2',
        '-vframes', str(total_frames_to_process),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    if debug:
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
    
    # Run FFmpeg command
    success = run_ffmpeg_command(cmd)
    
    if success:
        logger.info(f"Successfully processed video: {os.path.basename(input_path)}")
        return True
    else:
        logger.error(f"Failed to process video: {os.path.basename(input_path)}")
        return False
