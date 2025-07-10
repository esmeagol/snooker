"""
Test script for ball detection with table corner detection.

This script allows testing the ball detection pipeline with the following steps:
1. Load an image of a snooker table
2. Detect or manually mark table corners
3. Calculate expected ball size based on table dimensions
4. Run ball detection and save debug images
"""

import os
import cv2
import numpy as np
import argparse
import time
from typing import List, Tuple, Optional, Dict, Any
from table_detection_2 import detect_table_corners, visualize_corners
from sobel_ball_detector import SobelBallDetector

def setup_debug() -> str:
    """Create debug directory if it doesn't exist and return its path."""
    debug_dir = 'debug'
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir

def save_debug_image(image: np.ndarray, name: str) -> None:
    """Save an image to the debug directory with a timestamp."""
    debug_dir = setup_debug()
    timestamp = int(time.time() * 1000)  # Use milliseconds for better uniqueness
    cv2.imwrite(f'{debug_dir}/{name}_{timestamp}.jpg', image)
    print(f"Saved debug image: {debug_dir}/{name}_{timestamp}.jpg")

def get_manual_corners(image: np.ndarray) -> np.ndarray:
    """Allow user to manually select 4 corners of the table."""
    points = []
    clone = image.copy()
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(clone, points[-1], points[0], (0, 255, 0), 2)
                cv2.imshow('Select Table Corners (4 points)', clone)
    
    print("\n=== Manual Corner Selection ===")
    print("Click on the 4 corners of the table in this order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("Then press any key to continue...")
    
    cv2.namedWindow('Select Table Corners (4 points)')
    cv2.setMouseCallback('Select Table Corners (4 points)', click_event)
    
    while True:
        cv2.imshow('Select Table Corners (4 points)', clone)
        key = cv2.waitKey(1) & 0xFF
        if key != 255 or len(points) >= 4:  # Any key or 4 points selected
            break
    
    cv2.destroyAllWindows()
    
    if len(points) != 4:
        print("Error: Exactly 4 points must be selected.")
        return None
    
    return np.array(points, dtype=np.float32)

def calculate_ball_size(corners: np.ndarray) -> float:
    """
    Calculate expected ball size in pixels based on table dimensions.
    
    Standard table dimensions:
    - Bottom edge: 1780 mm
    - Ball diameter: 52.4 mm
    """
    # Calculate bottom edge length in pixels
    bottom_left = corners[3]  # Bottom-left corner
    bottom_right = corners[2]  # Bottom-right corner
    bottom_length_px = np.linalg.norm(bottom_right - bottom_left)
    
    # Calculate ball size in pixels using proportion
    ball_size_mm = 52.4  # Standard snooker ball diameter in mm
    table_edge_mm = 1820  # Bottom edge length in mm
    
    ball_size_px = (ball_size_mm / table_edge_mm) * bottom_length_px
    print(f"Calculated ball size: {ball_size_px:.2f} pixels")
    
    return ball_size_px

def create_baize_mask(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Create a binary mask for the baize (playing area) of the table."""
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
    return mask

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test ball detection on a snooker table image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--auto', action='store_true', help='Use automatic table detection')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    args = parser.parse_args()
    
    # Load image
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Failed to load image: {args.image_path}")
        return
    
    # Detect or manually select table corners
    if args.auto:
        print("Using automatic table detection...")
        corners = detect_table_corners(image, debug=args.debug)
        if corners is None:
            print("Automatic table detection failed. Falling back to manual selection.")
            corners = get_manual_corners(image)
    else:
        print("Please manually select table corners...")
        corners = get_manual_corners(image)
    
    if corners is None:
        print("Failed to get table corners. Exiting.")
        return
    
    # Visualize and save the detected corners
    corner_vis = visualize_corners(image.copy(), corners)
    cv2.imshow('Detected Table Corners', corner_vis)
    if args.debug:
        save_debug_image(corner_vis, 'table_corners')
    
    # Calculate ball size and initialize detector
    ball_diameter = calculate_ball_size(corners)
    detector = SobelBallDetector(ball_size=int(ball_diameter))
    
    # Create baize mask
    baize_mask = create_baize_mask(image, corners)
    
    # Detect balls
    result, balls = detector.process_frame(image, baize_mask)
    
    # Save and show results
    if args.debug:
        save_debug_image(baize_mask, 'baize_mask')
        save_debug_image(result, 'detected_balls')
    
    print(f"\nDetected {len(balls)} balls")

if __name__ == "__main__":
    main()
