"""
Improved table detection using adaptive thresholding and contour analysis.
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time

def setup_debug() -> None:
    """Create debug directory if it doesn't exist."""
    os.makedirs('debug', exist_ok=True)

def save_debug_image(image: np.ndarray, name: str, timestamp: int = None) -> None:
    """Save an image to the debug directory with a timestamp."""
    setup_debug()
    if timestamp is None:
        timestamp = int(time.time())
    cv2.imwrite(f'debug/{name}_{timestamp}.jpg', image)
    print(f"Debug: Saved {name} to debug/{name}_{timestamp}.jpg")

def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def preprocess_image(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, int]:
    """Preprocess the image for table detection using color-based segmentation."""
    timestamp = int(time.time()) if debug else None
    
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if debug:
        save_debug_image(hsv, '01_hsv', timestamp)
    
    # Define range for green color (snooker table)
    # These values may need adjustment based on your specific table color
    lower_green1 = np.array([35, 50, 50])
    upper_green1 = np.array([85, 255, 255])
    
    # Create a mask for green color
    mask_green = cv2.inRange(hsv, lower_green1, upper_green1)
    if debug:
        save_debug_image(mask_green, '02_mask_green', timestamp)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find the largest contour in the mask
    contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_green), timestamp
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask from the largest contour
    table_mask = np.zeros_like(mask_green)
    cv2.drawContours(table_mask, [largest_contour], -1, 255, -1)
    
    # Apply the mask to the original image
    masked = cv2.bitwise_and(image, image, mask=table_mask)
    if debug:
        save_debug_image(masked, '03_masked', timestamp)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    if debug:
        save_debug_image(clahe_img, '04_clahe', timestamp)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    if debug:
        save_debug_image(edges, '05_edges', timestamp)
    
    # Dilate edges to connect nearby edges
    dilated = cv2.dilate(edges, kernel, iterations=2)
    if debug:
        save_debug_image(dilated, '06_dilated', timestamp)
    
    return dilated, timestamp

def find_table_contour(processed: np.ndarray, original: np.ndarray, timestamp: int = None) -> Optional[np.ndarray]:
    """Find the largest rectangular contour in the processed image."""
    # Create a copy of the original for visualization
    debug_img = original.copy() if timestamp is not None else None
    
    # Find contours in the processed image
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if timestamp is not None:
            save_debug_image(debug_img, '07_no_contours', timestamp)
        return None
    
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find the best rectangle
    best_rect = None
    best_score = -1
    
    for i, contour in enumerate(contours):
        # Calculate contour area and skip small contours
        area = cv2.contourArea(contour)
        if area < 1000:  # Minimum area threshold
            continue
        
        # Simplify the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # We're looking for a quadrilateral (4 points)
        if len(approx) == 4:
            # Order the points consistently
            rect = order_points(approx.reshape(4, 2))
            
            # Calculate the bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Skip very small or very large rectangles
            min_dim = min(original.shape[:2])
            if w < min_dim * 0.2 or h < min_dim * 0.2 or \
               w > original.shape[1] * 0.9 or h > original.shape[0] * 0.9:
                continue
            
            # Calculate the aspect ratio (should be close to standard table aspect)
            aspect_ratio = float(w) / h
            if aspect_ratio < 1.0 or aspect_ratio > 3.0:  # Wider range for perspective distortion
                continue
            
            # Calculate the solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = float(area) / hull_area
            
            # Calculate the extent (area / bounding rectangle area)
            rect_area = w * h
            extent = float(area) / rect_area
            
            # Calculate a score based on area, solidity, and extent
            score = area * solidity * extent
            
            if timestamp is not None and debug_img is not None:
                # Draw the current contour
                cv2.drawContours(debug_img, [approx], -1, (0, 0, 255), 2)
                cv2.putText(debug_img, f"Score: {score:.1f}", (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if score > best_score:
                best_score = score
                best_rect = rect
    
    # Save the debug image with all contours and the best one
    if timestamp is not None and debug_img is not None:
        if best_rect is not None:
            # Draw the best rectangle
            cv2.drawContours(debug_img, [best_rect.astype(int)], -1, (255, 0, 0), 3)
            
            # Draw corner points with numbers
            for j, (x, y) in enumerate(best_rect):
                cv2.circle(debug_img, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(debug_img, str(j+1), (int(x) + 10, int(y) + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        save_debug_image(debug_img, '07_contours', timestamp)
    
    return best_rect

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in the following order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left point has smallest sum
    # Bottom-right point has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Top-right point has smallest difference
    # Bottom-left point has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def detect_table_corners(image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Detect the four corners of a snooker table in the given image.
    
    Args:
        image: Input BGR image
        debug: If True, save intermediate results
        
    Returns:
        Array of 4 points (x,y) representing the table corners in order:
        [top-left, top-right, bottom-right, bottom-left]
        Returns None if table cannot be detected
    """
    # Setup debug
    timestamp = int(time.time()) if debug else None
    if debug:
        setup_debug()
        save_debug_image(image, '00_original', timestamp)
    
    # Step 1: Preprocess the image
    processed, _ = preprocess_image(image, debug=debug)
    
    # Step 2: Find the table contour
    corners = find_table_contour(processed, image, timestamp)
    
    if corners is None:
        if debug:
            print("No valid table contour found")
        return None
    
    # Step 3: Order the points
    ordered_corners = order_points(corners)
    
    # Step 4: Create debug visualization if requested
    if debug:
        debug_img = image.copy()
        
        # Draw the contour
        cv2.drawContours(debug_img, [ordered_corners.astype(int)], -1, (0, 255, 0), 2)
        
        # Draw the corners with numbers
        for i, (x, y) in enumerate(ordered_corners):
            cv2.circle(debug_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i+1), (int(x) + 10, int(y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add corner coordinates
        for i, (x, y) in enumerate(ordered_corners):
            cv2.putText(debug_img, f"{i+1}: ({int(x)}, {int(y)})", 
                       (20, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
        
        save_debug_image(debug_img, '06_final_result', timestamp)
    
    return ordered_corners

def visualize_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Draw the detected corners on the image for visualization.
    
    Args:
        image: Original BGR image
        corners: Array of 4 corner points
        
    Returns:
        Image with corners and edges drawn
    """
    vis = image.copy()
    
    # Draw the contour
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw circles at the corners
    for i, (x, y) in enumerate(corners):
        cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i+1), (int(x) + 10, int(y) + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return vis
