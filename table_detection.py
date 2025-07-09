import os
import time
import cv2
import numpy as np
import itertools
from typing import List, Tuple, Optional

# Global counter for debug images
step = 0

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in the following order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left
    
    Args:
        pts: Input array of 4 points
        
    Returns:
        Ordered array of 4 points
    """
    # Initialize an array with the same data type as input
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def setup_debug() -> None:
    """Create debug directory if it doesn't exist."""
    os.makedirs('debug', exist_ok=True)

def save_debug_image(image: np.ndarray, name: str, timestamp: int = None) -> None:
    """Save an image to the debug directory with a timestamp."""
    global step
    setup_debug()
    if timestamp is None:
        timestamp = int(time.time())
    cv2.imwrite(f'debug/{step}_{name}_{timestamp}.jpg', image)
    step += 1
    print(f"Debug: Saved {name} to debug/{step-1}_{name}_{timestamp}.jpg")

def preprocess_image(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """Preprocess the image for table detection using color-based segmentation."""
    timestamp = int(time.time()) if debug else None
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        save_debug_image(gray, 'gray', timestamp)
    
    # Apply Gaussian blur
    blurred = cv2.bilateralFilter(gray, 5, 75, 75)
    if debug:
        save_debug_image(blurred, 'blurred', timestamp)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    if debug:
        save_debug_image(edges, 'edges', timestamp)
    
    # Dilate edges to connect nearby edges
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    if debug:
        save_debug_image(dilated, 'dilated', timestamp)
    
    # Erode to remove noise
    eroded = cv2.erode(dilated, np.ones((3, 3), np.uint8), iterations=1)
    if debug:
        save_debug_image(eroded, 'eroded', timestamp)
    
    # Probabilistic Hough Transform: Apply cv2.HoughLinesP to detect lines. Adjust the minLineLength parameter to ensure only longer lines are detected. For example, setting minLineLength=100 will filter out lines shorter than 100 pixels.
    lines = cv2.HoughLinesP(eroded, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=10) 
    lines_ndarray = np.array(lines)  # Convert to NumPy ndarray
    # draw lines on binary image
    line_image = np.zeros_like(gray)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    if debug:
        save_debug_image(line_image, 'lines', timestamp)

    return line_image

def detect_table_corners(image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Detect the four corners of a snooker table in the given image using contour detection.
    
    Args:
        image: Input BGR image
        debug: If True, save intermediate results and print debug info
        
    Returns:
        Array of 4 points (x,y) representing the table corners in order:
        [top-left, top-right, bottom-right, bottom-left]
        Returns None if table cannot be detected
    """
    debug_dir = 'debug'
    timestamp = int(time.time()) if debug else None
    
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        print("Starting table corner detection using contour-based approach...")
    
    # 1. Preprocess the image to get edges
    processed = preprocess_image(image, debug=debug)
    
    # # Use adaptive thresholding to handle varying lighting conditions
    # thresh = cv2.adaptiveThreshold(
    #     processed, 255, 
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY_INV, 
    #     11, 2
    # )
    
    # if debug and timestamp is not None:
    #     cv2.imwrite(f'{debug_dir}/01_threshold_{timestamp}.jpg', thresh)
    
    # 2. Find contours in the thresholded image
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        debug_img = image.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        save_debug_image(debug_img, 'contours', timestamp)
    
    # 3. Find the best quadrilateral contour
    best_quad = None
    max_area = 0
    min_contour_area = image.shape[0] * image.shape[1] * 0.6  # At least 60% of image area
    
    for contour in contours:
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
            
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # We're looking for a quadrilateral (4 vertices)
        if len(approx) == 4:
            # Check if the area is the largest so far
            if area > max_area:
                max_area = area
                best_quad = approx.reshape(4, 2)
    
    # If no perfect quad found, try with more lenient parameters
    if best_quad is None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
                
            # Try with a larger epsilon to get a simpler polygon
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we have more than 4 points, try to find a quad
            if len(approx) >= 4:
                # Get convex hull to simplify the shape
                hull = cv2.convexHull(approx, returnPoints=True)
                if len(hull) >= 4:
                    # Take the largest area quadrilateral from the convex hull
                    max_quad_area = 0
                    for quad_indices in itertools.combinations(range(len(hull)), 4):
                        quad = hull[list(quad_indices), 0, :]
                        quad_area = cv2.contourArea(quad)
                        if quad_area > max_quad_area:
                            max_quad_area = quad_area
                            best_quad = quad
                    
                    if best_quad is not None and max_quad_area > max_area:
                        max_area = max_quad_area
    
    # If still no quad found, use the minimum area rectangle
    if best_quad is None and len(contours) > 0:
        if debug:
            print("No good quadrilateral found, trying minimum area rectangle...")
        # Find the contour with maximum area
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        best_quad = cv2.boxPoints(rect)
    
    if best_quad is None:
        if debug:
            print("Could not find a valid table contour")
        return None
    
    # Order the points consistently
    ordered_quad = order_points(best_quad)
    
    # 4. Create debug visualization if requested
    if debug:
        debug_img = image.copy()
        
        # Draw the quadrilateral
        cv2.polylines(debug_img, [ordered_quad.astype(int)], True, (0, 255, 0), 3)
        
        # Draw the corners with numbers
        for i, (x, y) in enumerate(ordered_quad):
            cv2.circle(debug_img, (int(x), int(y)), 10, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i+1), (int(x) + 15, int(y) + 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add corner coordinates
        for i, (x, y) in enumerate(ordered_quad):
            cv2.putText(debug_img, f"{i+1}: ({int(x)}, {int(y)})", 
                       (20, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
        
        save_debug_image(debug_img, 'final_result', timestamp)
        
        # Print corner coordinates
        print("\n--- Final Detected Corners ---")
        print(f"Table area: {cv2.contourArea(ordered_quad):.1f} pixels")
        for i, (x, y) in enumerate(ordered_quad):
            print(f"  Corner {i+1}: ({x:.1f}, {y:.1f})")
    
    return ordered_quad

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
    
    # Draw the quadrilateral
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw circles and numbers at the corners
    for i, (x, y) in enumerate(corners):
        # Draw corner point
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
        
        # Draw corner number
        cv2.putText(vis, str(i+1), 
                   (int(x) + 15, int(y) + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw coordinates
        cv2.putText(vis, f"({int(x)}, {int(y)})",
                   (int(x) + 15, int(y) + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add a legend
    cv2.putText(vis, "Detected Table Corners", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return vis
