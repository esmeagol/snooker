"""
Snooker Ball Detector

This module provides functionality to detect and classify snooker balls in images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Ball:
    position: Tuple[int, int]
    radius: int
    hsv: Tuple[float, float, float]
    color: str = 'unknown'
    confidence: float = 0.0
    
    @property
    def x(self) -> int:
        return self.position[0]
    
    @property
    def y(self) -> int:
        return self.position[1]

class BallDetector:
    # Color ranges in HSV format (H: 0-179, S: 0-255, V: 0-255)
    COLOR_RANGES = {
        'red1':    {'lower': np.array([0, 180, 100]),   'upper': np.array([10, 255, 200])},  # Red (wrapped around 0)
        'red2':    {'lower': np.array([170, 180, 100]), 'upper': np.array([179, 255, 200])}, # Red (wrapped around 180)
        'yellow':  {'lower': np.array([15, 150, 160]),  'upper': np.array([30, 255, 255])},
        'black':   {'lower': np.array([0, 0, 0]),       'upper': np.array([179, 255, 50])},
        'white':   {'lower': np.array([0, 0, 200]),     'upper': np.array([179, 50, 255])},
        'pink':    {'lower': np.array([130, 100, 190]), 'upper': np.array([179, 255, 255])},
        'brown':   {'lower': np.array([0, 0, 0]),       'upper': np.array([50, 150, 180])},
        'blue':    {'lower': np.array([90, 150, 150]),  'upper': np.array([140, 255, 200])},
        'green':   {'lower': np.array([80, 150, 100]),  'upper': np.array([100, 255, 255])}
    }
    
    # Standard snooker ball counts (excluding reds)
    STANDARD_BALL_COUNTS = {
        'yellow': 1, 'green': 1, 'brown': 1,
        'blue': 1, 'pink': 1, 'black': 1, 'white': 1
    }
    
    # Color mapping for display
    COLOR_DISPLAY = {
        'red': (0, 0, 255),     # Red
        'yellow': (0, 255, 255), # Yellow
        'black': (0, 0, 0),     # Black
        'white': (255, 255, 255), # White
        'pink': (203, 192, 255), # Pink
        'brown': (42, 42, 165),  # Brown
        'blue': (255, 0, 0),     # Blue
        'green': (0, 128, 0),    # Green
        'unknown': (128, 128, 128) # Gray for unknown
    }

    def __init__(self, frame: np.ndarray):
        """
        Initialize the BallDetector with a frame for table boundary selection.
        
        Args:
            frame: Initial frame from the video to select table boundaries
        """
        self.frame = frame.copy()
        self.table_boundary = None
        self.mask = None
        self.ball_counts = {color: 0 for color in self.STANDARD_BALL_COUNTS}
        self.ball_counts['red'] = 0  # Will be updated based on detection
        
    def select_table_boundary(self) -> Optional[np.ndarray]:
        """
        Allow user to select 4 points defining the table boundary.
        Returns a mask for the table area.
        """
        points = []
        clone = self.frame.copy()
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append((x, y))
                    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                    if len(points) > 1:
                        cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                    if len(points) == 4:
                        cv2.line(clone, points[-1], points[0], (0, 255, 0), 2)
                    cv2.imshow('Select Table Boundary (4 points)', clone)
        
        cv2.namedWindow('Select Table Boundary (4 points)')
        cv2.setMouseCallback('Select Table Boundary (4 points)', click_event)
        
        print("Select 4 points to define the table boundary (in order: top-left, top-right, bottom-right, bottom-left)")
        
        while True:
            cv2.imshow('Select Table Boundary (4 points)', clone)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or len(points) >= 4:  # ESC key or 4 points selected
                break
        
        cv2.destroyAllWindows()
        
        if len(points) != 4:
            print("Error: Exactly 4 points must be selected.")
            return None
        
        self.table_boundary = np.array(points, dtype=np.int32)
        
        # Create mask for the table area
        self.mask = np.zeros_like(self.frame[:, :, 0])
        cv2.fillPoly(self.mask, [self.table_boundary], 255)
        
        return self.mask
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame to highlight snooker balls using color segmentation and contrast enhancement.
        """
        try:
            logger.debug("Starting frame preprocessing...")
            
            # Convert to HSV color space for better color segmentation
            logger.debug("Converting to HSV color space...")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for snooker balls (excluding green table)
            # Red balls (account for red hue wrap-around)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Yellow balls
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # Blue balls
            lower_blue = np.array([90, 100, 100])
            upper_blue = np.array([130, 255, 255])
            
            # Pink/White balls (high value, low saturation)
            lower_pink = np.array([140, 30, 200])
            upper_pink = np.array([180, 100, 255])
            
            # Create masks for each color
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
            
            # Combine all color masks
            ball_mask = cv2.bitwise_or(mask_red, mask_yellow)
            ball_mask = cv2.bitwise_or(ball_mask, mask_blue)
            ball_mask = cv2.bitwise_or(ball_mask, mask_pink)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            ball_mask = cv2.erode(ball_mask, kernel, iterations=1)
            ball_mask = cv2.dilate(ball_mask, kernel, iterations=2)
            
            # Apply the table mask if available
            if hasattr(self, 'mask') and self.mask is not None:
                logger.debug("Applying table mask...")
                ball_mask = cv2.bitwise_and(ball_mask, self.mask)
            
            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply the ball mask to the enhanced image
            result = cv2.bitwise_and(enhanced, enhanced, mask=ball_mask)
            
            # Apply adaptive thresholding to highlight the balls
            thresh = cv2.adaptiveThreshold(
                result, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Final morphological operations
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            
            logger.debug("Frame preprocessing completed")
            return thresh
            
        except Exception as e:
            logger.error(f"Error in preprocess_frame: {str(e)}", exc_info=True)
            # Fallback to simple grayscale conversion if preprocessing fails
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def detect_edges_sobel(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect edges in the frame using a combination of Sobel and Canny edge detection.
        Returns a binary edge image optimized for circle detection.
        """
        logger = logging.getLogger(__name__)
        logger.debug("Starting edge detection...")
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                logger.debug("Converted to grayscale")
            else:
                gray = frame
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            logger.debug("Applying CLAHE...")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            
            # Apply bilateral filter to reduce noise while preserving edges
            logger.debug("Applying bilateral filter...")
            bilateral = cv2.bilateralFilter(clahe_img, 9, 75, 75)
            
            # Apply adaptive thresholding to enhance contrast
            logger.debug("Applying adaptive thresholding...")
            thresh = cv2.adaptiveThreshold(
                bilateral, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            logger.debug("Applying morphological operations...")
            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
            
            # Apply Canny edge detection
            logger.debug("Applying Canny edge detection...")
            edges_canny = cv2.Canny(morph, 30, 100, L2gradient=True)
            
            # Apply Sobel edge detection
            logger.debug("Applying Sobel edge detection...")
            sobelx = cv2.Sobel(morph, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(morph, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.uint8(sobel / np.max(sobel) * 255)
            _, sobel = cv2.threshold(sobel, 40, 255, cv2.THRESH_BINARY)
            
            # Combine Canny and Sobel results
            logger.debug("Combining edge detection results...")
            edges_combined = cv2.bitwise_or(edges_canny, sobel)
            
            # Apply another round of morphological operations
            kernel = np.ones((2, 2), np.uint8)
            edges_combined = cv2.dilate(edges_combined, kernel, iterations=1)
            edges_combined = cv2.erode(edges_combined, kernel, iterations=1)
            
            # If we have a table mask, apply it to the edges
            if hasattr(self, 'mask') and self.mask is not None:
                logger.debug("Applying table mask to edges...")
                edges_combined = cv2.bitwise_and(edges_combined, self.mask)
            
            logger.debug("Edge detection completed")
            return edges_combined
            
        except Exception as e:
            logger.error(f"Error in detect_edges_sobel: {str(e)}", exc_info=True)
            # Return an empty edge map on error
            return np.zeros_like(gray) if 'gray' in locals() else np.zeros_like(frame)
    
    def detect_circles_2(self, edges: np.ndarray, min_radius: int = 10, max_radius: int = 100) -> List[Tuple[int, int, int]]:
        """
        Detect circles in the edge image using Hough Circle Transform with improved parameters.
        Returns list of (x, y, radius) tuples.
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting circle detection with min_radius={min_radius}, max_radius={max_radius}")
        
        try:
            # Check if edges image is valid
            if edges is None or edges.size == 0:
                logger.warning("Empty edge image provided to detect_circles")
                return []
            
            # Get image dimensions
            h, w = edges.shape
            logger.debug(f"Image dimensions: {w}x{h}")
            
            # Calculate dynamic parameters based on image size
            min_radius = int(w * 0.05)
            max_radius = int(w * 0.1)
            
            # Apply Hough Circle Transform
            circles = cv2.HoughCircles(
                edges, cv2.HOUGH_GRADIENT, dp=1, minDist=w//8,
                param1=50, param2=30,
                minRadius=min_radius, maxRadius=max_radius
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw the outer circle
                    cv2.circle(self.frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(self.frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                logger.debug(f"Detected {len(circles[0, :])} circles")
                return circles[0, :].tolist()
            else:
                logger.debug("No circles detected")
                return []
            
        except Exception as e:
            logger.error(f"Error in detect_circles_2: {str(e)}", exc_info=True)
            return []
    
    def detect_circles(self, edges: np.ndarray, min_radius: int = 10, max_radius: int = 100) -> List[Tuple[int, int, int]]:
        """
        Detect circles in the edge image using Hough Circle Transform with improved parameters.
        Returns list of (x, y, radius) tuples.
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting circle detection with min_radius={min_radius}, max_radius={max_radius}")
        
        try:
            # Check if edges image is valid
            if edges is None or edges.size == 0:
                logger.warning("Empty edge image provided to detect_circles")
                return []
            
            # Get image dimensions
            h, w = edges.shape
            logger.debug(f"Image dimensions: {w}x{h}")
            
            # Calculate dynamic parameters based on image size
            img_area = w * h
            min_radius = max(5, int(min(w, h) * 0.01))  # At least 1% of min dimension
            max_radius = min(150, int(min(w, h) * 0.2))  # At most 20% of min dimension
            min_dist = max(20, int(min(w, h) * 0.03))   # At least 3% of min dimension
            
            logger.debug(f"Adjusted parameters - min_radius: {min_radius}, max_radius: {max_radius}, min_dist: {min_dist}")
            
            # Apply Gaussian blur to reduce noise
            logger.debug("Applying Gaussian blur...")
            blurred_edges = cv2.GaussianBlur(edges, (5, 5), 0)
            
            # Try multiple parameter sets to detect circles of different sizes
            all_circles = []
            
            # Parameter set 1: For smaller balls (more lenient parameters)
            logger.debug("Trying parameter set 1 (smaller balls)...")
            circles1 = cv2.HoughCircles(
                blurred_edges, 
                cv2.HOUGH_GRADIENT, 
                dp=1.0,            # Lower for more precise detection
                minDist=min_dist,   # Minimum distance between circle centers
                param1=30,          # Lower edge detection threshold
                param2=15,          # Lower accumulator threshold (more circles)
                minRadius=min_radius,
                maxRadius=max_radius // 2
            )
            
            # Parameter set 2: For larger balls (more lenient parameters)
            logger.debug("Trying parameter set 2 (larger balls)...")
            circles2 = cv2.HoughCircles(
                blurred_edges,
                cv2.HOUGH_GRADIENT,
                dp=1.2,            # Slightly higher for larger circles
                minDist=min_dist,
                param1=25,          # Lower threshold for larger circles
                param2=12,          # Lower accumulator threshold
                minRadius=max_radius // 3,
                maxRadius=max_radius
            )
            
            # Parameter set 3: Very lenient parameters for difficult cases
            logger.debug("Trying parameter set 3 (very lenient)...")
            circles3 = cv2.HoughCircles(
                blurred_edges,
                cv2.HOUGH_GRADIENT,
                dp=0.9,
                minDist=min_dist // 2,
                param1=20,
                param2=8,           # Very low threshold to catch faint circles
                minRadius=max(5, min_radius // 2),
                maxRadius=max_radius
            )
            
            # Combine results from all parameter sets
            if circles1 is not None:
                all_circles.extend(circles1[0, :])
            if circles2 is not None:
                all_circles.extend(circles2[0, :])
            if circles3 is not None:
                all_circles.extend(circles3[0, :])
            
            if not all_circles:
                logger.debug("No circles detected with any parameter set")
                return []
                
            logger.debug(f"Found {len(all_circles)} raw circles before filtering")
            
            # Convert to list of (x, y, radius) tuples and round to integers
            circles = np.round(np.array(all_circles)).astype(int)
            logger.debug(f"Found {len(circles)} raw circles before filtering")
            
            # Filter circles to ensure they're within the table boundary
            valid_circles = []
            for (x, y, r) in circles:
                # Skip circles that are too small or too large
                if r < min_radius or r > max_radius:
                    continue
                    
                # Skip circles that are too close to the image edges
                margin = r + 5  # Add some margin
                if x < margin or x > w - margin or y < margin or y > h - margin:
                    continue
                
                # Check if the circle is mostly within the table mask
                if hasattr(self, 'mask') and self.mask is not None:
                    # Create a mask for this circle
                    circle_mask = np.zeros_like(self.mask)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    # Check what percentage of the circle is within the table
                    intersection = cv2.bitwise_and(circle_mask, self.mask)
                    circle_area = np.pi * r * r
                    intersection_area = np.count_nonzero(intersection)
                    if intersection_area < circle_area * 0.7:  # At least 70% inside table
                        continue
                
                valid_circles.append((x, y, r))
            
            # Remove duplicate or overlapping circles
            filtered_circles = []
            for i, (x1, y1, r1) in enumerate(valid_circles):
                keep = True
                for x2, y2, r2 in filtered_circles:
                    # Calculate distance between centers
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    # If circles are too close, keep the one with higher radius
                    if dist < (r1 + r2) * 0.7:  # 70% overlap threshold
                        if r1 > r2 * 1.2:  # If current circle is significantly larger
                            filtered_circles.remove((x2, y2, r2))
                        else:
                            keep = False
                            break
                if keep:
                    filtered_circles.append((x1, y1, r1))
            
            logger.debug(f"Returning {len(filtered_circles)} filtered circles")
            return filtered_circles
            
        except Exception as e:
            logger.error(f"Error in detect_circles: {str(e)}", exc_info=True)
            return []
    
    def classify_ball_color(self, hsv: Tuple[float, float, float]) -> Tuple[str, float]:
        """
        Classify a ball's color based on its HSV values.
        Returns (color_name, confidence) where confidence is 0-1.
        """
        h, s, v = hsv
        
        # Special case for red (wraps around 0/180)
        if (self.COLOR_RANGES['red1']['lower'][0] <= h <= self.COLOR_RANGES['red1']['upper'][0] or
            self.COLOR_RANGES['red2']['lower'][0] <= h <= self.COLOR_RANGES['red2']['upper'][0]):
            if (self.COLOR_RANGES['red1']['lower'][1] <= s <= self.COLOR_RANGES['red1']['upper'][1] and
                self.COLOR_RANGES['red1']['lower'][2] <= v <= self.COLOR_RANGES['red1']['upper'][2]):
                return 'red', 1.0
        
        # Check other colors
        for color, ranges in self.COLOR_RANGES.items():
            if color in ['red1', 'red2']:
                continue  # Already checked red
                
            if (ranges['lower'][0] <= h <= ranges['upper'][0] and
                ranges['lower'][1] <= s <= ranges['upper'][1] and
                ranges['lower'][2] <= v <= ranges['upper'][2]):
                return color, 1.0
        
        return 'unknown', 0.0
    
    def extract_ball_colors(self, frame: np.ndarray, circles: List[Tuple[int, int, int]]) -> List[Ball]:
        """
        Extract and classify balls based on their colors.
        Returns a list of Ball objects with color information.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        balls = []
        
        for circle in circles:
            x, y, r = circle
            
            # Skip balls that are too small or too large
            if r < 3 or r > 50:
                continue
            
            # Create a mask for the current circle
            mask = np.zeros_like(hsv_frame[:, :, 0])
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Get the HSV values within the circle
            mean_hsv = cv2.mean(hsv_frame, mask=mask)[:3]
            
            # Classify the ball color
            color, confidence = self.classify_ball_color(mean_hsv)
            
            # Create Ball object
            ball = Ball(
                position=(int(x), int(y)),
                radius=int(r),
                hsv=mean_hsv,
                color=color,
                confidence=confidence
            )
            
            balls.append(ball)
        
        return balls
    
    def enforce_ball_counts(self, balls: List[Ball]) -> List[Ball]:
        """
        Enforce standard snooker ball counts by keeping only the most confident detections.
        Allows multiple red balls (6-15) but only one of each other color.
        """
        # Reset counts
        counts = {color: 0 for color in self.ball_counts}
        
        # Sort balls by confidence (highest first)
        sorted_balls = sorted(balls, key=lambda b: b.confidence, reverse=True)
        
        result = []
        
        for ball in sorted_balls:
            color = ball.color
            
            # For red balls, allow multiple (up to 15)
            if color == 'red':
                if counts['red'] < 15:  # Maximum 15 red balls
                    result.append(ball)
                    counts['red'] += 1
            # For other colors, allow only one of each
            elif color in self.STANDARD_BALL_COUNTS and counts[color] < 1:
                result.append(ball)
                counts[color] += 1
            # Keep unknown balls if we have space
            elif color == 'unknown' and sum(counts.values()) < 22:  # 15 red + 6 colors + cue
                result.append(ball)
        
        self.ball_counts = counts
        return result
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame to detect and classify snooker balls.
        Returns a dictionary containing edge image, detected circles, and ball information.
        """
        if self.mask is None:
            raise ValueError("Table boundary not selected. Call select_table_boundary() first.")
        
        # Preprocess the frame
        # preprocessed = self.preprocess_frame(frame)
        
        # Detect edges using Sobel
        edges = self.detect_edges_sobel(frame)
        
        # Detect circles
        circles = self.detect_circles(edges)
        
        # Extract and classify ball colors
        balls = self.extract_ball_colors(frame, circles)
        
        # Enforce standard ball counts
        filtered_balls = self.enforce_ball_counts(balls)
        
        return {
            'edges': edges,
            'circles': circles,
            'balls': filtered_balls,
            'masked_frame': cv2.bitwise_and(frame, frame, mask=self.mask)
        }

def draw_ball_info(frame: np.ndarray, ball: Ball) -> None:
    """Draw ball information on the frame."""
    color = BallDetector.COLOR_DISPLAY.get(ball.color, (128, 128, 128))
    text = f"{ball.color}"
    
    # Draw the circle
    cv2.circle(frame, (ball.x, ball.y), ball.radius, color, 2)
    cv2.circle(frame, (ball.x, ball.y), 2, (0, 0, 0), -1)  # Center dot
    
    # Draw the text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = ball.x - text_size[0] // 2
    text_y = ball.y + ball.radius + 15
    
    # Draw text background
    cv2.rectangle(frame, 
                 (text_x - 2, text_y - text_size[1] - 2),
                 (text_x + text_size[0] + 2, text_y + 2),
                 (255, 255, 255), -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def test_detection(video_path: str):
    """
    Test the ball detection on a video file with color classification.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Read the first frame for table boundary selection
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        return
    
    # Initialize detector and select table boundary
    detector = BallDetector(frame)
    if detector.select_table_boundary() is None:
        return
    
    cv2.namedWindow('Ball Detection', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for better performance (process every 2nd frame)
        if frame_count % 2 != 0:
            continue
        
        # Process the frame
        result = detector.process_frame(frame)
        
        # Create output frame
        output = frame.copy()
        
        # Draw table boundary
        if detector.table_boundary is not None:
            cv2.polylines(output, [detector.table_boundary], True, (255, 0, 0), 2)
        
        # Draw detected balls with color information
        for ball in result['balls']:
            draw_ball_info(output, ball)
        
        # Display ball counts
        y_offset = 30
        for color, count in detector.ball_counts.items():
            if count > 0 or color == 'red':  # Always show red count
                text = f"{color}: {count}"
                cv2.putText(output, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y_offset += 25
        
        # Display results
        cv2.imshow('Ball Detection', output)
        
        # Show edges (for debugging)
        cv2.imshow('Edges', result['edges'])
        
        # Exit on 'q' press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ball_detector.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_detection(video_path)
