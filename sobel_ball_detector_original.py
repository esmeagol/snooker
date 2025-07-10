"""
Sobel-based Ball Detector

This module implements a ball detection system using the Sobel operator for edge detection
and Hough Circle Transform for identifying circular shapes in the image.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import dataclasses

@dataclasses.dataclass
class BallInfo:
    """Class to store information about a detected ball."""
    center: Tuple[float, float]  # (x, y) coordinates of the ball center
    radius: float                # Radius of the ball in pixels
    color: str                   # Detected color of the ball
    number: int                  # Ball number (if applicable)
    confidence: float            # Confidence score of the detection


class SobelBallDetector:
    """
    A class to detect snooker/pool balls on a table using the Sobel operator for edge detection.
    
    The detector uses the following pipeline:
    1. Apply Sobel operator to detect edges
    2. Combine gradients to create a magnitude image
    3. Apply thresholding to get binary edge mask
    4. Use Hough Circle Transform to detect circular shapes
    """
    
    def __init__(self, ball_size: int):
        """
        Initialize the SobelBallDetector with expected ball size.
        
        Args:
            ball_size: Expected diameter of balls in pixels
        """
        self.ball_size = ball_size
        self.min_radius = int(ball_size * 0.8 // 2)  # 80% of expected radius
        self.max_radius = int(ball_size * 1.2 // 2)  # 120% of expected radius
        self.ball_colors = {}  # Will store ball color information
    
    def set_ball_colors(self, ball_colors: Dict[str, List[int]]):
        """
        Set the expected ball colors and their corresponding numbers.
        
        Args:
            ball_colors: Dictionary mapping color names to ball numbers.
                         Example: {"red": [1, 2, 3, 4, 5, 6, 7], "yellow": 2}
        """
        self.ball_colors = ball_colors
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel operator to detect edges in the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary edge mask where white pixels represent edges
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Sobel operator in X and Y directions
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Convert to 8-bit and normalize
        deriv_frame = cv2.convertScaleAbs(gradient_magnitude)
        
        # Apply threshold to get binary edge mask
        _, edge_mask = cv2.threshold(deriv_frame, 50, 255, cv2.THRESH_BINARY)
        
        return edge_mask
    
    def detect_balls(self, frame: np.ndarray, baize_region: np.ndarray) -> List[BallInfo]:
        """
        Detect balls in the given frame within the specified baize region.
        
        Args:
            frame: Input BGR image containing the snooker table
            baize_region: Binary mask where white pixels represent the baize (playing area)
            
        Returns:
            List of BallInfo objects containing information about detected balls
        """
        # Apply mask to get only the baize region
        baize_img = cv2.bitwise_and(frame, frame, mask=baize_region)
        
        # Detect edges using Sobel operator
        edge_mask = self.detect_edges(baize_img)
        
        # Apply Hough Circle Transform to detect circular shapes
        circles = cv2.HoughCircles(
            image=edge_mask,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.ball_size,  # Minimum distance between centers
            param1=50,              # Upper threshold for edge detection
            param2=30,              # Threshold for center detection
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_balls = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                x, y, r = circle
                
                # Create a ball info object (color and number will be determined later)
                ball = BallInfo(
                    center=(float(x), float(y)),
                    radius=float(r),
                    color="unknown",
                    number=-1,
                    confidence=1.0  # Default confidence
                )
                
                detected_balls.append(ball)
        
        return detected_balls
    
    def process_frame(self, frame: np.ndarray, baize_region: np.ndarray) -> Tuple[np.ndarray, List[BallInfo]]:
        """
        Process a frame to detect and draw balls.
        
        Args:
            frame: Input BGR image
            baize_region: Binary mask of the baize region
            
        Returns:
            Tuple of (annotated_frame, detected_balls) where:
            - annotated_frame: Frame with detected balls drawn
            - detected_balls: List of BallInfo objects for detected balls
        """
        # Create a copy of the frame to draw on
        result = frame.copy()
        
        # Detect balls
        balls = self.detect_balls(frame, baize_region)
        
        # Draw detected balls
        for ball in balls:
            # Draw the circle
            cv2.circle(result, 
                      (int(ball.center[0]), int(ball.center[1])), 
                      int(ball.radius), 
                      (0, 255, 0),  # Green color
                      2)
            
            # Draw the center of the circle
            cv2.circle(result, 
                      (int(ball.center[0]), int(ball.center[1])), 
                      2, 
                      (0, 0, 255),  # Red color
                      3)
            
            # Add text with ball info
            text = f"{ball.color} ({ball.number})" if ball.number > 0 else ball.color
            cv2.putText(result, 
                       text, 
                       (int(ball.center[0] - ball.radius), int(ball.center[1] - ball.radius - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 255, 255),  # White color
                       2)
        
        return result, balls


def test_detection():
    """Test function to demonstrate the ball detection on a sample image."""
    import os
    
    # Example usage
    ball_diameter = 40  # Expected ball diameter in pixels
    detector = SobelBallDetector(ball_diameter)
    
    # Set up test data
    test_image_path = "test_table.jpg"
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        print("Please provide a test image of a snooker table.")
        return
    
    # Load test image
    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Failed to load image from {test_image_path}")
        return
    
    # Create a simple baize mask (in a real application, this would be more sophisticated)
    height, width = frame.shape[:2]
    baize_region = np.ones((height, width), dtype=np.uint8) * 255  # All white mask
    
    # Process the frame
    result, balls = detector.process_frame(frame, baize_region)
    
    # Display results
    cv2.imshow("Detected Balls", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Detected {len(balls)} balls")


if __name__ == "__main__":
    test_detection()
