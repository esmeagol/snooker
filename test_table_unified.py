"""
Unified test script for table detection implementations.
Set USE_V2 to True to test table_detection_2, or False to test table_detection.
"""
import os
import cv2
import numpy as np
import time
import unittest
from pathlib import Path

# Set this flag to switch between implementations
USE_V2 = True

if USE_V2:
    from table_detection_2 import detect_table_corners, visualize_corners, order_points
    IMPL_NAME = "table_detection_2"
else:
    from table_detection import detect_table_corners, visualize_corners, order_points
    IMPL_NAME = "table_detection"

class TestTableDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load test images once for all test methods"""
        # Test image paths
        test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cls.test_images = [
            os.path.join(test_data_dir, '1.jpg'),
            os.path.join(test_data_dir, '2.jpg')
        ]
        
        # Create output directory for test results
        cls.output_dir = f"test_output_{IMPL_NAME}"
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Load test images
        cls.images = []
        for img_path in cls.test_images:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Test image not found: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            cls.images.append((img_path, img))
    
    def test_detect_table_corners(self):
        """Test that table corners can be detected"""
        for img_path, img in self.images:
            print(f"\nTesting: {os.path.basename(img_path)}")
            
            # Detect corners with debug output
            start_time = time.time()
            corners = detect_table_corners(img.copy(), debug=True)
            elapsed = time.time() - start_time
            
            # Basic validation of corners
            self.assertIsNotNone(corners, "Failed to detect corners")
            self.assertEqual(corners.shape, (4, 2), "Expected 4 corners with (x,y) coordinates")
            
            # Check that corners are within image bounds
            height, width = img.shape[:2]
            for x, y in corners:
                self.assertGreaterEqual(x, 0, f"Corner x-coordinate {x} is out of bounds")
                self.assertLess(x, width, f"Corner x-coordinate {x} exceeds image width {width}")
                self.assertGreaterEqual(y, 0, f"Corner y-coordinate {y} is out of bounds")
                self.assertLess(y, height, f"Corner y-coordinate {y} exceeds image height {height}")
            
            # Visualize and save result
            vis = visualize_corners(img, corners)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_result.jpg")
            cv2.imwrite(output_path, vis)
            
            # Print detection info
            print(f"  ✓ Table detected in {elapsed:.2f} seconds")
            print(f"  ✓ Saved result to {output_path}")
            print("  Detected corners (x, y):")
            for i, (x, y) in enumerate(corners, 1):
                print(f"    Corner {i}: ({x:.1f}, {y:.1f})")
            
            # Check that corners form a reasonable quadrilateral
            ordered = order_points(corners)
            
            # Calculate edge lengths
            def distance(p1, p2):
                return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                
            top_width = distance(ordered[0], ordered[1])
            bottom_width = distance(ordered[3], ordered[2])
            left_height = distance(ordered[0], ordered[3])
            right_height = distance(ordered[1], ordered[2])
            
            print(f"  Table dimensions (pixels):")
            print(f"    Top width: {top_width:.1f}")
            print(f"    Bottom width: {bottom_width:.1f}")
            print(f"    Left height: {left_height:.1f}")
            print(f"    Right height: {right_height:.1f}")
            
            # Basic shape validation
            self.assertGreater(top_width, 100, "Table width seems too small")
            self.assertGreater(left_height, 100, "Table height seems too small")
            
            # Check that aspect ratio is reasonable (roughly 1:2 for snooker tables)
            aspect_ratio = (top_width + bottom_width) / (left_height + right_height)
            self.assertGreater(aspect_ratio, 1.5, "Table aspect ratio seems too narrow")
            self.assertLess(aspect_ratio, 2.5, "Table aspect ratio seems too wide")

def run_standalone():
    """Run tests without unittest for better debugging"""
    print(f"Testing implementation: {IMPL_NAME}")
    
    # Create output directory
    output_dir = f"test_output_{IMPL_NAME}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test images
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    test_images = [
        os.path.join(test_data_dir, '1.jpg'),
        os.path.join(test_data_dir, '2.jpg')
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Warning: Test image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image: {img_path}")
            continue
        
        print(f"\nProcessing: {os.path.basename(img_path)}")
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            # Run detection with debug output
            start_time = time.time()
            corners = detect_table_corners(img.copy(), debug=True)
            elapsed = time.time() - start_time
            
            if corners is not None:
                print(f"  ✓ Table detected in {elapsed:.2f} seconds")
                
                # Visualize the result
                vis = visualize_corners(img, corners)
                
                # Save the result
                output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
                cv2.imwrite(output_path, vis)
                print(f"  ✓ Saved result to {output_path}")
                
                # Print corner coordinates
                print("  Detected corners (x, y):")
                for i, (x, y) in enumerate(corners, 1):
                    print(f"    Corner {i}: ({x:.1f}, {y:.1f})")
                
                # Calculate and print table dimensions
                ordered = order_points(corners)
                def distance(p1, p2):
                    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                
                print(f"  Table dimensions (pixels):")
                print(f"    Top width: {distance(ordered[0], ordered[1]):.1f}")
                print(f"    Bottom width: {distance(ordered[3], ordered[2]):.1f}")
                print(f"    Left height: {distance(ordered[0], ordered[3]):.1f}")
                print(f"    Right height: {distance(ordered[1], ordered[2]):.1f}")
            else:
                print("  ✗ Failed to detect table")
                
        except Exception as e:
            print(f"  ✗ Error processing image: {str(e)}")

if __name__ == "__main__":
    print(f"Running tests for {IMPL_NAME}")
    print("1. Run unittests (press 1)")
    print("2. Run standalone test (press 2)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        run_standalone()
