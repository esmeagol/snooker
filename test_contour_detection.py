import unittest
import os
import cv2
import numpy as np
from table_detection import detect_table_corners, visualize_corners, order_points

class TestContourTableDetection(unittest.TestCase):
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
        cls.output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Load test images
        cls.images = []
        for img_path in cls.test_images:
            if not os.path.exists(img_path):
                print(f"Warning: Test image not found: {img_path}")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image: {img_path}")
                continue
            cls.images.append((os.path.basename(img_path), img))
    
    def test_detect_table_corners(self):
        """Test that table corners can be detected"""
        for img_name, img in self.images:
            with self.subTest(image=img_name):
                # Run detection with debug output
                corners = detect_table_corners(img, debug=True)
                
                # Verify we got a result
                self.assertIsNotNone(corners, f"No table detected in {img_name}")
                self.assertEqual(corners.shape, (4, 2), 
                              f"Expected 4 corners, got {corners.shape} in {img_name}")
                
                # Verify corners are within image bounds with small tolerance for floating point
                height, width = img.shape[:2]
                for i, (x, y) in enumerate(corners):
                    # Check with small tolerance for floating point inaccuracies
                    self.assertTrue(x >= -1e-6, 
                                 f"Corner {i+1} x-coordinate ({x}) out of bounds in {img_name}")
                    self.assertTrue(x <= width + 1e-6, 
                                 f"Corner {i+1} x-coordinate ({x}) out of bounds in {img_name}")
                    self.assertTrue(y >= -1e-6, 
                                 f"Corner {i+1} y-coordinate ({y}) out of bounds in {img_name}")
                    self.assertTrue(y <= height + 1e-6, 
                                 f"Corner {i+1} y-coordinate ({y}) out of bounds in {img_name}")
                    
                    # Clamp values to image bounds for visualization
                    corners[i] = [
                        max(0, min(x, width - 1)),
                        max(0, min(y, height - 1))
                    ]
                
                # Save visualization
                if corners is not None:
                    vis = visualize_corners(img, corners)
                    output_path = os.path.join(self.output_dir, f"result_{img_name}")
                    cv2.imwrite(output_path, vis)
                    print(f"Saved visualization to {output_path}")
    
    def test_order_points(self):
        """Test that points are ordered correctly"""
        # Create a test quadrilateral
        test_quad = np.array([
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 200],  # bottom-right
            [100, 200]   # bottom-left
        ], dtype=np.float32)
        
        # Shuffle the points
        np.random.shuffle(test_quad)
        
        # Order the points
        ordered = order_points(test_quad)
        
        # Verify order is correct
        expected_order = [
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 200],  # bottom-right
            [100, 200]   # bottom-left
        ]
        
        np.testing.assert_array_equal(ordered, expected_order, 
                                    "Points are not ordered correctly")

if __name__ == "__main__":
    unittest.main(verbosity=2)
