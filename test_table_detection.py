import unittest
import os
import cv2
import numpy as np
from table_detection import (
    edge_detection,
    lines_detection,
    find_intersections,
    order_points,
    detect_table_corners,
    visualize_corners,
    find_quadrilateral_corners,
    compute_intersection
)

class TestTableDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load test images once for all test methods"""
        # Test image paths
        cls.test_images = [
            "/Users/abhinavrai/Playground/snooker/test_data/1.jpg",
            "/Users/abhinavrai/Playground/snooker/test_data/2.jpg"
        ]
        
        # Load test images
        cls.images = []
        for img_path in cls.test_images:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Test image not found: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            cls.images.append(img)
    
    def test_edge_detection(self):
        """Test that edge detection works correctly"""
        for img in self.images:
            edges = edge_detection(img)
            self.assertEqual(len(edges.shape), 2, 
                          "Edge image should be single-channel")
            # Check that we have some edges (not all black)
            self.assertGreater(np.sum(edges > 0), 100, 
                            "Edge image should contain some edges")
    
    def test_lines_detection(self):
        """Test that line detection works correctly"""
        for img in self.images:
            edges = edge_detection(img)
            lines = lines_detection(edges)
            self.assertIsInstance(lines, list, "Should return a list of lines")
            if lines:  # If lines were found
                for rho, theta in lines:
                    self.assertIsInstance(rho, (int, float), "Rho should be a number")
                    self.assertIsInstance(theta, (int, float), "Theta should be a number")
    
    def test_order_points(self):
        """Test that points are ordered correctly"""
        # Create a test rectangle
        rect = np.array([
            [100, 200],  # top-right
            [100, 100],  # top-left
            [200, 100],  # bottom-left
            [200, 200]   # bottom-right
        ])
        
        # Shuffle the points
        np.random.shuffle(rect)
        
        # Order the points
        ordered = order_points(rect)
        
        # Check the order
        expected_order = [
            [100, 100],  # top-left
            [200, 100],  # top-right
            [200, 200],  # bottom-right
            [100, 200]   # bottom-left
        ]
        
        np.testing.assert_array_equal(ordered, expected_order)
    
    def test_detect_table_corners_shape(self):
        """Test that detect_table_corners returns correct shape"""
        for img in self.images:
            corners = detect_table_corners(img)
            self.assertIsNotNone(corners, "Failed to detect corners")
            self.assertEqual(corners.shape, (4, 2), 
                           f"Expected 4 points with 2 coordinates, got {corners.shape}")
    
    def test_detect_table_corners_values(self):
        """
        Test that detected corners are within image bounds and have expected geometric properties.
        Validates:
        1. All points are within image bounds
        2. Top edge (top-left to top-right) is approximately 300-350 pixels
        3. Bottom edge (bottom-left to bottom-right) is approximately 400-450 pixels
        """
        for img in self.images:
            height, width = img.shape[:2]
            corners = detect_table_corners(img)
            
            # Check all points are within image bounds
            for x, y in corners:
                self.assertGreaterEqual(x, 0, "X coordinate out of bounds (negative)")
                self.assertGreaterEqual(y, 0, "Y coordinate out of bounds (negative)")
                self.assertLess(x, width, "X coordinate out of bounds (exceeds width)")
                self.assertLess(y, height, "Y coordinate out of bounds (exceeds height)")
            
            # Extract the four corners
            top_left, top_right, bottom_right, bottom_left = corners
            
            # Calculate distances
            def distance(p1, p2):
                return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            # Check top edge (top-left to top-right)
            top_edge = distance(top_left, top_right)
            self.assertGreaterEqual(top_edge, 50, "Top edge too short (min 50px expected)")
            self.assertLessEqual(top_edge, 200, f"Top edge too long (max 200px expected, got {top_edge:.2f})")
            
            # Check bottom edge (bottom-left to bottom-right)
            bottom_edge = distance(bottom_left, bottom_right)
            self.assertGreaterEqual(bottom_edge, 50, "Bottom edge too short (min 50px expected)")
            self.assertLessEqual(bottom_edge, 200, f"Bottom edge too long (max 200px expected, got {bottom_edge:.2f})")
            
            # Check that the table has reasonable proportions (not too narrow or wide)
            table_width = max(top_edge, bottom_edge)
            table_height = max(left_side, right_side)
            if table_height > 0:
                aspect_ratio = table_width / table_height
                self.assertGreaterEqual(aspect_ratio, 0.3, "Table appears too narrow")
                self.assertLessEqual(aspect_ratio, 3.0, "Table appears too wide")
            
            # Check that the table is roughly rectangular by comparing the lengths of opposite sides
            left_side = distance(top_left, bottom_left)
            right_side = distance(top_right, bottom_right)
            top_side = distance(top_left, top_right)
            bottom_side = distance(bottom_left, bottom_right)
            
            # Only perform ratio checks if sides are non-zero
            if right_side > 0 and bottom_side > 0:
                # Check ratio of left/right sides
                side_ratio = left_side / right_side
                self.assertGreaterEqual(side_ratio, 0.7, "Left side is too short compared to right side")
                self.assertLessEqual(side_ratio, 1.3, "Right side is too short compared to left side")
                
                # Check ratio of top/bottom sides
                top_bottom_ratio = top_side / bottom_side
                self.assertGreaterEqual(top_bottom_ratio, 0.8, "Top side is too short compared to bottom side")
                self.assertLessEqual(top_bottom_ratio, 1.2, "Bottom side is too short compared to top side")
                
                # Check that the table is not too skewed (diagonals should be roughly equal)
                diag1 = distance(top_left, bottom_right)
                diag2 = distance(top_right, bottom_left)
                diag_ratio = min(diag1, diag2) / max(diag1, diag2) if max(diag1, diag2) > 0 else 1.0
                self.assertGreaterEqual(diag_ratio, 0.85, "Table appears too skewed (diagonals too different)")
    
    def test_visualize_corners(self):
        """Test that visualization returns an image with correct dimensions"""
        for img in self.images:
            corners = detect_table_corners(img)
            vis = visualize_corners(img, corners)
            
            # Visualization should have same dimensions as input image
            self.assertEqual(vis.shape, img.shape)
            # Should be 3-channel (BGR) image
            self.assertEqual(len(vis.shape), 3)
    
    def test_find_quadrilateral_corners(self):
        """Test that quadrilateral corners are found correctly"""
        # Test with a perfect square
        points = [
            (0, 0), (100, 0), (100, 100), (0, 100),  # Square corners
            (50, 50), (50, 0), (0, 50), (100, 50), (50, 100)  # Points inside
        ]
        corners = find_quadrilateral_corners(points, (200, 200))
        self.assertIsNotNone(corners, "Should find corners for a square")
        self.assertEqual(corners.shape, (4, 2), "Should return 4 corners")
        
        # Test with not enough points
        with self.assertRaises(ValueError):
            find_quadrilateral_corners([(0,0), (1,1), (2,2)], (200, 200))
    
    def test_compute_intersection(self):
        """Test that line intersections are computed correctly"""
        # Two perpendicular lines
        line1 = (1, 0)  # x = 1
        line2 = (1, np.pi/2)  # y = 1
        point = compute_intersection(line1, line2)
        self.assertIsNotNone(point, "Should find intersection")
        x, y = point
        self.assertAlmostEqual(x, 1, delta=1e-6)
        self.assertAlmostEqual(y, 1, delta=1e-6)
        
        # Parallel lines should not intersect
        line3 = (2, 0)  # x = 2
        self.assertIsNone(compute_intersection(line1, line3), 
                         "Parallel lines should not intersect")
    
    def test_integration(self):
        """Test the full pipeline on all test images"""
        for i, img in enumerate(self.images):
            with self.subTest(image_idx=i):
                # Test the full pipeline with debug output
                corners = detect_table_corners(img, debug=True)
                self.assertIsNotNone(corners, f"No corners found in image {i}")
                
                # Check we got 4 corners
                self.assertEqual(corners.shape, (4, 2), 
                              f"Should find 4 corners in image {i}")
                
                # Test visualization
                vis = visualize_corners(img, corners)
                self.assertEqual(vis.shape, img.shape)
                
                # Save visualization for manual inspection
                output_dir = "test_output"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"test_output_{i}.jpg")
                cv2.imwrite(output_path, vis)
                print(f"Saved visualization to {output_path}")
                
                # Check that debug images were created
                debug_files = [f for f in os.listdir('debug') if f.startswith(('edge_', 'lines_', 'intersections', 'final_'))]
                self.assertGreater(len(debug_files), 0, "Debug images should be created")

if __name__ == "__main__":
    unittest.main()
