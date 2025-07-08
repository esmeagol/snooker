import cv2
import numpy as np
import argparse
from pathlib import Path

def select_points(image, num_points=4):
    """
    Select points on the image for perspective transformation.
    Points should be selected in this order:
    1. Top-left corner of the table
    2. Top-right corner of the table
    3. Bottom-right corner of the table
    4. Bottom-left corner of the table
    """
    points = []
    clone = image.copy()
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == num_points:
                cv2.line(clone, points[-1], points[0], (0, 255, 0), 2)
            cv2.imshow('Select Table Corners', clone)
    
    cv2.namedWindow('Select Table Corners')
    cv2.setMouseCallback('Select Table Corners', click_event)
    
    print(f'Please select {num_points} points in order (press any key when done)')
    print('Order: Top-left -> Top-right -> Bottom-right -> Bottom-left')
    
    while True:
        cv2.imshow('Select Table Corners', clone)
        key = cv2.waitKey(1) & 0xFF
        if key != 255 or len(points) >= num_points:
            break
    
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)

def process_video(input_path, output_path, src_points, output_size=(1000, 2000)):
    """
    Process the video and apply perspective transformation.
    """
    # Define destination points (top-down view)
    dst_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
    
    print("Processing video...")
    print("Press 'q' to quit processing.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(frame, M, output_size)
        
        # Write the frame to output video
        out.write(warped)
        
        # Display the frame
        cv2.imshow('Original', cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
        cv2.imshow('Top-Down View', cv2.resize(warped, (0,0), fx=0.5, fy=0.5))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Apply perspective transformation to snooker match video')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    args = parser.parse_args()
    
    # Load the first frame for point selection
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_video}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame from video.")
        return
    
    # Resize frame for easier point selection
    display_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    
    # Select points for perspective transformation
    print("Select the four corners of the snooker table in order:")
    print("1. Top-left\n2. Top-right\n3. Bottom-right\n4. Bottom-left")
    
    points = select_points(display_frame)
    
    if len(points) != 4:
        print("Error: Exactly 4 points must be selected.")
        return
    
    # Scale points back to original size
    points = points * 2
    
    # Process the video with 500x1000 resolution
    process_video(args.input_video, args.output, points, output_size=(500, 1000))

if __name__ == "__main__":
    main()
