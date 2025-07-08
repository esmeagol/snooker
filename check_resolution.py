import cv2

# Open the video file
video_path = "/Users/abhinavrai/Playground/snooker/videos/champion-snooker-frame-1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps > 0 else 0

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Duration: {duration:.2f} seconds")
print(f"Frame count: {frame_count}")

# Release the video capture object
cap.release()
