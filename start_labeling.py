import os
import subprocess
import json

# Create a label map file for YOLO format
def create_label_map():
    label_map = {
        "cue_ball": 0,
        "red_ball": 1,
        "yellow_ball": 2,
        "green_ball": 3,
        "brown_ball": 4,
        "blue_ball": 5,
        "pink_ball": 6,
        "black_ball": 7,
        "white_ball": 8,
        "player": 9
    }
    
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4)
    
    return label_map

def main():
    # Create necessary directories
    os.makedirs('data/annotated_frames', exist_ok=True)
    
    # Create label map
    label_map = create_label_map()
    
    print("Label Map Created:")
    for label, idx in label_map.items():
        print(f"  {label}: {idx}")
    
    print("\nStarting labelme...")
    print(f"Frames directory: {os.path.abspath('data/raw_frames')}")
    print(f"Annotations will be saved to: {os.path.abspath('data/annotated_frames')}")
    print("\nLabeling Instructions:")
    print("1. Click 'Open Dir' and select the 'data/raw_frames' directory")
    print("2. Click 'Change Save Dir' and select 'data/annotated_frames'")
    print("3. Use 'Create Rectangle' to draw bounding boxes")
    print("4. Select the correct label for each box")
    print("5. Save with Ctrl+S")
    print("6. Use 'd' for next image, 'a' for previous")
    
    # Start labelme
    subprocess.run(["labelme", "--nodata", "--autosave"])

if __name__ == "__main__":
    main()
