import json
from PIL import Image, ImageDraw, ImageFont

# CORRECTED JSON data for the 1920x1080 image
JSON_DATA = """
[
  {"box_2d": [187, 219, 227, 246], "label": "Red"},
  {"box_2d": [702, 396, 738, 419], "label": "Red"},
  {"box_2d": [647, 368, 683, 391], "label": "Red"},
  {"box_2d": [748, 742, 781, 762], "label": "Red"},
  {"box_2d": [662, 734, 696, 755], "label": "Red"},
  {"box_2d": [698, 734, 731, 755], "label": "Red"},
  {"box_2d": [516, 569, 549, 586], "label": "Red"},
  {"box_2d": [556, 523, 589, 541], "label": "Red"},
  {"box_2d": [521, 509, 552, 525], "label": "Red"},
  {"box_2d": [587, 523, 621, 543], "label": "Red"},
  {"box_2d": [580, 507, 614, 526], "label": "Red"},
  {"box_2d": [569, 451, 601, 469], "label": "Red"},
  {"box_2d": [545, 471, 578, 489], "label": "Red"},
  {"box_2d": [576, 471, 606, 490], "label": "Red"},
  {"box_2d": [531, 459, 559, 477], "label": "Red"},
  {"box_2d": [545, 439, 577, 457], "label": "Red"},
  {"box_2d": [520, 370, 552, 388], "label": "Red"},
  {"box_2d": [638, 559, 672, 578], "label": "Red"},
  {"box_2d": [201, 424, 233, 444], "label": "Yellow"},
  {"box_2d": [617, 442, 654, 464], "label": "Green"},
  {"box_2d": [201, 541, 228, 558], "label": "Blue"},
  {"box_2d": [342, 482, 369, 500], "label": "Blue"},
  {"box_2d": [506, 479, 537, 498], "label": "Pink"},
  {"box_2d": [742, 450, 782, 475], "label": "Black"},
  {"box_2d": [679, 107, 920, 273], "label": "Bottom-left"},
  {"box_2d": [698, 725, 915, 893], "label": "Bottom-right"},
  {"box_2d": [64, 534, 154, 629], "label": "middle-right"},
  {"box_2d": [62, 375, 148, 469], "label": "middle-left"},
  {"box_2d": [68, 868, 171, 990], "label": "Top-right"},
  {"box_2d": [67, 7, 169, 134], "label": "Top-left"}
]
"""

# Load the data from the JSON string
data = json.loads(JSON_DATA)

# Open the original image
image_path = '/Users/abhinavrai/Playground/snooker/data/raw_frames/ROS-Frame-1_frame_004470.jpg'
try:
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    exit()

# Define font for the labels (you may need to change the font path)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except IOError:
    print("Arial font not found, using default font.")
    font = ImageFont.load_default()

# Loop through the data and draw the labels and boxes
for item in data:
    box = item['box_2d']
    label = item['label']

    # Draw the bounding box rectangle
    draw.rectangle(box, outline="yellow", width=2)
    
    # Position for the text label, slightly above the box
    text_position = (box[0] + 4, box[1] - 22)

    # Draw a small background for the text for better visibility
    text_bbox = draw.textbbox(text_position, label, font=font)
    draw.rectangle(text_bbox, fill="black")
    
    # Draw the text label
    draw.text(text_position, label, fill="yellow", font=font)

# Save the new, annotated image
image.save('snooker_labeled.png')

print("Successfully created 'snooker_labeled.png' with all annotations!")
