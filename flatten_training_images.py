import os
import shutil

data_dir = "/Users/abhinavrai/Playground/snooker_data/step_2_training_data"

# Gather all jpg files from subfolders
def get_all_images(root):
    images = []
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.jpg'):
                images.append(os.path.join(subdir, f))
    return sorted(images)

def main():
    images = get_all_images(data_dir)
    # Remove subfolders except .placeholder
    for entry in os.listdir(data_dir):
        p = os.path.join(data_dir, entry)
        if os.path.isdir(p):
            shutil.rmtree(p)
    # Move and rename images
    for idx, img_path in enumerate(images):
        ext = os.path.splitext(img_path)[1]
        new_name = f"{idx:03d}{ext}"
        new_path = os.path.join(data_dir, new_name)
        shutil.move(img_path, new_path)
    print(f"Moved {len(images)} images to {data_dir} as flat 000.jpg, 001.jpg, ...")

if __name__ == "__main__":
    main()
