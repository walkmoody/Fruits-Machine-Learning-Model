from PIL import Image
import os

data_dir = r'C:\Fruits\Fruit'
valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if not fname.lower().endswith(valid_extensions):
                print(f"Invalid file: {fpath}")
            else:
                try:
                    Image.open(fpath).verify()  # checks for corruption
                except:
                    print(f"Corrupted image: {fpath}")
