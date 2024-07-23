# move images and split them into data folder
import json
import os
import shutil

def copy_images(json_file, source_dir, target_dir):
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    
    image_filenames = [entry['file_name'] for entry in data['images']]

    # makedir if not exist
    os.makedirs(target_dir, exist_ok=True)

    # move
    for filename in image_filenames:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        try:
            shutil.copy(source_path, target_path)
            print(f"Copied {filename} to {target_dir}")
        except FileNotFoundError:
            print(f"File {filename} not found in {source_dir}")

if __name__ == "__main__":
    
    train_json_file_path = "/home/wzhmiasanmia/hi_workspace/wheel_bolt_recognition/data/data_2/train.json"
    val_json_file_path = "/home/wzhmiasanmia/hi_workspace/wheel_bolt_recognition/data/data_2/val.json"
    image_source_directory = "/home/wzhmiasanmia/hi_workspace/wheel_bolt_recognition/images/images_2"
    train_directory = "/home/wzhmiasanmia/hi_workspace/wheel_bolt_recognition/data/data_2/train"
    val_directory = "/home/wzhmiasanmia/hi_workspace/wheel_bolt_recognition/data/data_2/val"
    
    copy_images(train_json_file_path, image_source_directory, train_directory)
    copy_images(val_json_file_path, image_source_directory, val_directory)
