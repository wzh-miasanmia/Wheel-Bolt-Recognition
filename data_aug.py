from PIL import Image, ImageEnhance
import numpy as np
import random
import os
import shutil

def flip_image(img, direction):
    if direction == 'vertical':
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif direction == 'horizontal':
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img

def rotate_image(img, angle):
    return img.rotate(angle)

def shear_image(img, shear_factor):
    width, height = img.size
    shear_matrix = [1, shear_factor, 0, shear_factor, 1, 0]
    return img.transform((width, height), Image.AFFINE, shear_matrix)

def crop_image(img, crop_factor):
    width, height = img.size
    left = random.randint(0, int(width * crop_factor))
    top = random.randint(0, int(height * crop_factor))
    right = random.randint(int((1 - crop_factor) * width), width)
    bottom = random.randint(int((1 - crop_factor) * height), height)
    return img.crop((left, top, right, bottom))

def sharpen_image(img, factor):
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def add_gaussian_noise(img, mean=0, std=25):
    img_array = np.array(img)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img = Image.fromarray(np.uint8(img_array + noise))
    return noisy_img

def adjust_brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def data_augmentation(original_images_path, image_path, save_path):
    original_image = Image.open(os.path.join(original_images_path, image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Apply flipping
    augmented_image = flip_image(original_image, 'vertical')
    save_and_show(augmented_image, save_path, f"{image_name}_flip_vertical")

    augmented_image = flip_image(original_image, 'horizontal')
    save_and_show(augmented_image, save_path, f"{image_name}_flip_horizontal")

    # Apply rotating
    augmented_image = original_image.copy()
    augmented_image = rotate_image(augmented_image, 90)
    save_and_show(augmented_image, save_path, f"{image_name}_rotate_90")

    augmented_image = original_image.copy()
    augmented_image = rotate_image(augmented_image, 180)
    save_and_show(augmented_image, save_path, f"{image_name}_rotate_180")

    augmented_image = rotate_image(original_image, 270)
    save_and_show(augmented_image, save_path, f"{image_name}_rotate_270")

    # Apply shearing
    augmented_image = shear_image(original_image, random.uniform(-0.2, 0.2))
    save_and_show(augmented_image, save_path, f"{image_name}_shear")

    # Apply cropping
    augmented_image = crop_image(original_image, random.uniform(0.1, 0.3))
    save_and_show(augmented_image, save_path, f"{image_name}_crop")

    # Apply sharpening
    augmented_image = sharpen_image(original_image, random.uniform(0.5, 2.0))
    save_and_show(augmented_image, save_path, f"{image_name}_sharpen")

    # Apply adding Gaussian noise
    augmented_image = add_gaussian_noise(original_image)
    save_and_show(augmented_image, save_path, f"{image_name}_gaussian_noise")

    # Apply adjusting brightness
    augmented_image = adjust_brightness(original_image, random.uniform(0.5, 2.0))
    save_and_show(augmented_image, save_path, f"{image_name}_brightness")

    
def save_and_show(img, save_path, save_name):
    save_name_with_extension = f"{save_name}.jpg"
    save_path_with_extension = os.path.join(save_path, save_name_with_extension)
    img.save(save_path_with_extension)
    print(f"Saved: {save_path_with_extension}")
    
    
def copy_labels(original_images_path,label_path, save_path):
    # Copy json files when augmentation does not change the labels
    label_name= os.path.splitext(os.path.basename(label_path))[0]
    nochangelist = ["_gaussian_noise", "_brightness", "_sharpen"]

    for change in nochangelist:
        new_label_name = f"{label_name}{change}.json"
        new_label_path = os.path.join(save_path, new_label_name)
        # Copy the JSON file
        shutil.copyfile(os.path.join(original_images_path, label_path), new_label_path)
        print(f"Saved: {new_label_path}")
if __name__ == "__main__":
    original_images_path = "./original_images"
    save_path = "./images"
    os.makedirs(save_path, exist_ok=True)
    for file_path in os.listdir(original_images_path):
        if file_path.endswith(".jpg"):
            image_path = file_path
            data_augmentation(original_images_path, image_path, save_path)
        if file_path.endswith(".json"):
            label_path = file_path
            copy_labels(original_images_path, label_path, save_path)