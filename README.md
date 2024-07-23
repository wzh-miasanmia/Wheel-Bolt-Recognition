# Wheel Bolt Recognition

Hello, welcome to the wheel_bolt_recognition project!

## Purpose

The primary objective of this project is to leverage mmdetection in developing a program capable of identifying the precise locations of screws in car tires. The ultimate goal is to streamline their subsequent manipulation by a robotic arm.

## Data Processing Workflow
Install all the libraries first :)
```bash
pip install -r requirements.txt
```

[Check out this notebook to know the pipeline](./train_model.ipynb)

We provide the following functionalities:

1. **Data Augmentation:**
    - Use the  `data_aug.py` file to enhance the original dataset by transforming a single image into 9 training-effective images.
    - Augmentation includes horizontal and vertical flips, 90/180/270-degree rotations, sharpening, shear, adding Gaussian noise, and adjusting brightness.

2. **Annotation using Labelme:**
    - Utilize the Labelme tool for target annotation, generating corresponding JSON files for each image.
    - For sharpening, noise addition, and brightness changes, maintain the annotation data and only modify the corresponding image names in the respective JSON files.


3. **Convert Annotation Format to COCO:**
    - Use the `convert` function in labelme2coco to transform labelme-formatted annotations into COCO format.
    ```python
    convert(labelme_folder, export_dir, train_split_rate)
    ```

4. **Dataset Splitting:**
    - Employ the `copy_images` function in split_data to divide the dataset. Manually reserve some images as the test set.
    - The labeled dataset will be split into training and testing sets based on the user-specified ratio.
    ```python
    copy_images(train_json_file_path, image_source_directory, train_directory)
    copy_images(val_json_file_path, image_source_directory, val_directory)
    ```

5. **Model Training:**
    - Train the model using hyperparameters specified in the chosen configuration file from the configs folder.
    - Start TensorBoard to monitor the training process.
    ```python
    python tools/train.py configs/your_config_file.py
    ```

6. **Load Trained Model for Inference:**
    - Load the trained checkpoint file and configuration file to initialize the inferencer.
    - Use the inferencer to predict on the test dataset.
    ```python
    inferencer = DetInferencer(config, checkpoint, device)
    inferencer(img_path, show=False, out_dir='outputs/', no_save_pred=False)
    ```

7. **Visualization of the test dataset:**
    - have a look at the results in the outputs/vis folder


## Results

![Test Result](outputs/vis/rgb_image_00000.png)