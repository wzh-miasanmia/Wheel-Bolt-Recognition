import logging
import os
from pathlib import Path

from sahi.utils.file import save_json
from get_coco_from_labelme_folder import get_coco_from_labelme_folder
from typing import List

def convert(
    labelme_folder: str,
    export_dir: str = "runs/labelme2coco/",
    train_split_rate: float = 1,
    skip_labels: List[str] = [],
):
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        export_dir: path for coco jsons to be exported
        train_split_rate: ration fo train split
    """
    coco = get_coco_from_labelme_folder(labelme_folder, skip_labels=skip_labels)
    if train_split_rate < 1:
        result = coco.split_coco_as_train_val(train_split_rate)
        # export train split
        save_path = str(Path(export_dir) / "train.json")
        save_json(result["train_coco"].json, save_path)
        # export val split
        save_path = str(Path(export_dir) / "val.json")
        save_json(result["val_coco"].json, save_path)
    else:
        save_path = str(Path(export_dir) / "dataset.json")
        save_json(coco.json, save_path)

if __name__ == "__main__":
    labelme_folder = "/home/wzhmiasanmia/hi_workspace/wheel_bolt_recognition/images"
    export_dir = "/home/wzhmiasanmia/hi_workspace/wheel_bolt_recognition/data/"
    train_split_rate = 0.75
    convert(labelme_folder, export_dir, train_split_rate)