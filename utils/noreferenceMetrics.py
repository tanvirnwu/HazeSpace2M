import os
import numpy as np
from brisque import BRISQUE
from PIL import Image


def calculate_brisque(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    obj = BRISQUE()
    score = obj.score(image_np)
    return score


def calculate_brisque_for_folder(folder):
    brisque_values = []
    images = sorted(os.listdir(folder))

    for img_name in images:
        img_path = os.path.join(folder, img_name)

        if not os.path.isfile(img_path):
            print(f"File missing for {img_name}, skipping.")
            continue

        try:
            brisque_value = calculate_brisque(img_path)
            brisque_values.append(brisque_value)
            # print(f"BRISQUE for {img_name}: {brisque_value}")
        except Exception as e:
            print(f"Skipping {img_name} due to error: {e}")

    return brisque_values