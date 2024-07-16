import cv2
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from brisque import BRISQUE
from PIL import Image
import torch
import os
import numpy as np

def transform_and_save_image(gt_image_path, storage_folder, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    pil_image = Image.open(gt_image_path)
    transformed_tensor = transform(pil_image)
    transformed_image = transforms.ToPILImage()(transformed_tensor)
    file_name = os.path.basename(gt_image_path)
    save_path = os.path.join(storage_folder, file_name)
    ensure_directory_exists(storage_folder)
    transformed_image.save(save_path, format='JPEG')
    return save_path

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def calculate_psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_psnr_ssim(gt_image_path, hazy_image_path):
    imageA = cv2.imread(gt_image_path)
    imageB = cv2.imread(hazy_image_path)
    height, width = imageA.shape[:2]
    imageB = resize_image(imageB, (width, height))
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    psnr_value = calculate_psnr(imageA_gray, imageB_gray)
    ssim_value = ssim(imageA_gray, imageB_gray)
    return psnr_value, ssim_value

def calculate_mse(gt_image_path, dehazed_image_path):
    gt_image = Image.open(gt_image_path).convert('RGB')
    dehazed_image = Image.open(dehazed_image_path).convert('RGB')
    gt_array = np.array(gt_image)
    dehazed_array = np.array(dehazed_image)
    mse = np.mean((gt_array - dehazed_array) ** 2)
    return mse

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
        except Exception as e:
            print(f"Skipping {img_name} due to error: {e}")
    return brisque_values

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
