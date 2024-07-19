"""
This module provides various utility functions for image processing, including image transformation,
PSNR and SSIM calculation, MSE calculation, and BRISQUE score calculation. It also includes a function
to ensure the existence of directories.

Functions:
    transform_and_save_image: Transforms and saves an image.
    resize_image: Resizes an image to a given size.
    calculate_psnr: Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    calculate_psnr_ssim: Calculates PSNR and Structural Similarity Index (SSIM) between two images.
    calculate_mse: Calculates the Mean Squared Error (MSE) between two images.
    calculate_brisque: Calculates the BRISQUE score for an image.
    calculate_brisque_for_folder: Calculates the BRISQUE scores for all images in a folder.
    ensure_directory_exists: Ensures that a directory exists.
"""

import os, cv2
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from brisque import BRISQUE
from PIL import Image
import numpy as np

def transform_and_save_image(gt_image_path, storage_folder, img_size):
    """
    Transforms and saves an image to a specified directory.

    Args:
        gt_image_path (str): Path to the input image.
        storage_folder (str): Directory where the transformed image will be saved.
        img_size (int): Size to which the image will be resized.

    Returns:
        str: Path to the saved transformed image.
    """
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
    """
    Resizes an image to the given size.

    Args:
        image (numpy.ndarray): Input image to be resized.
        size (tuple): Desired size as (width, height).

    Returns:
        numpy.ndarray: Resized image.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def calculate_psnr(imageA, imageB):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        imageA (numpy.ndarray): First input image.
        imageB (numpy.ndarray): Second input image.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((imageA - imageB) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_psnr_ssim(gt_image_path, hazy_image_path):
    """
    Calculates PSNR and Structural Similarity Index (SSIM) between two images.

    Args:
        gt_image_path (str): Path to the ground truth image.
        hazy_image_path (str): Path to the hazy image.

    Returns:
        tuple: PSNR and SSIM values.
    """
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
    """
    Calculates the Mean Squared Error (MSE) between two images.

    Args:
        gt_image_path (str): Path to the ground truth image.
        dehazed_image_path (str): Path to the dehazed image.

    Returns:
        float: MSE value.
    """
    gt_image = Image.open(gt_image_path).convert('RGB')
    dehazed_image = Image.open(dehazed_image_path).convert('RGB')
    gt_array = np.array(gt_image)
    dehazed_array = np.array(dehazed_image)
    mse = np.mean((gt_array - dehazed_array) ** 2)
    return mse

def calculate_brisque(image_path):
    """
    Calculates the BRISQUE score for an image.

    Args:
        image_path (str): Path to the image.

    Returns:
        float: BRISQUE score.
    """
    image = Image.open(image_path)
    image_np = np.array(image)
    obj = BRISQUE()
    score = obj.score(image_np)
    return score

def calculate_brisque_for_folder(folder):
    """
    Calculates the BRISQUE scores for all images in a folder.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        list: List of BRISQUE scores for each image in the folder.
    """
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
    """
    Ensures that a directory exists. If it does not exist, it creates it.

    Args:
        directory_path (str): Path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
