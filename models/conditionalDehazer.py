import os, torch
import numpy as np
from utils import metrics, config
from PIL import Image
from torchvision import transforms
from models import classifier, dehazer
from utils.helper import ensure_directory_exists, get_class_name_from_index, load_model, preprocess_image, visualize_images


def conditionalDehazing(gt_image, hazy_image, dehazers, classifier, output_dir, output_dir_folder="dehazed"):
    """
    Perform conditional dehazing on a single image or a batch of images.

    Args:
        gt_image (str): Path to the ground truth (GT) image or folder.
        hazy_image (str): Path to the hazy image or folder.
        dehazers (list): List of paths to the dehazer models.
        classifier (str): Path to the classifier model.
        output_dir (str): Directory to save dehazed images.
        output_dir_folder (str): Subfolder in output directory to save dehazed images. Default is "dehazed".
    """
    if os.path.isfile(hazy_image):
        predicted_class, _, _ = classification_inference(classifier, hazy_image)
        dehaze_inference(dehazers, gt_image, hazy_image, predicted_class_name=predicted_class, output_dir=output_dir,
                         output_dir_folder=output_dir_folder)
    elif os.path.isdir(hazy_image):
        batch_dehaze_and_evaluate(dehazers, gt_image, hazy_image, classifier, output_dir, output_dir_folder)
    else:
        print('Version 2 can only inference on Single Image or Directory. Please provide a valid path.')



def classification_inference(classifier_weight, image_path, test_path=config.test_path_for_class_name,
                             transform=config.val_test_transform):
    """
    Perform classification inference to predict the haze type in an image.

    Args:
        classifier_weight (str): Path to the classifier model.
        image_path (str): Path to the image to be classified.
        test_path (str): Path to the test data for class names.
        transform (callable): Transformation to be applied to the image.

    Returns:
        tuple: Predicted class name, actual class name, and predicted probability.
    """
    model = classifier.ResNet152()
    model.load_state_dict(torch.load(classifier_weight, map_location=config.device))
    model.to(config.device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(config.device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_probabilities = torch.nn.functional.softmax(outputs, dim=1)

    predicted_probability, _ = predicted_probabilities.max(1)
    actual_class_name = os.path.basename(os.path.dirname(image_path))
    predicted_class_name = get_class_name_from_index(predicted_idx, test_path)

    print(f"Predicted class: {predicted_class_name}")
    print(f"Predicted probability: {predicted_probability.item()}")

    return predicted_class_name, actual_class_name, predicted_probability.item()



def dehaze_inference(dehazer_model_names, gt_image, hazy_image, predicted_class_name, output_dir, output_dir_folder):
    """
    Perform dehazing inference on a single image.

    Args:
        dehazer_model_names (list): List of paths to the dehazer models.
        gt_image (str): Path to the ground truth (GT) image.
        hazy_image (str): Path to the hazy image.
        predicted_class_name (str): Predicted class name from the classifier.
        output_dir (str): Directory to save dehazed images.
        output_dir_folder (str): Subfolder in output directory to save dehazed images.
    """
    hazy_image_path = hazy_image
    dehazer_model_path = get_dehazer_model_path(dehazer_model_names, predicted_class_name)

    model = load_model(dehazer_model_path)
    image_tensor = preprocess_image(hazy_image)

    with torch.no_grad():
        output_tensor = model(image_tensor)

    transform = transforms.ToPILImage()
    hazy_image = transform(image_tensor.squeeze(0))
    dehazed_image = transform(output_tensor.squeeze(0))

    if gt_image:
        gt_image_pil = Image.open(gt_image).convert('RGB')
    else:
        gt_image_pil = None

    visualize_images(hazy_image, dehazed_image, gt_image_pil, predicted_class_name)

    dehazed_image_path = save_dehazed_image(hazy_image_path, dehazed_image, output_dir, output_dir_folder)

    if gt_image:
        evaluate_images(gt_image, hazy_image_path, dehazed_image_path, output_dir)
    else:
        brisque_value = metrics.calculate_brisque(dehazed_image_path)
        print(f"BRISQUE for {hazy_image_path}: {brisque_value}")



def batch_dehaze_and_evaluate(dehazer_model_names, gt_folder, hazy_folder, classifier_weight, output_dir,
                              output_dir_folder):
    """
    Perform batch dehazing and evaluation on a directory of images.

    Args:
        dehazer_model_names (list): List of paths to the dehazer models.
        gt_folder (str): Path to the folder containing ground truth (GT) images.
        hazy_folder (str): Path to the folder containing hazy images.
        classifier_weight (str): Path to the classifier model.
        output_dir (str): Directory to save dehazed images.
        output_dir_folder (str): Subfolder in output directory to save dehazed images.
    """
    psnr_values, ssim_values, mse_values, brisque_values = [], [], [], []

    for hazy_image_filename in os.listdir(hazy_folder):
        hazy_image_path = os.path.join(hazy_folder, hazy_image_filename)
        gt_image_path = os.path.join(gt_folder, hazy_image_filename) if gt_folder else None

        predicted_class, _, _ = classification_inference(classifier_weight, hazy_image_path)
        dehazer_model_path = get_dehazer_model_path(dehazer_model_names, predicted_class)
        model = load_model(dehazer_model_path)

        hazy_image_tensor = preprocess_image(hazy_image_path)
        with torch.no_grad():
            dehazed_image_tensor = model(hazy_image_tensor)

        transform = transforms.ToPILImage()
        dehazed_image = transform(dehazed_image_tensor.squeeze(0))
        dehazed_image_path = save_dehazed_image(hazy_image_path, dehazed_image, output_dir, output_dir_folder)

        if gt_image_path and os.path.isfile(gt_image_path):
            ensure_directory_exists(os.path.join(output_dir, "gt"))
            resize_gt_image = metrics.transform_and_save_image(gt_image_path, os.path.join(output_dir, "gt"), 512)

            mse = metrics.calculate_mse(resize_gt_image, dehazed_image_path)
            psnr, ssim = metrics.calculate_psnr_ssim(resize_gt_image, dehazed_image_path)

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)

            print(f"PSNR: {psnr} | SSIM: {ssim} | MSE: {mse}")
        else:
            brisque_value = metrics.calculate_brisque(dehazed_image_path)
            brisque_values.append(brisque_value)
            print(f"BRISQUE for {hazy_image_path}: {brisque_value}")

    if psnr_values and ssim_values and mse_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_mse = np.mean(mse_values)
        print(f"Average PSNR: {avg_psnr} | Average SSIM: {avg_ssim} | Average MSE: {avg_mse}")

    if brisque_values:
        avg_brisque = np.mean(brisque_values)
        print(f"Average BRISQUE: {avg_brisque}")



def get_dehazer_model_path(dehazer_model_names, predicted_class_name):
    """
    Get the path to the dehazer model based on the predicted class name.

    Args:
        dehazer_model_names (list): List of paths to the dehazer models.
        predicted_class_name (str): Predicted class name from the classifier.

    Returns:
        str: Path to the dehazer model.
    """
    if predicted_class_name == 'Cloud':
        return dehazer_model_names[0]
    elif predicted_class_name == 'EH':
        return dehazer_model_names[1]
    elif predicted_class_name == 'Fog':
        return dehazer_model_names[2]
    else:
        raise ValueError(f'Invalid predicted class name: {predicted_class_name}')



def save_dehazed_image(hazy_image_path, dehazed_image, output_dir, output_dir_folder):
    """
    Save the dehazed image to the specified directory.

    Args:
        hazy_image_path (str): Path to the hazy image.
        dehazed_image (PIL.Image.Image): Dehazed image.
        output_dir (str): Directory to save dehazed images.
        output_dir_folder (str): Subfolder in output directory to save dehazed images.

    Returns:
        str: Path where the dehazed image is saved.
    """
    ensure_directory_exists(output_dir)
    ensure_directory_exists(os.path.join(output_dir, output_dir_folder))
    dehazed_image_filename = os.path.basename(hazy_image_path).split('.')[0] + "_dehazed.jpg"
    dehazed_image_path = os.path.join(output_dir, output_dir_folder, dehazed_image_filename)
    dehazed_image.save(dehazed_image_path)
    return dehazed_image_path



def evaluate_images(gt_image_path, hazy_image_path, dehazed_image_path, output_dir):
    """
    Evaluate the dehazed image against the ground truth image.

    Args:
        gt_image_path (str): Path to the ground truth (GT) image.
        hazy_image_path (str): Path to the hazy image.
        dehazed_image_path (str): Path to the dehazed image.
        output_dir (str): Directory to save the evaluation results.
    """
    psnr, ssim = metrics.calculate_psnr_ssim(gt_image_path, hazy_image_path)
    print(f'GT VS Dehazed Image | PSNR: {psnr} | SSIM of : {ssim}')

    img_size = 512
    ensure_directory_exists(os.path.join(output_dir, "gt"))
    resize_gt_image = metrics.transform_and_save_image(gt_image_path, os.path.join(output_dir, "gt"), img_size)
    psnr, ssim = metrics.calculate_psnr_ssim(resize_gt_image, dehazed_image_path)
    print(f'GT VS Dehazed Image (resized to {img_size}) | PSNR: {psnr} | SSIM of : {ssim}')
