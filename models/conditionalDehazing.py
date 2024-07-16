import os
import numpy as np
from utils import metrics, config
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from models import classifier, dehazer

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def TTCDehazeNet(gt_image, hazy_image, dehazers, classifier, output_dir):
    if os.path.isfile(hazy_image):
        predicted_class, _, _ = classification_inference(classifier, hazy_image)
        dehaze_inference(dehazers, gt_image, hazy_image, predicted_class_name=predicted_class, output_dir=output_dir)
    elif os.path.isdir(hazy_image):
        batch_dehaze_and_evaluate(dehazers, gt_image, hazy_image, classifier, output_dir)
    else:
        print('Version 2 can only inference on Single Image or Directory. Please provide a valid path.')

def get_class_name_from_index(index, test_path):
    classes = sorted(os.listdir(test_path))
    return classes[index]

def classification_inference(classifier_weight, image_path, test_path=config.test_path_for_class_name, transform=config.val_test_transform):
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

    print(f"Actual class: {actual_class_name}")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Predicted probability: {predicted_probability.item()}")

    return predicted_class_name, actual_class_name, predicted_probability.item()

def load_model(model_path):
    model = dehazer.LightDehaze_Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return config.val_test_transform(image).unsqueeze(0)

def dehaze_inference(dehazer_model_names, gt_image, hazy_image, predicted_class_name, output_dir):
    gt_image_path = gt_image
    hazy_image_path = hazy_image
    dehazer_model_path = get_dehazer_model_path(dehazer_model_names, predicted_class_name)

    model = load_model(dehazer_model_path)
    image_tensor = preprocess_image(hazy_image)

    with torch.no_grad():
        output_tensor = model(image_tensor)

    transform = transforms.ToPILImage()
    hazy_image = transform(image_tensor.squeeze(0))
    dehazed_image = transform(output_tensor.squeeze(0))

    ncols = 3 if gt_image else 2
    visualize_images(hazy_image, dehazed_image, gt_image, predicted_class_name)

    dehazed_image_path = save_dehazed_image(hazy_image_path, dehazed_image, output_dir)

    if gt_image:
        evaluate_images(gt_image_path, hazy_image_path, dehazed_image_path, output_dir)

def batch_dehaze_and_evaluate(dehazer_model_names, gt_folder, hazy_folder, classifier_weight, output_dir):
    psnr_values, ssim_values, mse_values = [], [], []

    for hazy_image_filename in os.listdir(hazy_folder):
        hazy_image_path = os.path.join(hazy_folder, hazy_image_filename)
        gt_image_path = os.path.join(gt_folder, hazy_image_filename)

        predicted_class, _, _ = classification_inference(classifier_weight, hazy_image_path)
        dehazer_model_path = get_dehazer_model_path(dehazer_model_names, predicted_class)
        model = load_model(dehazer_model_path)

        hazy_image_tensor = preprocess_image(hazy_image_path)
        with torch.no_grad():
            dehazed_image_tensor = model(hazy_image_tensor)

        transform = transforms.ToPILImage()
        dehazed_image = transform(dehazed_image_tensor.squeeze(0))
        dehazed_image_path = save_dehazed_image(hazy_image_path, dehazed_image, output_dir)

        if os.path.isfile(gt_image_path):
            ensure_directory_exists(os.path.join(output_dir, "GT"))
            resize_gt_image = metrics.transform_and_save_image(gt_image_path, os.path.join(output_dir, "GT"), 512)

            mse = metrics.calculate_mse(resize_gt_image, dehazed_image_path)
            psnr, ssim = metrics.calculate_psnr_ssim(resize_gt_image, dehazed_image_path)

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)

            print(f"PSNR: {psnr} | SSIM: {ssim} | MSE: {mse}")

    if psnr_values and ssim_values and mse_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_mse = np.mean(mse_values)

        print(f"Average PSNR: {avg_psnr} | Average SSIM: {avg_ssim} | Average MSE: {avg_mse}")

        return avg_psnr, avg_ssim, avg_mse
    else:
        print("No GT images found for evaluation.")
        return None, None, None

def get_dehazer_model_path(dehazer_model_names, predicted_class_name):
    if predicted_class_name == 'Cloud':
        return dehazer_model_names[0]
    elif predicted_class_name == 'EH':
        return dehazer_model_names[1]
    elif predicted_class_name == 'Fog':
        return dehazer_model_names[2]
    else:
        raise ValueError(f'Invalid predicted class name: {predicted_class_name}')

def save_dehazed_image(hazy_image_path, dehazed_image, output_dir):
    ensure_directory_exists(output_dir)
    ensure_directory_exists(os.path.join(output_dir, "Haze"))
    dehazed_image_filename = os.path.basename(hazy_image_path).split('.')[0] + "_dehazed.jpg"
    dehazed_image_path = os.path.join(output_dir, "Haze", dehazed_image_filename)
    dehazed_image.save(dehazed_image_path)
    return dehazed_image_path

def visualize_images(hazy_image, dehazed_image, gt_image, predicted_class_name):
    ncols = 3 if gt_image else 2

    fig, ax = plt.subplots(1, ncols, figsize=(15, 6))
    ax[0].imshow(hazy_image)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(dehazed_image)
    ax[1].set_title('Dehazed Image | Predicted Class: ' + predicted_class_name)
    ax[1].axis('off')

    if gt_image:
        ax[2].imshow(gt_image)
        ax[2].set_title('Ground Truth Image')
        ax[2].axis('off')

    plt.show()

def evaluate_images(gt_image_path, hazy_image_path, dehazed_image_path, output_dir):
    psnr, ssim = metrics.calculate_psnr_ssim(gt_image_path, gt_image_path)
    print(f'GT VS GT Image | PSNR: {psnr} | SSIM of : {ssim}')

    psnr, ssim = metrics.calculate_psnr_ssim(gt_image_path, hazy_image_path)
    print(f'GT VS Dehazed Image | PSNR: {psnr} | SSIM of : {ssim}')

    ensure_directory_exists(os.path.join(output_dir, "Dehazed"))
    resize_gt_image = metrics.transform_and_save_image(gt_image_path, os.path.join(output_dir, "Dehazed"), 512)
    psnr, ssim = metrics.calculate_psnr_ssim(resize_gt_image, dehazed_image_path)
    print(f'GT VS Dehazed Image | PSNR: {psnr} | SSIM of : {ssim}')
