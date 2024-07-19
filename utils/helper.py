import os, torch
from models import dehazer
from PIL import Image
from utils import config
import matplotlib.pyplot as plt


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def get_class_name_from_index(index, test_path):
    classes = sorted(os.listdir(test_path))
    return classes[index]


def load_model(model_path):
    model = dehazer.LightDehaze_Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return config.val_test_transform(image).unsqueeze(0)


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