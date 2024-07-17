import utils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def test_loader_function(test_path, transform=utils.val_test_transform, batch_size=utils.batch_size):
    """
    Creates a DataLoader for the test dataset.

    Args:
        test_path (str): Path to the directory containing the test images.
        transform (callable, optional): Transformation to be applied to the images. Defaults to utils.val_test_transform.
        batch_size (int, optional): Number of images per batch. Defaults to utils.batch_size.

    Returns:
        DataLoader: DataLoader for the test dataset.
        dict: Mapping from class name to class index.
        list: List of class names.
    """
    # Create the ImageFolder dataset
    test_dataset = ImageFolder(root=test_path, transform=transform)

    # Create the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract the class_to_idx mapping and class names from the dataset
    class_to_idx = test_dataset.class_to_idx
    class_names = test_dataset.classes

    return test_loader, class_to_idx, class_names
