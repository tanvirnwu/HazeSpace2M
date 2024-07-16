import utils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def test_loader_function(test_path, transform= utils.val_test_transform, batch_size= utils.batch_size):
    # Create the ImageFolder dataset
    test_dataset = ImageFolder(root=test_path, transform=transform)

    # Create the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract the class_to_idx mapping and class names from the dataset
    class_to_idx = test_dataset.class_to_idx
    class_names = test_dataset.classes

    return test_loader, class_to_idx, class_names
