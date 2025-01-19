import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt

STL10_LABELS = {
    1: "airplane",
    2: "bird",
    3: "car",
    4: "cat",
    5: "deer",
    6: "dog",
    7: "horse",
    8: "monkey",
    9: "ship",
    10: "truck",
}


def read_stl10_images(file_path):
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
        num_images = data.size // (3 * 96 * 96)
        images = data.reshape(num_images, 3, 96, 96)
        images = images.transpose(0, 3, 2, 1)
        return images


def read_stl10_labels(file_path):
    with open(file_path, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def show_images(images, labels=None, num_images=3):
    images = images.permute(0, 2, 3, 1).numpy()
    print(images.shape)  # Convert to NumPy format (B, H, W, C)
    if labels is not None:
        labels = (
            labels.argmax(dim=1).numpy() + 1
        )  # Convert one-hot to class index (1-indexed for STL10_LABELS)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = images[i]  # Scale images to [0, 1] for proper display
        plt.imshow(np.clip(img, 0, 1))
        if labels is not None:
            plt.title(f"{STL10_LABELS[labels[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


labeled_images_path = "stl/stl10_binary/train_X.bin"  # TODO: Change with the path
labeled_labels_path = "stl/stl10_binary/train_y.bin"  # TODO: Change with the path
unlabeled_images_path = "stl/stl10_binary/unlabeled_X.bin"  # TODO: Change with the path


class STL10DatasetLabelled(Dataset):
    def __init__(self, path_images: str, path_labels: np.ndarray):
        self.images = read_stl10_images(path_images)
        self.labels = read_stl10_labels(path_labels)
        self.label_mapping = {
            1: "airplane",
            2: "bird",
            3: "car",
            4: "cat",
            5: "deer",
            6: "dog",
            7: "horse",
            8: "monkey",
            9: "ship",
            10: "truck",
        }

        self.nb_classes = len(self.label_mapping) if self.labels is not None else 0

    def __len__(self):
        """
        Returns the length of the dataset, which is the length of `df`.
        """
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float()

        img = img.permute(2, 0, 1)
        img = img / 255.0
        label = self.labels[idx] - 1
        target = torch.zeros(self.nb_classes)
        target[label] = 1.0
        return img, target


class STL10DatasetUnlabelled(Dataset):
    def __init__(self, path_images: str):
        self.images = read_stl10_images(path_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float()

        img = img.permute(2, 0, 1)
        img = img / 255.0  # 3 x 96 x 96
        return img


# dataset_pretrain = STL10DatasetLabelled(labeled_images_path, labeled_labels_path)
# dataset_downstream = STL10DatasetUnlabelled(unlabeled_images_path)
#
# labeled_loader = DataLoader(dataset_pretrain, batch_size=32, shuffle=True)
# unlabeled_loader = DataLoader(dataset_downstream, batch_size=32, shuffle=True)
#
# for images, labels in labeled_loader:
#    print(images[0].shape)
#    show_images(images=images, labels=labels, num_images=1)
#    break
#
#
# for images in unlabeled_loader:
#    print(images[0].shape)
#    show_images(images=images, labels=None, num_images=1)
#    break
#
