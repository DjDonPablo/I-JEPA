import pandas as pd
import torch
import os

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image


class JEPADataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        labels_filename: str,
    ):
        self.df = pd.read_csv(os.path.join(dataset_path, labels_filename))
        self.le = LabelEncoder()
        self.df["label_encoded"] = self.le.fit_transform(self.df["label"])

        self.nb_classes = len(self.le.classes_)

        self.dataset_path = dataset_path

    def __len__(self):
        """
        Returns the length of the dataset, which is the length of `df`.
        """
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.df.iloc[idx]["pth"])
        label = self.df.iloc[idx]["label_encoded"]
        img = pil_to_tensor(Image.open(img_path)).float() / 255  # 3 x 96 x 96

        target = torch.zeros(self.nb_classes)
        target[label] = 1.0
        return img  # , target
