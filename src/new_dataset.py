import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from sklearn.preprocessing import LabelEncoder
from PIL import Image


class CIFAR10Dataset(
    Dataset,
):  # UNSUPERVISED 40000 train | 5000 val -- SUPERVISED 4500 train | 500 test
    def __init__(self, dataset_path: str, mode: str, split: str):
        if mode not in ("supervised", "unsupervised"):
            print("Need supervised or unsupervised mode !")

        self.df = pd.read_csv(os.path.join(dataset_path, "trainLabels.csv"))
        self.le = LabelEncoder()
        self.df["label_encoded"] = self.le.fit_transform(self.df["label"])

        self.nb_classes = len(self.le.classes_)

        self.data_path = os.path.join(dataset_path, "train")

        self.mode = mode
        if self.mode == "unsupervised":
            if split == "train":
                self.df = self.df[:40000]
            else:
                self.df = self.df[40000:45000]
        else:
            if split == "train":
                self.df = self.df[45000:49500]
            else:
                self.df = self.df[49500:]

    def __len__(self):
        """
        Returns the length of the dataset, which is the length of `df`.
        """
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, str(self.df.iloc[idx]["id"]) + ".png")
        img = pil_to_tensor(Image.open(img_path)).float() / 255

        if self.mode == "unsupervised":
            return img

        label = self.df.iloc[idx]["label_encoded"]
        target = torch.zeros(self.nb_classes)
        target[label] = 1.0
        return img, target
