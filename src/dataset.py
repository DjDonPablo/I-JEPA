import pandas as pd
import torch
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from typing import Tuple


class AffectNetDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        labels_filename: str,
        img_size: int,
        patch_size: int,
        nb_mask: int = 4,
        aspect_ratio_range_mask: Tuple[int, int] = (0.75, 1.5),
        scale_range_mask: Tuple[int, int] = (0.15, 0.2),
        aspect_ratio_range_context: Tuple[int, int] = (
            1.0,
            1.0,
        ),  # unit aspect ratio for context
        scale_range_context: Tuple[int, int] = (0.85, 1.0),
    ):
        self.df = pd.read_csv(os.path.join(dataset_path, labels_filename))
        self.dataset_path = dataset_path

        self.img_size = img_size
        self.patch_size = patch_size
        self.sqrt_count_patch = img_size // patch_size
        self.nb_mask = nb_mask

        self.aspect_ratio_range_mask = aspect_ratio_range_mask
        self.scale_range_mask = scale_range_mask

        self.aspect_ratio_range_context = aspect_ratio_range_context
        self.scale_range_context = scale_range_context

    def __len__(self):
        """
        Returns the length of the dataset, which is the length of `df`.
        """
        return len(self.df)

    def get_random_width_and_height(self, n: int):
        """
        Returns a random height and weight based on a random aspect ratio and a random scale.\\
        If `n` equals 1, it is deduced that the random generation is for context block, so it uses the aspect ratio and scale range specific to the context.\\
        Else, it is deduced that the random generation is for mask blocks, so it uses the aspect ratio and scale range specific to the masks. 

        Parameters:
        - `n`: the number of height and width to be generated. 
        """
        if n == 1:  # 1 generation => for context
            aspect_ratio = torch.FloatTensor(n).uniform_(
                *self.aspect_ratio_range_context
            )
            scale = torch.FloatTensor(n).uniform_(*self.scale_range_context)
        else:
            aspect_ratio = torch.FloatTensor(n).uniform_(*self.aspect_ratio_range_mask)
            scale = torch.FloatTensor(n).uniform_(*self.scale_range_mask)

        # calculate area, width and height of each mask
        area = scale * (self.sqrt_count_patch**2)
        height = torch.sqrt(area / aspect_ratio).round().int()
        width = torch.sqrt(area * aspect_ratio).round().int()

        return height, width

    def __get_item__(self, idx):
        """
        Returns a tuple of 3 elements:
        -
        -
        -
        """
        img_path = os.path.join(self.dataset_path, self.df.iloc[idx]["pth"])
        img = pil_to_tensor(Image.open(img_path))  # 96 x 96 x 3

        # random height and width for masks
        height, width = self.get_random_width_and_height(self.nb_mask)

        # calculate patch indexes of mask
        masks = []
        masks_indexes = set()
        x = np.random.randint(low=self.sqrt_count_patch - width.int() + 1)
        y = np.random.randint(low=self.sqrt_count_patch - height.int() + 1)
        for i in range(self.nb_mask):
            tmp_mask = torch.cat(
                [
                    torch.arange(x, x + width[i]) + (y + i) * self.sqrt_count_patch
                    for i in range(height[i])
                ]
            )

            masks_indexes.update(tmp_mask.tolist())
            masks.append(tmp_mask)

        # random height and width for context
        height, width = self.get_random_width_and_height(1)
        x = np.random.randint(low=self.sqrt_count_patch - int(width[0].item()) + 1)
        y = np.random.randint(low=self.sqrt_count_patch - int(height[0].item()) + 1)

        context_indexes = set(
            torch.cat(
                [
                    torch.arange(x, x + width[i]) + (y + i) * self.sqrt_count_patch
                    for i in range(height[i])
                ]
            ).tolist()
        )

        return (
            img.view(  # 144 x 8 x 8 x 3
                self.sqrt_count_patch**2,
                self.patch_size,
                self.patch_size,
                3,
            ),
            context_indexes - masks_indexes,
            masks,
        )


# dataset = AffectNetDataset(
#     dataset_path=os.path.join("dataset", "archive"),
#     labels_filename="labels.csv",
#     img_size=96,
#     patch_size=8,
# )
