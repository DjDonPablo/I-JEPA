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
        Length of height and width is `n+1`, the first `n` elements being masks height and width, and the last one being the context height and weight.

        Parameters:
        - `n`: the number of height and width to be generated. 
        """
        aspect_ratio = torch.FloatTensor(n).uniform_(*self.aspect_ratio_range_mask)
        scale = torch.FloatTensor(n).uniform_(*self.scale_range_mask)

        aspect_ratio = torch.cat(
            aspect_ratio,
            torch.FloatTensor(1).uniform_(*self.aspect_ratio_range_context),
        )
        scale = torch.cat(
            scale, torch.FloatTensor(1).uniform_(*self.scale_range_context)
        )

        # calculate area, width and height of each mask and context
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
        heights, widths = self.get_random_width_and_height(self.nb_mask)
        xs = np.random.randint(low=self.sqrt_count_patch - widths[:-1].int() + 1)
        ys = np.random.randint(low=self.sqrt_count_patch - heights[:-1].int() + 1)

        # calculate patch indexes of mask
        masks = []
        masks_indexes = set()
        z = zip(xs, ys, heights[:-1], widths[:-1])
        for x, y, height, width in z:
            tmp_mask = torch.cat(
                [
                    torch.arange(x, x + width) + (y + i) * self.sqrt_count_patch
                    for i in range(height)
                ]
            )

            masks_indexes.update(tmp_mask.tolist())
            masks.append(tmp_mask)

        context_indexes = set(
            torch.cat(
                [
                    torch.arange(xs[-1], xs[-1] + widths[-1])
                    + (ys[-1] + i) * self.sqrt_count_patch
                    for i in range(heights[-1])
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
