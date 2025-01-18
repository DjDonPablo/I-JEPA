import torch
import numpy as np


def apply_masks(x: torch.Tensor, masks: list):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """

    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]

    return torch.cat(all_x, dim=0)


def get_random_width_and_height(
    n: int,
    aspect_ratio_range_mask,
    aspect_ratio_range_context,
    scale_range_context,
    scale_range_mask,
    sqrt_count_patch,
):
    """
    Returns a random height and weight based on a random aspect ratio and a random scale.\\
    Length of height and width is `n+1`, the first `n` elements being masks height and width, and the last one being the context height and weight.

    Parameters:
    - `n`: the number of height and width to be generated. 
    """
    aspect_ratio = torch.FloatTensor(n).uniform_(*aspect_ratio_range_mask)
    scale = torch.FloatTensor(n).uniform_(*scale_range_mask)

    aspect_ratio = torch.cat(
        (
            aspect_ratio,
            torch.FloatTensor(1).uniform_(*aspect_ratio_range_context),
        )
    )
    scale = torch.cat((scale, torch.FloatTensor(1).uniform_(*scale_range_context)))

    # calculate area, width and height of each mask and context
    area = scale * (sqrt_count_patch**2)
    height = torch.sqrt(area / aspect_ratio).round().int()
    width = torch.sqrt(area * aspect_ratio).round().int()

    return height, width


def generate_masks(nb_mask: int, sqrt_count_patch: int, batch_size: int):  # 96 / 8 * 8
    ci = []
    for _ in range(batch_size):
        heights, widths = get_random_width_and_height(
            nb_mask, (0.75, 1.5), (1.0, 1.0), (0.85, 1.0), (0.15, 0.2), sqrt_count_patch
        )
        xs = np.random.randint(low=sqrt_count_patch - widths[:-1].int() + 1)
        ys = np.random.randint(low=sqrt_count_patch - heights[:-1].int() + 1)

        # calculate patch indexes of mask
        masks = []
        masks_indexes = set()
        z = zip(xs, ys, heights[:-1], widths[:-1])
        for x, y, height, width in z:
            tmp_mask = torch.cat(
                [
                    torch.arange(x, x + width) + (y + i) * sqrt_count_patch
                    for i in range(height)
                ]
            )

            masks_indexes.update(tmp_mask.tolist())
            masks.append(tmp_mask)

        context_indexes = set(
            torch.cat(
                [
                    torch.arange(xs[-1], xs[-1] + widths[-1])
                    + (ys[-1] + i) * sqrt_count_patch
                    for i in range(heights[-1])
                ]
            ).tolist()
        )

        ci.append(
            torch.tensor(list(context_indexes - masks_indexes), dtype=torch.int64)
        )

    return ci
