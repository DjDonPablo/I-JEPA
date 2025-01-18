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
    area = scale * (sqrt_count_patch ** 2)
    height = torch.sqrt(area / aspect_ratio).round().int()
    width = torch.sqrt(area * aspect_ratio).round().int()

    return height, width


def generate_masks(nb_mask: int, sqrt_count_patch: int, batch_size: int):  # 96 / 8 * 8
    ci = []
    for _ in range(batch_size):
        heights, widths = get_random_width_and_height(
            nb_mask,
            (0.75, 1.5),
            (1.0, 1.0),
            (0.85, 1.0),
            (0.15, 0.2),
            sqrt_count_patch
        )

        print(f"1|\t\theights: {heights.shape}, widths: {widths.shape}\n")

        xs = np.random.randint(low=sqrt_count_patch - widths[:-1].int() + 1)
        ys = np.random.randint(low=sqrt_count_patch - heights[:-1].int() + 1)

        print(f"2|\t\txs: {xs.shape}, ys: {ys.shape}\n")

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


def generate_masks_badbatching(nb_mask: int, sqrt_count_patch: int, batch_size: int):
    heights_list = []
    widths_list = []
    for _ in range(batch_size):
        heights, widths = get_random_width_and_height(
            nb_mask,
            (0.75, 1.5),
            (1.0, 1.0),
            (0.85, 1.0),
            (0.15, 0.2),
            sqrt_count_patch
        )
        heights_list.append(heights)
        widths_list.append(widths)

    # dimension : (batch_size * (nb_mask + 1),)
    heights = torch.cat(heights_list)
    widths = torch.cat(widths_list)

    print(f"1|\t\theights: {heights.shape}, widths: {widths.shape}\n")

    # dimension : (batch_size, nb_mask + 1)
    # lignes : batch_size : 1 ligne pour chaque élément du batch
    # colonnes : nb_mask + 1 : nb_mask masques + 1 contexte
    heights = heights.view(batch_size, nb_mask + 1)
    widths = widths.view(batch_size, nb_mask + 1)

    print(f"2|\t\theights: {heights.shape}, widths: {widths.shape}\n")

    # dimension : (batch_size, nb_mask)
    # lignes : batch_size : 1 ligne pour chaque élément du batch
    # colonnes : nb_mask : 1 colonne pour chaque masque
    xs = torch.randint(
        low=0,
        high=(sqrt_count_patch - widths[:, :-1].int() + 1).min().item(),
        size=(batch_size, nb_mask)
    )
    ys = torch.randint(
        low=0,
        high=(sqrt_count_patch - heights[:, :-1].int() + 1).min().item(),
        size=(batch_size, nb_mask)
    )

    print(f"3|\t\txs: {xs.shape}, ys: {ys.shape}\n")

    # dimension : (batch_size, nb_mask, Z)
    masks = torch.cat([
        torch.cat([
            torch.arange(start=int(xs[i, j]),
                         end=int(xs[i, j] + widths[i, j])) + (ys[i, j] + k) * sqrt_count_patch
            for k in range(heights[i, j])
        ])
        for i in range(batch_size)
        for j in range(nb_mask)
    ]).view(batch_size, nb_mask, -1)

    print(f"4|\t\tmasks: {masks.shape}\n")

    # dimension : (batch_size, nb_mask * Z)
    mask_indexes = masks.view(batch_size, -1)

    print(f"5|\t\tmask_indexes: {mask_indexes.shape}\n")

    # dimension : (batch_size, Z)
    context_xs = xs[:, -1]
    context_ys = ys[:, -1]
    context_heights = heights[:, -1]
    context_widths = widths[:, -1]

    print(
        f"6|\t\tcontext_xs: {context_xs.shape}, context_ys: {context_ys.shape}, context_heights: {context_heights.shape}, context_widths: {context_widths.shape}\n")

    # dimension : (batch_size, YZ)
    context_indexes = torch.cat([
        torch.cat([
            torch.arange(start=int(context_xs[i]),
                         end=int(context_xs[i] + context_widths[i])) + (context_ys[i] + k) * sqrt_count_patch
            for k in range(context_heights[i])
        ])
        for i in range(batch_size)
    ]).view(batch_size, -1)

    print(f"7|\t\tcontext_indexes: {context_indexes.shape}\n")

    # dimension : [(context_indexes,), ..., (context_indexes,)] -> len(list) = batch_size
    context_indexes = [
        torch.tensor(
            list(
                set(context_indexes[i].tolist()) - set(mask_indexes[i].tolist())
            ), dtype=torch.int64)
        for i in range(batch_size)
    ]

    print(f"8|\t\tcontext_indexes: {len(context_indexes)}\n")

    return context_indexes


if __name__ == "__main__":
    generate_masks_batched(4, 12, 32)
