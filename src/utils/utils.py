import torch
import torch.nn as nn


def image_to_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Splits a batch of images into non-overlapping patches.

    Args:
        images: A batch of images of shape (B, C, H, W).
        patch_size: The patch width and height.

    Returns:
        patches: A tensor of shape (B, N, C, patch_size, patch_size),
                 where N = (H / patch_size)*(W / patch_size).
    """
    B, C, H, W = images.shape
    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    # unfold_out: (B, C*patch_size*patch_size, N)
    unfold_out = unfold(images)
    # -> (B, N, C*patch_size*patch_size)
    unfold_out = unfold_out.transpose(1, 2)
    # -> (B, N, C, patch_size, patch_size)
    patches = unfold_out.view(B, -1, C, patch_size, patch_size)
    return patches


def patches_to_image(patches: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """
    Rebuild (B,C,H,W) from (B,N,C,p,p).
    """
    B, N, C, p, p2 = patches.shape
    fold = nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    x = patches.view(B, N, C * p * p).transpose(1, 2)  # (B, C*p^2, N)
    images = fold(x)  # (B, C, H, W)
    return images


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x
