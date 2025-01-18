import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class LinearWeightDecay:
    def __init__(self, adamw: optim.AdamW, initial_wd, end_wd, num_steps):
        self.adamw = adamw
        self.initial_wd = initial_wd
        self.end_wd = end_wd
        self.num_steps = num_steps
        self.linear_increase = (end_wd - initial_wd) / num_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1
        new_wd = self.initial_wd + self.linear_increase * self.step_count
        for param_group in self.adamw.param_groups:
            param_group['weight_decay'] = new_wd
