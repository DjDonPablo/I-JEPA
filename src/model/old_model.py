import torch.nn as nn
import torch
from torchvision.models import VisionTransformer

from src.utils.utils import repeat_interleave_batch, get_2d_sincos_pos_embed, image_to_patches, patches_to_image
from src.mask.mask import apply_masks

import numpy as np
import math

"""
Architectures.

For I-JEPA pretraining, we use Vision Transformer [29] (ViT) architectures for the :
* context-encoder,
* target-encoder,
* predictor.

While the context-encoders and target-encoders correspond to standard ViT architectures,
the predictor is designed as a light-weight (narrow) ViT architecture.

Specifically, we fix the embedding dimension of the predictor to 384,
while keeping the number of self-attention heads equal to that of the backbone context-encoder.

For the smaller ViT-B/16 context-encoder, we set the depth of the predictor to 6.
For ViT-L/16, ViT-H/16, and ViT-H/14 context- encoders, we set the depth of the predictor to 12.

Finally, the ViT-G/16 uses a predictor of depth 16. I-JEPA is pretrained without a [cls] token.
We use the target-encoder for evaluation and average pool its output to produce a global image representation.
"""


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class ViTPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
            self,
            img_size: int = 96,
            patch_size: int = 8,
            num_patches: int = 12*12,
            embed_dim=768,
            predictor_embed_dim=384,
            num_heads: int = 12,
            num_layers: int = 12,
            num_classes: int = 0,
            norm=nn.LayerNorm,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
    ):
        super().__init__()

        # ViT encoder
        self.vit = VisionTransformer(
            image_size=img_size,  # TODO: Gros point d'inquiétude ici
            patch_size=patch_size,  # TODO: Gros point d'inquiétude ici
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,  # TODO: Gros point d'inquiétude ici
            mlp_dim=4 * embed_dim,  # TODO: Gros point d'inquiétude ici
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            norm_layer=norm,
        )

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.predictor_norm = norm(predictor_embed_dim)

        # Positional embedding
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # Batch size
        B = len(x) // len(masks_x)

        # Embed to predictor dimension
        x = self.predictor_embed(x)

        # Positional embedding
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        # TODO: pas clair du tout
        _, N_ctxt, D = x.shape

        # Masking
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))  # FIXME: pas clair du tout

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Forward propagation
        x = self.vit(x)
        x = self.predictor_norm(x)  # FIXME: utile si norm dans vit ?

        # TODO: pas clair du tout
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class ViTEncoder(nn.Module):
    """ Vision Transformer Encoder : Context-Encoder / Target-Encoder """
    def __init__(
            self,
            img_size: int = 96,
            patch_size: int = 8,
            in_chans=3,
            embed_dim=768,
            num_heads: int = 12,
            num_layers: int = 12,
            num_classes: int = 0,
            norm=nn.LayerNorm,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
    ):
        super().__init__()

        # ViT encoder
        self.vit = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=4 * embed_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            norm_layer=norm,
        )

        self.vit.heads = nn.Identity()

        self.norm = norm(embed_dim)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Patch Embedding
        x = self.patch_embed(x)

        # Positional Embedding
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # Masking
        if masks is not None:
            x = apply_masks(x, masks)

        # Forward propagation
        x = self.vit(x)

        # Normalization layer
        if self.norm is not None:
            x = self.norm(x)  # FIXME: utile si norm dans vit ?

        return x


class IJEPAModel(nn.Module):
    def __init__(
            self,
            img_size: int = 96,
            patch_size: int = 8,
            num_patches: int = 12 * 12,
            embed_dim=768,
            predictor_embed_dim=384,
            num_heads: int = 12,
            num_layers: int = 12,
            num_classes: int = 0,
            norm=nn.LayerNorm,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            nb_masks: int = 5,
    ):
        super().__init__()

        self.context_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            norm=norm,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.target_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            norm=norm,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.predictor = ViTPredictor(
            img_size=img_size,
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            norm=norm,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))

        self.nb_masks = nb_masks
        self.patch_size = patch_size

    def forward(self, images, context_indices, masks):
        """
        images: (B, 3, 96, 96)
        context_indices: (B, 144) with some -1 paddings
        masks: (B, nb_masks-1, max_mask_length) with -1 paddings
        """
        B, C, H, W = images.shape

        patches = image_to_patches(images, patch_size=self.patch_size)
        masked_images = torch.zeros_like(images)  # (B, 3, H, W)

        for batch in range(B):
            valid_idxs = context_indices[batch][context_indices[batch] != -1]
            for idx_ in valid_idxs:
                idx_int = idx_.item()  # 0..143
                row = idx_int // (W // self.patch_size)  # e.g. 12
                col = idx_int % (W // self.patch_size)
                y0, x0 = row * self.patch_size, col * self.patch_size
                masked_images[batch, :, y0:y0 + self.patch_size, x0:x0 + self.patch_size] = patches[batch, idx_int]

        context_emb = self.context_encoder(masked_images)  # (B, embed_dim)
        target_emb = self.target_encoder(images)  # (B, embed_dim)

        predictions = []
        target_reps = []

        for m_i in range(masks.shape[1]):
            mask_tok = self.mask_token.expand(B, -1)  # (B, embed_dim)

            # Predict
            pred_i = self.predictor(context_emb, mask_tok)  # (B, embed_dim)
            predictions.append(pred_i)      # (B, embed_dim)
            target_reps.append(target_emb)  # (B, embed_dim)

        final_patches = patches.clone()  # shape (B,144,3,8,8)

        reconstructed = patches_to_image(final_patches, self.patch_size, H, W)  # (B,3,96,96)
        return context_emb, target_emb, reconstructed
