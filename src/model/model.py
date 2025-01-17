import torch.nn as nn
import torch
from torchvision.models import VisionTransformer
from src.utils.utils import image_to_patches, patches_to_image

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


# FIXME: this class is NOT good at all. It's just a placeholder.
class PatchDecoder(nn.Module):
    """
    Maps (B, embed_dim) -> (B, 3, patch_size, patch_size).
    A simple MLP for demonstration.
    """
    def __init__(self, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        out_dim = 3 * patch_size * patch_size
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, z):
        # z: (B, embed_dim)
        x = self.net(z)  # => (B, out_dim)
        B, outdim = x.shape
        x = x.view(B, 3, self.patch_size, self.patch_size)
        return x  # => (B,3,p,p)


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.context_encoder = VisionTransformer(
            image_size=96,
            patch_size=8,
            num_layers=12,
            num_heads=12,
            hidden_dim=embed_dim,
            mlp_dim=4 * embed_dim,
            num_classes=0,
        )
        # Return the final embedding => shape (B, embed_dim)
        self.context_encoder.heads = nn.Identity()

    def forward(self, x):
        return self.context_encoder(x)  # (B, embed_dim) by default


class ViTPredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, context_repr, mask_tokens):
        # shape of context_repr => (B, embed_dim)
        # shape of mask_tokens => (B, embed_dim)
        x = context_repr + mask_tokens
        return self.predictor(x)  # (B, embed_dim)


class IJEPAModel(nn.Module):
    def __init__(self, embed_dim=768, nb_masks=4, patch_size=8):
        super().__init__()
        self.context_encoder = ViTEncoder(embed_dim)
        self.target_encoder = ViTEncoder(embed_dim)
        self.predictor = ViTPredictor(embed_dim)
        self.decoder = PatchDecoder(embed_dim, patch_size=patch_size)

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
        # 1) patchify images
        patches = image_to_patches(images, patch_size=self.patch_size)  # (B, N=144, C=3, 8, 8)
        masked_images = torch.zeros_like(images)  # (B, 3, H, W)

        # 2) Rebuild the masked_image from the context block
        for b in range(B):
            valid_idxs = context_indices[b][context_indices[b] != -1]
            for idx_ in valid_idxs:
                idx_int = idx_.item()  # 0..143
                row = idx_int // (W // self.patch_size)  # e.g. 12
                col = idx_int % (W // self.patch_size)
                y0, x0 = row * self.patch_size, col * self.patch_size
                masked_images[b, :, y0:y0 + self.patch_size, x0:x0 + self.patch_size] = patches[b, idx_int]

        # 3) Encode the masked images => context embedding
        context_emb = self.context_encoder(masked_images)
        # shape => (B, embed_dim) by default with torchvision VisionTransformer(num_classes=0)

        # 4) Encode the full images => target embedding
        target_emb = self.target_encoder(images)  # (B, embed_dim)

        # return predictions, target_reps
        final_patches = patches.clone()  # shape (B,144,3,8,8)

        # ------------ WITH DECODER ------------
        # for each mask block
        for m_i in range(masks.shape[1]):
            block_idxs = masks[:, m_i, :]  # (B, max_mask_len)
            for b in range(B):
                valid_mask = block_idxs[b][block_idxs[b] != -1]
                for idx_ in valid_mask:
                    idx_int = idx_.item()
                    # we predict an embedding for this patch
                    # we do predictor(context_emb[b], mask_token)
                    # shape => (1, embed_dim)
                    ctx_b = context_emb[b].unsqueeze(0)  # (1, embed_dim)
                    mask_tok = self.mask_token.expand(1, -1)
                    pred_emb = self.predictor(ctx_b, mask_tok)  # => (1, embed_dim)

                    # decode to patch
                    pred_patch = self.decoder(pred_emb)  # => (1,3,8,8)

                    # place in final_patches
                    final_patches[b, idx_int] = pred_patch[0]

        # 5) build the "reconstructed" image from final_patches
        reconstructed = patches_to_image(final_patches, self.patch_size, H, W)  # (B,3,96,96)
        return context_emb, target_emb, reconstructed
