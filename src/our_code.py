import torch.nn as nn
import torch


from custom_vit import VisionTransformer


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 96,  # dataset depedant
        patch_size: int = 8,  # -> 96 / 12 = 8 (get close to the paper '14')
        in_chans=3,  # 3, see dataset description
        embed_dim=192,  # 192 -> vit_tiny implem
        num_heads: int = 12,  # TODO: find good value between 12-100 (8 in source code)
        num_layers: int = 12,  # TODO: find good value
        num_classes: int = 0,  # Let this param to 0, (it will keep the size of the generated embeddings)
        norm=nn.LayerNorm,  # Layer Norm from Torch (ref: source code)
        dropout: float = 0.0,  # TODO: 0. for now (ref: source code)
        attention_dropout: float = 0.0,  # TODO: 0. for now (ref: source code)
    ):
        super().__init__()

        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=4
            * embed_dim,  # mlp_dim is the FIRST hidden layer size inside MLP (convention?)
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            norm_layer=norm,
        )

        # replace the heads by the identity operator that is argument-insensitive.
        self.vit.heads = nn.Identity()

        # normalization init
        # self.norm = norm(embed_dim)

        # ==== Positional embeddings ====
        nb_patches = (image_size // patch_size) ** 2

        self.positional_embeddings = nn.Parameter(
            # (1, nb_patches, embed_dim)
            torch.zeros(1, nb_patches, embed_dim),
            requires_grad=False,
        )

    def forward(self, x, masks=None):
        # x (B, C, H, W)
        # x = self.patch_embed(x)  # (B, PW * PH, N)

        # add positional encoding

        # if masks != None:
        #     x = apply_masks(x, masks)

        # pass through transformer

        # eux ils font une layernorm à la fin ???

        x = self.vit(x, masks)

        return x


class ViTPredictor(nn.Module):
    pass
