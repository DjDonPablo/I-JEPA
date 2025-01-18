import torch.nn as nn

from vit import TransformerEncoder
from ...custom_pred import TransformerPrediction

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

        self.vit = TransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=4 * embed_dim,  # mlp_dim is the FIRST hidden layer size inside MLP (convention?)
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            norm_layer=norm,
        )

        # replace the heads by the identity operator that is argument-insensitive.
        self.vit.heads = nn.Identity()


    def forward(self, x, masks=None):
        x = self.vit(x, masks)

        return x


class ViTPredictor(nn.Module):
    def __init__(
            self,
            image_size: int = 96,           # dataset depedant
            patch_size: int = 8,            # -> 96 / 12 = 8 (get close to the paper '14')

            embed_dim: int = 192,           # 192 -> vit_tiny implem
            predictor_embed_dim: int = 96,  # 96
            num_heads: int = 12,            # TODO: find good value between 12-100 (8 in source code)
            num_layers: int = 12,           # TODO: find good value
            num_classes: int = 0,           # Let this param to 0, (it will keep the size of the generated embeddings)
            norm=nn.LayerNorm,              # Layer Norm from Torch (ref: source code)
            dropout: float = 0.0,           # TODO: 0. for now (ref: source code)
            attention_dropout: float = 0.0  # TODO: 0. for now (ref: source code)
        ):
        super().__init__()

        self.vit = TransformerPrediction(
            num_patches = (image_size // patch_size) ** 2,
            embed_dim = embed_dim,
            predictor_embed_dim = predictor_embed_dim,

            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=4 * embed_dim, # mlp_dim is the FIRST hidden layer size inside MLP (convention?)
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            norm_layer=norm,
        )

        # replace the heads by the identity operator that is argument-insensitive.
        self.vit.heads = nn.Identity()

    def forward(self, x, masks_enc, mask_pred):
        x = self.vit(x, masks_enc, mask_pred)
        return x













