import torch.nn as nn
import torch
import torch.nn.functional as F

from src.model.vit_enc import TransformerEncoder
from src.mask.mask import apply_masks
from src.model.vit_pred import TransformerPrediction


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 96,  # dataset depedant
        patch_size: int = 8,  # -> 96 / 12 = 8 (get close to the paper '14')
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
            mlp_dim=4 * embed_dim,  # mlp_dim is the FIRST hidden layer size inside MLP
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            norm_layer=norm,
        )

    def forward(self, x, masks=None):
        x = self.vit(x, masks)

        return x


class ViTPredictor(nn.Module):
    def __init__(
        self,
        image_size: int = 96,  # dataset depedant
        patch_size: int = 8,  # -> 96 / 12 = 8 (get close to the paper '14')
        embed_dim: int = 192,  # 192 -> vit_tiny implem
        predictor_embed_dim: int = 96,  # 96
        num_heads: int = 12,  # TODO: find good value between 12-100 (8 in source code)
        num_layers: int = 12,  # TODO: find good value
        num_classes: int = 0,  # Let this param to 0, (it will keep the size of the generated embeddings)
        norm=nn.LayerNorm,  # Layer Norm from Torch (ref: source code)
        dropout: float = 0.0,  # TODO: 0. for now (ref: source code)
        attention_dropout: float = 0.0,  # TODO: 0. for now (ref: source code)
    ):
        super().__init__()

        self.vit = TransformerPrediction(
            num_patches=(image_size // patch_size) ** 2,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=4 * embed_dim,  # mlp_dim is the FIRST hidden layer size inside MLP
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            norm_layer=norm,
        )

    def forward(self, x, masks_enc, mask_pred):
        x = self.vit(x, masks_enc, mask_pred)
        return x


class IJEPA(nn.Module):
    def __init__(
        self,
        evaluation_on: bool = False,
        nb_mask: int = 4,
        image_size: int = 96,
        patch_size: int = 12,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 6,
    ):
        super().__init__()
        self.evaluation_on = evaluation_on
        self.nb_mask = nb_mask
        self.context_encoder = ViTEncoder(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.predictor = ViTPredictor(
            embed_dim=embed_dim,
            predictor_embed_dim=embed_dim // 2,
            image_size=image_size,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.target_encoder = ViTEncoder(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    def forward(self, x, mask_enc=None, mask_pred=None):
        if self.evaluation_on:
            with torch.no_grad():
                return self.target_encoder(x)

        x_tmp = self.context_encoder(x, mask_enc)
        with torch.no_grad():
            targets = self.target_encoder(x)

        v = self.predictor(x_tmp, mask_enc, mask_pred)
        targets = F.layer_norm(targets, (targets.size(-1),))
        target = apply_masks(targets, mask_pred)

        return v, target
