# ===== ViT Torch source code =====

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once

from src.mask.mask import apply_masks


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i * B:(i + 1) * B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))



    def forward(self, input: torch.Tensor, mask_enc: torch.Tensor, mask_pred: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        if mask_enc:
            input += apply_masks(input, mask_enc)

        pos_embs = torch.zeros_like(input) + self.pos_embedding
        if mask_pred:
            pos_embs = apply_masks(pos_embs, mask_pred)

        pos_embs = repeat_interleave_batch(pos_embs)
        # trunc_normal_ ?
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
    
        pred_tokens += pos_embs
        input = input.repeat(len(mask_pred), 1, 1)
        input = torch.cat([input, pred_tokens], dim=1)
        
        return self.ln(self.layers(self.dropout(input)))


class TransformerPrediction(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        num_patches: int,           # new
        embed_dim: int,             # new
        predictor_embed_dim: int,   # new

        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        #torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        #self.image_size = image_size
        #self.patch_size = patch_size
        #self.hidden_dim = hidden_dim

        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        # self.heads = None # added

        # if conv_stem_configs is not None:
        #     # As per https://arxiv.org/abs/2106.14881
        #     seq_proj = nn.Sequential()
        #     prev_channels = 3
        #     for i, conv_stem_layer_config in enumerate(conv_stem_configs):
        #         seq_proj.add_module(
        #             f"conv_bn_relu_{i}",
        #             Conv2dNormActivation(
        #                 in_channels=prev_channels,
        #                 out_channels=conv_stem_layer_config.out_channels,
        #                 kernel_size=conv_stem_layer_config.kernel_size,
        #                 stride=conv_stem_layer_config.stride,
        #                 norm_layer=conv_stem_layer_config.norm_layer,
        #                 activation_layer=conv_stem_layer_config.activation_layer,
        #             ),
        #         )
        #         prev_channels = conv_stem_layer_config.out_channels
        #     seq_proj.add_module(
        #         "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
        #     )
        #     self.conv_proj: nn.Module = seq_proj
        # else:

        # patch embed
        # self.conv_proj = nn.Conv2d(
        #     in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        # )


        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)


        # TODO:
        #trunc_normal_(self.mask_token, std=self.init_std) # 0.02


        # ------------ old
        seq_length = num_patches + 1

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            predictor_embed_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(predictor_embed_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(predictor_embed_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # if isinstance(self.conv_proj, nn.Conv2d):
        #     # Init the patchify stem
        #     fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        #     nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        #     if self.conv_proj.bias is not None:
        #         nn.init.zeros_(self.conv_proj.bias)
        # elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
        #     # Init the last 1x1 conv of the conv stem
        #     nn.init.normal_(
        #         self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
        #     )
        #     if self.conv_proj.conv_last.bias is not None:
        #         nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor, mask_enc: Optional[torch.Tensor], mask_pred: Optional[torch.Tensor]):
        assert (mask_enc is not None) and (mask_pred is not None), 'Cannot run predictor without mask indices'
        # Reshape and permute the input tensor

        # TODO: verify if really useless
        # x = self._process_input(x)

        # TODO: verify utility
        #if not isinstance(masks_x, list):
        #    masks_x = [masks_x]
        #
        #if not isinstance(masks, list):
        #    masks = [masks]

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)  # TODO useful ???
        x = torch.cat([batch_class_token, x], dim=1)            # TODO useful ???
        
        self.predictor_embed(x)
        x = self.encoder(x, mask_enc, mask_pred)

        # Classifier "token" as used by standard language architectures
        # x = x[:, 0]

        # x = self.heads(x)
        
        _, N_ctxt, _ = x.shape
        x = x[:, N_ctxt:]

        self.predictor_proj(x)

        return x



def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state