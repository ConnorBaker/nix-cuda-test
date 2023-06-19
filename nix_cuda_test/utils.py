from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor


@dataclass(kw_only=True, eq=False)
class PatchExtractor(pl.LightningModule):
    # Args
    patch_size: int

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, input_data: Tensor) -> Tensor:  # type: ignore[override]
        batch_size, _channels, height, width = input_data.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, (
            f"Input height ({height}) and width ({width}) must be divisible " f"by the patch size ({self.patch_size})"
        )

        num_patches_h: int = height // self.patch_size
        num_patches_w: int = width // self.patch_size
        num_patches: int = num_patches_h * num_patches_w

        patches: Tensor = (
            input_data.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(batch_size, num_patches, -1)
        )

        return patches


@dataclass(kw_only=True, eq=False)
class InputEmbedding(pl.LightningModule):
    # Args
    batch_size: int
    latent_size: int
    n_channels: int
    patch_size: int

    # Non-args
    class_token: nn.Parameter = field(init=False)
    input_size: int = field(init=False)
    LinearProjection: nn.Linear = field(init=False)
    patchify: PatchExtractor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        # Linear projection
        self.input_size = self.patch_size * self.patch_size * self.n_channels
        self.LinearProjection = nn.Linear(
            in_features=self.input_size,
            out_features=self.latent_size,
        )
        # Class token
        self.class_token = nn.Parameter(torch.randn(size=(self.batch_size, 1, self.latent_size)))
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(size=(self.batch_size, 1, self.latent_size)))
        # Patchify
        self.patchify = PatchExtractor(patch_size=self.patch_size)

    def forward(self, input_data: Tensor) -> Tensor:  # type: ignore[override]
        # Patchifying the Image
        patches: Tensor = self.patchify(input_data)

        linear_projection: Tensor = self.LinearProjection(patches)
        _b, n, _ = linear_projection.shape
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed: Tensor = self.pos_embedding[:, : n + 1, :]
        ret: Tensor = linear_projection + pos_embed

        return ret


@dataclass(kw_only=True, eq=False)
class EncoderBlock(pl.LightningModule):
    # Args
    dropout: float
    latent_size: int
    num_heads: int

    # Non-args
    attention: nn.MultiheadAttention = field(init=False)
    enc_MLP: nn.Sequential = field(init=False)
    norm: nn.LayerNorm = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=self.latent_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.latent_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.enc_MLP = nn.Sequential(
            nn.Linear(in_features=self.latent_size, out_features=self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.latent_size * 4, out_features=self.latent_size),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, emb_patches: Tensor) -> Tensor:  # type: ignore[override]
        first_norm: Tensor = self.norm(emb_patches)
        attention_out: Tensor = self.attention(first_norm, first_norm, first_norm)[0]
        first_added: Tensor = attention_out + emb_patches
        second_norm: Tensor = self.norm(first_added)
        mlp_out: Tensor = self.enc_MLP(second_norm)
        ret: Tensor = mlp_out + first_added

        return ret
