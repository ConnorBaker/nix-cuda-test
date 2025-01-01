from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
from torch import Tensor, nn


class SelfAttention(pl.LightningModule):
    """
    A reusable block that applies self-attention to its input.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x, x, x)[0]


class SkipConnection(pl.LightningModule):
    """
    A reusable block that adds a skip (residual) connection around a given module.
    Essentially: output = x + module(x).
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return x + self.module(x)


@dataclass(kw_only=True, eq=False)
class InputEmbedding(pl.LightningModule):
    # Args
    latent_size: int
    n_channels: int
    num_patches: int
    patch_size: int

    # Non-args
    class_token: nn.Parameter = field(init=False)
    input_size: int = field(init=False)
    linear_proj: nn.Linear = field(init=False)
    pos_embedding: nn.Parameter = field(init=False)
    unfold: nn.Unfold = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        # Unfold extracts patches of size (patch_size x patch_size)
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

        # Linear projection from patch_size^2 * n_channels -> latent_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels
        self.linear_proj = nn.Linear(
            in_features=self.input_size,
            out_features=self.latent_size,
        )

        # Class token (shape = [1, 1, latent_size]), broadcast over batch dimension
        self.class_token = nn.Parameter(torch.zeros(size=(1, 1, self.latent_size)))

        # Positional embedding (shape = [1, 1 + num_patches, latent_size])
        # The “+1” accounts for the class token.
        self.pos_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(size=(1, 1 + self.num_patches, self.latent_size)))
        )

    def forward(self, input_data: Tensor) -> Tensor:
        # input_data: [B, C, H, W]
        B = input_data.size(0)

        # 1) Patchify
        #    shape after unfold: [B, C*patch_size^2, num_patches]
        patches = self.unfold(input_data).transpose(1, 2)  # -> [B, num_patches, C*patch_size^2]

        # 2) Linear projection
        embeddings = self.linear_proj(patches)  # -> [B, num_patches, latent_size]

        # 3) Prepend class token
        class_token = self.class_token.expand(B, -1, -1)  # -> [B, 1, latent_size]
        embeddings = torch.cat([class_token, embeddings], dim=1)  # -> [B, 1+num_patches, latent_size]

        # 4) Add positional embedding
        embeddings += self.pos_embedding[:, : embeddings.size(1), :]  # -> [B, 1+num_patches, latent_size]

        return embeddings


@dataclass(kw_only=True, eq=False)
class EncoderBlock(pl.LightningModule):
    # Args
    dropout: float
    latent_size: int
    num_heads: int

    # Non-args
    module: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.module = nn.Sequential(
            # Attention with skip connection
            SkipConnection(
                nn.Sequential(
                    nn.LayerNorm(self.latent_size),
                    SelfAttention(
                        nn.MultiheadAttention(
                            embed_dim=self.latent_size,
                            num_heads=self.num_heads,
                            dropout=self.dropout,
                            batch_first=True,
                        )
                    ),
                )
            ),
            # MLP with skip connection
            SkipConnection(
                nn.Sequential(
                    nn.LayerNorm(self.latent_size),
                    nn.Linear(self.latent_size, self.latent_size * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.latent_size * 4, self.latent_size),
                    nn.Dropout(self.dropout),
                )
            ),
        )

    def forward(self, emb_patches: Tensor) -> Tensor:
        return self.module(emb_patches)


@dataclass(kw_only=True, eq=False)
class EncoderStack(pl.LightningModule):
    # Args
    dropout: float
    latent_size: int
    num_heads: int
    num_encoders: int

    # Non-args
    module: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.module = nn.Sequential(*[
            EncoderBlock(
                dropout=self.dropout,
                latent_size=self.latent_size,
                num_heads=self.num_heads,
            )
            for _ in range(self.num_encoders)
        ])

    def forward(self, emb_patches: Tensor) -> Tensor:
        # emb_patches: [B, 1+num_patches, latent_size]
        # Run through the encoders and then take the class token
        return self.module(emb_patches)[:, 0]  # -> shape: [B, latent_size]
