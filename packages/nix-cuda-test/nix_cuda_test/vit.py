from dataclasses import dataclass, field

import pytorch_lightning as pl
from torch import Tensor, nn

from nix_cuda_test.utils import EncoderStack, InputEmbedding


@dataclass(kw_only=True, eq=False)
class ViT(pl.LightningModule):
    # Args
    dropout: float
    latent_size: int
    n_channels: int
    num_classes: int
    num_encoders: int
    num_heads: int
    num_patches: int
    patch_size: int

    # Non-args
    module: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.module = nn.Sequential(
            InputEmbedding(
                latent_size=self.latent_size,
                n_channels=self.n_channels,
                num_patches=self.num_patches,
                patch_size=self.patch_size,
            ),
            EncoderStack(
                dropout=self.dropout,
                latent_size=self.latent_size,
                num_heads=self.num_heads,
                num_encoders=self.num_encoders,
            ),
            # Classifier
            nn.Linear(
                in_features=self.latent_size,
                out_features=self.num_classes,
            ),
        )

    def forward(self, test_input: Tensor) -> Tensor:
        return self.module(test_input)
