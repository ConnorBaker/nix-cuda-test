from dataclasses import dataclass, field

import pytorch_lightning as pl
from torch import Tensor, nn

from nix_cuda_test.utils import EncoderBlock, InputEmbedding


@dataclass(kw_only=True, eq=False)
class ViT(pl.LightningModule):
    # Args
    batch_size: int
    dropout: float
    latent_size: int
    n_channels: int
    num_classes: int
    num_encoders: int
    num_heads: int
    patch_size: int

    # Non-args
    embedding: InputEmbedding = field(init=False)
    encoders: nn.Sequential = field(init=False)
    MLPHead: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.embedding = InputEmbedding(
            batch_size=self.batch_size,
            latent_size=self.latent_size,
            n_channels=self.n_channels,
            patch_size=self.patch_size,
        )
        # Encoder Stack
        encoder_blocks = [
            EncoderBlock(
                dropout=self.dropout,
                latent_size=self.latent_size,
                num_heads=self.num_heads,
            )
            for _ in range(self.num_encoders)
        ]
        self.encoders = nn.Sequential(*encoder_blocks)
        self.MLPHead = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.latent_size),
            nn.Linear(in_features=self.latent_size, out_features=self.latent_size),
            nn.Linear(in_features=self.latent_size, out_features=self.num_classes),
        )

    def forward(self, test_input: Tensor) -> Tensor:
        embedded: Tensor = self.embedding(test_input)
        enc_output: Tensor = self.encoders(embedded)
        class_token_embed: Tensor = enc_output[:, 0]
        ret: Tensor = self.MLPHead(class_token_embed)
        return ret
