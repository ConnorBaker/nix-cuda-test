from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.adam import Adam

from nix_cuda_test.vit import ViT


@dataclass(kw_only=True, eq=False)
class WrappedViT(pl.LightningModule):
    """
    On default settings:

    Training Loss : 2.3081023390197752
    Valid Loss : 2.302861615943909

    However, this score is not competitive compared to the
    high results in the original paper, which were achieved
    through pre-training on JFT-300M dataset, then fine-tuning
    it on the target dataset. To improve the model quality
    without pre-training, we could try training for more epochs,
    using more Transformer layers, resizing images or changing
    patch size.
    """

    # Args
    batch_size: int
    dropout: float
    latent_size: int
    n_channels: int
    num_classes: int
    num_encoders: int
    num_heads: int
    patch_size: int

    # Optimizer args
    lr: float
    weight_decay: float

    # Non-args
    criterion: nn.CrossEntropyLoss = field(init=False)
    model: ViT = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = ViT(
            batch_size=self.batch_size,
            dropout=self.dropout,
            latent_size=self.latent_size,
            n_channels=self.n_channels,
            num_classes=self.num_classes,
            num_encoders=self.num_encoders,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
        )

    def forward(self, test_input: Tensor) -> Tensor:  # type: ignore[override]
        ret: Tensor = self.model(test_input)
        return ret

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # type: ignore[override]
        images, labels = batch
        logits: Tensor = self(images)
        loss: Tensor = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)  # type: ignore
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # type: ignore[override]
        images, labels = batch
        logits: Tensor = self(images)
        loss: Tensor = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)  # type: ignore
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer: Optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
