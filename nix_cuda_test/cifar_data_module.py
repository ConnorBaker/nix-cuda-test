from dataclasses import dataclass, field
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose


@dataclass(kw_only=True)
class CIFARDataModule(pl.LightningDataModule):
    """
    DataModule for CIFAR10 dataset.
    """

    # Args
    batch_size: int
    data_dir: str
    drop_last: bool
    num_workers: int
    pin_memory: bool
    train_transforms: Compose
    val_transforms: Compose

    # Non-args
    train_dataset: CIFAR10 = field(init=False)
    val_dataset: CIFAR10 = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.train_transforms,
            download=True,
        )
        self.val_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.val_transforms,
            download=True,
        )

    # Depends on the type of train_transforms
    def train_dataloader(self) -> DataLoader[Any]:
        train_loader: DataLoader[Any] = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )
        return train_loader

    # Depends on the type of val_transforms
    def val_dataloader(self) -> DataLoader[Any]:
        val_loader: DataLoader[Any] = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
        )
        return val_loader
