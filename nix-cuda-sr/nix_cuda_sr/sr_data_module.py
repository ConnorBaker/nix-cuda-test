from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import rawpy
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


def read_arw(p: Path) -> tuple[Tensor, Tensor]:
    """
    Read a raw image file and return tensors with shapes (H, W) and (C, H, W).
    """
    with rawpy.imread(p.as_posix()) as raw:
        return torch.asarray(raw.raw_colors), torch.asarray(raw.postprocess(half_size=False)).permute(2, 0, 1)


class SRDataset(VisionDataset):
    """
    Base class for SR datasets.
    """

    train: bool
    burst_paths: list[Path]

    def __init__(self, train: bool) -> None:
        super().__init__()
        self.train = train

        root = Path("/home/connorbaker/Pictures/Tensor-ARW")
        excluded_paths = {
            # Contains images which are rotated, so the tensors are not the same size
            root / "2015-11-28" / "000",
            root / "2015-11-28" / "003",
            # Contains panning sequences
            root / "2015-08-16" / "001",
            root / "2015-08-17" / "016",
            root / "2015-08-21" / "019",
            root / "2015-12-25" / "001",
            root / "2015-12-25" / "003",
            root / "2017-07-15" / "049",  # NOTE: Should be combined with 050
            root / "2017-07-15" / "050",
            root / "2017-07-15" / "056",
            # Contains different exposure sequences
            root / "2015-08-17" / "035",
            root / "2015-08-17" / "036",
            root / "2015-08-17" / "040",
            root / "2015-08-17" / "041",
            # Contains very little in common across frames
            root / "2016-07-03" / "005",
            root / "2016-07-03" / "015",
            # Contains 115 frames, so too large to process
            root / "2016-11-13" / "000",
        }
        train_years = {"2015", "2016"}
        val_years = {"2017", "2018"}
        self.burst_paths = []
        for date_dir in sorted(root.iterdir()):
            include_dirs = False
            include_dirs |= train and any(map(date_dir.name.startswith, train_years))
            include_dirs |= not train and any(map(date_dir.name.startswith, val_years))
            if include_dirs:
                burst_paths = sorted(
                    set(
                        no_ext
                        for d in date_dir.iterdir()
                        if (no_ext := d.with_suffix("").with_suffix("")) not in excluded_paths
                    )
                )
                self.burst_paths.extend(burst_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # CPU because cannot re-initialize CUDA in subprocess, so must use num_workers=0
        # Add a channel dimension before the frame dimension
        # NOTE: Shoving torch.load inside the function call to enable Python to free up memory
        # when the function returns
        hr_mosaicked_imgs = torch.load(
            self.burst_paths[index].with_suffix(".mosaiced.pt"), map_location="cpu", weights_only=True
        )
        hr_demosaicked_imgs = torch.load(
            self.burst_paths[index].with_suffix(".demosaiced.pt"), map_location="cpu", weights_only=True
        )
        return hr_mosaicked_imgs, hr_demosaicked_imgs

    def __len__(self) -> int:
        # Implement the logic to get the length of the dataset
        return len(self.burst_paths)


@dataclass(kw_only=True)
class SRDataModule(pl.LightningDataModule):
    """
    DataModule for SR dataset.
    """

    # Args
    batch_size: int
    drop_last: bool
    num_workers: int
    pin_memory: bool

    # Non-args
    train_dataset: SRDataset = field(init=False)
    val_dataset: SRDataset = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = SRDataset(train=True)
        self.val_dataset = SRDataset(train=False)

    # Depends on the type of train_transforms
    def train_dataloader(self) -> DataLoader[Any]:
        train_loader: DataLoader[Any] = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            # prefetch_factor=4,
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
            # prefetch_factor=4,
            shuffle=False,
        )
        return val_loader
