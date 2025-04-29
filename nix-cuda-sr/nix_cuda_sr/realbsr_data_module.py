from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_dtype_image


class RealBSRDataset(VisionDataset):
    """
    Base class for SR datasets.
    """

    lr_bursts_paths: Sequence[Sequence[Path]]
    hr_paths: Sequence[Path]

    def __init__(self, root: Path) -> None:
        super().__init__(root=root)

        lr_bursts_paths = []
        hr_paths = []
        for dir in sorted(root.iterdir()):
            lr_burst_paths = sorted(dir.glob("*_MFSR_Sony_*_x1_*.png"))
            assert len(lr_burst_paths) == 14, f"Expected 14 LR burst paths, got {len(lr_burst_paths)}"
            lr_bursts_paths.append(lr_burst_paths)

            hr_path = sorted(dir.glob("*_MFSR_Sony_*_x4_rgb.png"))
            assert len(hr_path) == 1, f"Expected 1 HR path, got {len(hr_path)}"
            hr_paths.extend(hr_path)

        self.lr_bursts_paths = lr_bursts_paths
        self.hr_paths = hr_paths

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        lr_unpacked_mosaicked_imgs = torch.stack(
            [
                to_dtype_image(
                    decode_image(burst_path.as_posix()),
                    dtype=torch.bfloat16,
                    scale=True,
                )
                for burst_path in self.lr_bursts_paths[index]
            ],
            dim=1,  # (4, F, H / 2, W / 2)
        )
        hr_demosaicked_img = to_dtype_image(
            decode_image(self.hr_paths[index].as_posix()),
            dtype=torch.bfloat16,
            scale=True,
        )  # (3, H * 4, W * 4)
        return lr_unpacked_mosaicked_imgs, hr_demosaicked_img

    def __len__(self) -> int:
        return len(self.hr_paths)


@dataclass(kw_only=True)
class RealBSRDataModule(pl.LightningDataModule):
    """
    DataModule for SR dataset.
    """

    # Args
    root: Path
    batch_size: int

    # Non-args
    train_dataset: RealBSRDataset = field(init=False)
    val_dataset: RealBSRDataset = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = RealBSRDataset(self.root / "train")
        self.val_dataset = RealBSRDataset(self.root / "test")

    # NOTE: RealBSR dataset is saturated training with 2 workers on a 4090.

    # Depends on the type of train_transforms
    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            # num_workers=0,
            # prefetch_factor=4,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    # Depends on the type of val_transforms
    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            # num_workers=0,
            # prefetch_factor=4,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
