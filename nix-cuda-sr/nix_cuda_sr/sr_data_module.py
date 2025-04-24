import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


class SRDataset(VisionDataset):
    """
    Base class for SR datasets.
    """

    root: Path
    batch_size: int
    split: Literal["train", "test"]
    batch_burst_paths: Sequence[Sequence[Path]]
    batched_unpacked_mosaicked_imgs: Sequence[Tensor]
    batched_demosaicked_imgs: Sequence[Tensor]

    def __init__(self, root: Path, batch_size: int, split: Literal["train", "test"]) -> None:
        super().__init__(root=root)
        self.batch_size = batch_size
        self.split = split

        manifest: Mapping[str, tuple[int, int, int, int]] = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )

        # Group by burst length
        batch_burst_paths: dict[int, list[Path]] = {}
        for idx, (rel_path, shape) in enumerate(sorted(manifest.items())):
            # training set is 80% of the data, test set is 20%
            if not (split == "train" and idx % 5 != 0) and not (split == "test" and idx % 5 == 0):
                continue

            burst_length = shape[1]
            batch_burst_paths.setdefault(burst_length, [])
            batch_burst_paths[burst_length].append(root / rel_path)

        # Iterate over the sorted burst lengths, creating batches of size `batch_size // burst_length`
        # TODO(@connorbaker): adjusted_batch_size should scale based on the size of the image as well, not just burst
        # length.
        self.batch_burst_paths = []
        for burst_length in sorted(batch_burst_paths.keys()):
            # Skip burst lengths that are too long or don't have enough bursts.
            # if burst_length > 13:
            #     continue

            # TODO(@connorbaker): Gross, just while testing to make sure the network *can* converge.
            if burst_length != 1:
                continue
            # if burst_length < 4 or 24 < burst_length:
            #     continue

            # Adjust the batch size based on the burst length.
            # Each batch will contain at least one burst.
            adjusted_batch_size = max(self.batch_size // burst_length, 1)
            paths = sorted(batch_burst_paths[burst_length])
            while paths:
                self.batch_burst_paths.append(paths[:adjusted_batch_size])
                paths = paths[adjusted_batch_size:]

        self.batched_unpacked_mosaicked_imgs = []
        self.batched_demosaicked_imgs = []
        for burst_paths in self.batch_burst_paths:
            batched = self._read_batch(burst_paths)
            self.batched_unpacked_mosaicked_imgs.append(batched[0])
            self.batched_demosaicked_imgs.append(batched[1])

    @staticmethod
    def _read_batch(burst_paths: Sequence[Path]) -> tuple[Tensor, Tensor]:
        unpacked_mosaicked_imgs = torch.stack(
            [
                T.functional.to_dtype_image(
                    torch.load(
                        burst_path.with_suffix(".mosaiced.pt"),
                        map_location=torch.device("cuda"),
                        mmap=True,
                        weights_only=True,
                    # ),
                    ).to(memory_format=torch.channels_last),
                    # dtype=torch.float32,
                    dtype=torch.bfloat16,
                    scale=True,
                )
                for burst_path in burst_paths
            ],
            dim=0,  # (B, 4, F, H//2, W//2)
        )
        demosaicked_imgs = torch.stack(
            [
                T.functional.to_dtype_image(
                    torch.load(
                        burst_path.with_suffix(".demosaiced.pt"),
                        map_location=torch.device("cuda"),
                        mmap=True,
                        weights_only=True,
                    # ),
                    ).to(memory_format=torch.channels_last),
                    # dtype=torch.float32,
                    dtype=torch.bfloat16,
                    scale=True,
                )
                for burst_path in burst_paths
            ],
            dim=0,  # (B, 3, F, H, W)
        )
        return unpacked_mosaicked_imgs, demosaicked_imgs

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.batched_unpacked_mosaicked_imgs[index], self.batched_demosaicked_imgs[index]

    def __len__(self) -> int:
        return len(self.batch_burst_paths)


@dataclass(kw_only=True)
class SRDataModule(pl.LightningDataModule):
    """
    DataModule for SR dataset.
    """

    # Args
    root: Path
    batch_size: int

    # Non-args
    train_dataset: SRDataset = field(init=False)
    val_dataset: SRDataset = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

    def setup(self, stage: str | None = None) -> None:
        # Consistently more than one frame for bursts of 1-13 frames.
        # After that it gets patchy.

        # Frequency of burst length:
        # jq <~/Pictures/Tensor-256-ARW/manifest.json '.[][1]' | sort -n | uniq -c
        # 106 1
        #  85 2
        #  60 3
        #  55 4
        #  51 5
        #  15 6
        #  24 7
        #  30 8
        #   9 9
        #  14 10
        #   8 11
        #  13 12
        #   7 13
        #   1 14
        #   2 15
        #   1 16
        #   3 17
        #   5 18
        #   4 19
        #   2 20
        #   8 22
        #  35 23
        #  15 24
        #   3 25
        #   2 26
        #   4 27
        #   2 28
        #   1 29
        #   2 30
        #   2 33
        #   1 34
        #   2 40
        #   1 41
        #   1 44
        #   1 45
        #   1 46
        self.train_dataset = SRDataset(self.root, self.batch_size, split="train")
        self.val_dataset = SRDataset(self.root, self.batch_size, split="test")

    # Depends on the type of train_transforms
    def train_dataloader(self) -> Iterable[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
        )

    # Depends on the type of val_transforms
    def val_dataloader(self) -> Iterable[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )
