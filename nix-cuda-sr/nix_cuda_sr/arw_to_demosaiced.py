import json
from pathlib import Path

import rawpy
import torch
import torchvision.transforms.v2 as T
from torch import Tensor

from nix_cuda_sr.bayer import bayer_mosaic_unpack


def read_arw(p: Path) -> tuple[Tensor, Tensor]:
    """
    Read a raw image file and return tensors with shapes (4, H//2, W//2) and (3, H, W).
    """
    with rawpy.imread(p.as_posix()) as raw:
        return bayer_mosaic_unpack(torch.asarray(raw.raw_colors)), torch.asarray(
            raw.postprocess(half_size=False)
        ).permute(2, 0, 1)


def read_arw_burst(burst_dir: Path) -> tuple[Tensor, Tensor]:
    """
    Read a burst of raw image files and return tensors with shapes (4, F, H//2, W//2) and (3, F, H, W).
    """
    mosaiced_stack = []
    demosaiced_stack = []
    for image_path in sorted(burst_dir.iterdir()):
        mosaiced, demosaiced = read_arw(image_path)
        mosaiced_stack.append(mosaiced)
        demosaiced_stack.append(demosaiced)

    # Stack the mosaiced and demosaiced images
    mosaiced = torch.stack(mosaiced_stack, dim=1)
    demosaiced = torch.stack(demosaiced_stack, dim=1)
    return mosaiced, demosaiced


def pre_process_image(image: Tensor) -> Tensor:
    """
    Pre-process the image tensor by resizing and cropping.
    """
    # Resize to 1200x1600 and crop to 1024x1024
    # Resize to 600x800 and crop to 512x512
    # Resize to 300x400 and crop to 256x256
    image = T.functional.resize_image(image, size=[1200, 1600], antialias=True)
    image = T.functional.center_crop_image(image, [1024, 1024])
    return image


def main() -> None:
    # Manifest maps burst directories to shape
    manifest = {}
    source_root: Path = Path("/home/connorbaker/Pictures/ARW")
    dest_root: Path = Path("/home/connorbaker/Pictures/Tensor-1024-ARW")

    excluded_paths = {
        # Contains images which are rotated, so the tensors are not the same size
        source_root / "2015-11-28" / "000",
        source_root / "2015-11-28" / "003",
        # Contains panning sequences
        source_root / "2015-08-16" / "001",
        source_root / "2015-08-17" / "016",
        source_root / "2015-08-21" / "019",
        source_root / "2015-12-25" / "001",
        source_root / "2015-12-25" / "003",
        source_root / "2017-07-15" / "049",  # NOTE: Should be combined with 050
        source_root / "2017-07-15" / "050",
        source_root / "2017-07-15" / "056",
        # Contains different exposure sequences
        source_root / "2015-08-17" / "035",
        source_root / "2015-08-17" / "036",
        source_root / "2015-08-17" / "040",
        source_root / "2015-08-17" / "041",
        # Contains very little in common across frames
        source_root / "2016-07-03" / "005",
        source_root / "2016-07-03" / "015",
        # Contains 115 frames, so too large to process
        source_root / "2016-11-13" / "000",
    }

    dest_root.mkdir(parents=True, exist_ok=True)
    for date_dir in sorted(source_root.iterdir()):
        for burst_dir in sorted(date_dir.iterdir()):
            # Skip excluded paths
            if burst_dir in excluded_paths:
                print(f"Skipping excluded path: {burst_dir}")
                continue

            mosaiced, demosaiced = read_arw_burst(burst_dir)
            mosaiced = pre_process_image(mosaiced)
            demosaiced = pre_process_image(demosaiced)

            # Update the manifest with the burst directory and its shape
            manifest[burst_dir.relative_to(source_root).as_posix()] = mosaiced.shape
            print(f"burst_dir: {burst_dir.relative_to(source_root)}, shape: {mosaiced.shape}")

            # Save the burst directory as a tensor
            output_path: Path = (dest_root / burst_dir.relative_to(source_root)).with_suffix(".mosaiced.pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mosaiced, output_path)
            output_path = (dest_root / burst_dir.relative_to(source_root)).with_suffix(".demosaiced.pt")
            torch.save(demosaiced, output_path)

    # Save the manifest to a file
    manifest_path = dest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
