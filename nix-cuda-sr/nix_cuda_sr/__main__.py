import argparse
from pathlib import Path

from pytorch_lightning.trainer.trainer import Trainer

import nix_cuda_sr
from nix_cuda_sr.mlp_mixer.mlp_mixer import MLPMixer
from nix_cuda_sr.mlp_mixer.te_mlp_mixer import TEMLPMixer
from nix_cuda_sr.sr_data_module import SRDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Transformer in PyTorch")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="patch size for images (default : 16)",
    )
    parser.add_argument("--embed-dim", type=int, default=256, help="embed dim (default : 256)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs (default : 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="base learning rate (default : 0.0001)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-2,
        help="weight decay value (default : 0.03)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size (default : 1)")
    args = parser.parse_args()

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=args.epochs,
        check_val_every_n_epoch=50,
        precision="transformer-engine",
        strategy="auto",
    )

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = MLPMixer(
            model=TEMLPMixer(
                # Model parameters
                upscale_factor=2,
                embed_dim=args.embed_dim,
                depth=1,
                patch_size=args.patch_size,
                # Model optimization parameters
                compile_model=True,
                fuse_wgrad_accumulation=True,
                channels_last=True,
            ),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        datamodule = SRDataModule(
            root=Path("/home/connorbaker/Pictures/Tensor-256-ARW"),
            batch_size=args.batch_size,
        )

    trainer.fit(
        model,  # type: ignore[arg-type]
        datamodule=datamodule,
    )


if __name__ == "__main__":
    nix_cuda_sr._setup()
    main()
