import argparse
from pathlib import Path

from pytorch_lightning.trainer.trainer import Trainer

import nix_cuda_sr
from nix_cuda_sr.mlp_mixer.mlp_mixer import MLPMixer
from nix_cuda_sr.mlp_mixer.nn_mlp_mixer import NNMLPMixer
from nix_cuda_sr.mlp_mixer.te_mlp_mixer import TEMLPMixer
from nix_cuda_sr.realbsr_data_module import RealBSRDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="MLPMixer in PyTorch")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=20,
        help="patch size for images (default : 20)",
    )
    parser.add_argument("--embed-dim", type=int, default=256, help="embed dim (default : 256)")
    parser.add_argument("--depth", type=int, default=2, help="depth (default : 2)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs (default : 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="base learning rate (default : 0.0001)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-2,
        help="weight decay value (default : 0.03)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size (default : 1)")
    parser.add_argument("--transformer-engine", action="store_true", help="use transformer engine")
    args = parser.parse_args()

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=args.epochs,
        # check_val_every_n_epoch=50,
        precision="transformer-engine" if args.transformer_engine else "bf16",
        strategy="auto",
    )

    model_kwargs = dict(
        # Model parameters
        upscale_factor=4,
        embed_dim=args.embed_dim,
        depth=args.depth,
        patch_size=args.patch_size,
        # Model optimization parameters
        compile_model=True,
        channels_last=True,
    )
    if args.transformer_engine:
        # Model optimization parameters
        # TODO: Do I need to set after_backwards_hook to something to use this?
        # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html#Gradient-accumulation-fusion
        model_kwargs["fuse_wgrad_accumulation"] = True

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = MLPMixer(
            model=(TEMLPMixer if args.transformer_engine else NNMLPMixer)(**model_kwargs),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    datamodule = RealBSRDataModule(
        root=Path("/home/connorbaker/RealBSR/RAW"),
        batch_size=args.batch_size,
    )

    trainer.fit(
        model,  # type: ignore[arg-type]
        datamodule=datamodule,
    )


if __name__ == "__main__":
    nix_cuda_sr._setup()
    main()
