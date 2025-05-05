import argparse
from pathlib import Path

import torch
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner

import nix_cuda_sr
from nix_cuda_sr.mlp_mixer.mlp_mixer import MLPMixer
from nix_cuda_sr.mlp_mixer.nn_mlp_mixer import NNMLPMixer
from nix_cuda_sr.mlp_mixer.te_mlp_mixer import TEMLPMixer
from nix_cuda_sr.realbsr_data_module import RealBSRDataModule

_BATCH_SIZE_DEFAULT = 1
_LR_DEFAULT = 1e-4


def main(args: argparse.Namespace) -> None:
    trainer = Trainer(
        accelerator="auto",
        # accumulate_grad_batches=8,
        # gradient_clip_val=0.75,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=args.epochs,
        # check_val_every_n_epoch=5,
        precision="transformer-engine" if args.transformer_engine else "bf16",
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        strategy="auto",
    )

    datamodule = RealBSRDataModule(
        root=Path("/home/connorbaker/RealBSR/RAW"),
        batch_size=max(args.batch_size, _BATCH_SIZE_DEFAULT),
    )

    model_kwargs = dict(
        # Model parameters
        upscale_factor=4,
        embed_dim=args.embed_dim,
        hidden_sizes=args.hidden_size,
        patch_size=args.patch_size,
        # Model optimization parameters
        compile_model=False,  # TODO: trying to compile the pytorch lightning model overall
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
            lr=max(args.lr, _LR_DEFAULT),
            weight_decay=args.weight_decay,
        )

    if args.batch_size == -1:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=datamodule)

    if args.lr == -1:
        tuner = Tuner(trainer)
        tuner.lr_find(model, datamodule=datamodule)

    assert trainer.logger is not None
    trainer.logger.log_hyperparams(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLPMixer in PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO: Running with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    parser.add_argument("--patch-size", type=int, default=20, help="patch size for images")
    parser.add_argument("--embed-dim", type=int, default=512, help="embed dim")
    parser.add_argument("--hidden-size", type=int, nargs="+", default=[512 * 4] * 4, help="hidden size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=_LR_DEFAULT, help="base learning rate (-1 for automatic)")
    parser.add_argument("--weight-decay", type=float, default=3e-2, help="weight decay value")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE_DEFAULT, help="batch size (-1 for automatic)")
    parser.add_argument("--transformer-engine", action="store_true", help="use transformer engine")
    args = parser.parse_args()

    nix_cuda_sr._setup()
    main(args)
