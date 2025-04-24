import argparse
from pathlib import Path

import torch
from pytorch_lightning.plugins import TransformerEnginePrecision
from pytorch_lightning.trainer.trainer import Trainer
from transformer_engine.common.recipe import DelayedScaling

import nix_cuda_sr
from nix_cuda_sr.sr_data_module import SRDataModule
from nix_cuda_sr.wrapped_te_vit import WrappedViT
from nix_cuda_sr.wrapped_conv_net import WrappedConvNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Transformer in PyTorch")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="patch size for images (default : 16)",
    )
    parser.add_argument("--embed-dim", type=int, default=256, help="embed dim (default : 256)")
    parser.add_argument("--ffn-dim", type=int, default=512, help="feed forward network dim (default : 512)")
    parser.add_argument("--num-heads", type=int, default=8, help="(default : 8)")
    parser.add_argument("--num-layers", type=int, default=4, help="number of encoder layers (default : 4)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs (default : 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="base learning rate (default : 0.0001)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-2,
        help="weight decay value (default : 0.03)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size (default : 1)")
    # TODO: compile not working with TE's scheduler for optimizer.
    # parser.add_argument("--compile", action="store_true", help="compile the model")
    args = parser.parse_args()

    # precision = TransformerEnginePrecision(
    #     weights_dtype=torch.bfloat16,
    #     # NOTE: Both of these require Hopper or newer.
    #     recipe=DelayedScaling(
    #         fp8_dpa=False,
    #         fp8_mha=False,
    #     ),
    #     replace_layers=True,
    # )

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=args.epochs,
        # plugins=[precision],
        # fast_dev_run=True,
        precision="transformer-engine",
        # precision="bf16-mixed",
        strategy="auto",
        # profiler="simple",
    )

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = WrappedConvNet(
            upscale_factor=2,
            embed_dim=args.embed_dim,
            # ffn_dim=args.ffn_dim,
            # num_heads=args.num_heads,
            # num_layers=args.num_layers,
            patch_size=args.patch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        model = model.to(memory_format=torch.channels_last_3d)

    # NOTE: didn't see a performance improvement with `fuse_wgrad_accumulation` on the 4090.
    # Did see a large decrease in training and validation accuracy.

    # if args.compile:
    # model = torch.compile(model)  # type: ignore[assignment]

    trainer.fit(
        datamodule=SRDataModule(
            root=Path("/home/connorbaker/Pictures/Tensor-256-ARW"),
            batch_size=args.batch_size,
        ),
        model=model,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    nix_cuda_sr._setup()
    # from lightning_fabric.fabric import Fabric  # noqa: PLC0415
    # Fabric.seed_everything(42, workers=True)
    # torch._dynamo.reset()  # type: ignore[no-untyped-call]
    main()
