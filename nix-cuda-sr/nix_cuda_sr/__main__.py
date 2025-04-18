import argparse
import warnings

import torch._inductor.config
import torch.backends.cuda
import torch.backends.cudnn
from pytorch_lightning.plugins import TransformerEnginePrecision
from pytorch_lightning.trainer.trainer import Trainer
from torch import nn
from torchvision.transforms.v2 import (
    Lambda,
    RandomCrop,
    Resize,
    ToDtype,
    ToImage,
)
from transformer_engine.common.recipe import DelayedScaling

from nix_cuda_sr.bayer import bayer_mosaic_unpack
from nix_cuda_sr.sr_data_module import SRDataModule
from nix_cuda_sr.wrapped_te_vit import WrappedViT

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main() -> None:  # noqa: PLR0915
    from lightning_fabric.fabric import Fabric  # noqa: PLC0415

    Fabric.seed_everything(42, workers=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)

    torch.backends.cudnn.allow_tf32 = True

    # BF16 should be enough for our use case.
    # See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("medium")  # type: ignore

    # torch._inductor.config.compile_threads = 1
    torch._inductor.config.dce = True
    torch._inductor.config.permute_fusion = True
    torch._inductor.config.b2b_gemm_pass = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_pointwise = True
    torch._inductor.config.max_autotune_gemm = True
    torch._inductor.config.warn_mix_layout = True

    # enable the combo kernel that combines data-independent kernels (additional
    # to foreach kernels) into a single one (Experimental)
    torch._inductor.config.combo_kernels = True
    # benchmark combo kernels and only allow ones with perf gains
    torch._inductor.config.benchmark_combo_kernel = True
    # combo_kernel autotuning options: 0 - disable, 1 - enable except for foreach,
    # 2 - enable for all
    torch._inductor.config.combo_kernels_autotune = 2
    # Enable masking for combining kernels of mixed sizes: 0 - disable, 1 - enable
    # for all except for foreach, 2 - enable for all
    torch._inductor.config.combo_kernel_allow_mixed_sizes = 2
    # Enable dynamic shapes for foreach kernels
    torch._inductor.config.combo_kernel_foreach_dynamic_shapes = True

    torch._inductor.config.permute_fusion = True
    torch._inductor.config.size_asserts = False

    # torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.triton.autotune_at_compile_time = True
    torch._inductor.config.triton.multi_kernel = True

    torch._inductor.config.cuda.arch = "89"
    torch._inductor.config.cuda.version = "12.8"
    torch._inductor.config.cuda.compile_opt_level = "-O3"
    torch._inductor.config.cuda.enable_cuda_lto = True
    torch._inductor.config.cuda.use_fast_math = True
    torch._dynamo.reset()  # type: ignore[no-untyped-call]

    # te_attention._log_level = 2
    # te_attention.fa_logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Vision Transformer in PyTorch")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="patch size for images (default : 16)",
    )
    parser.add_argument("--latent-size", type=int, default=768, help="latent size (default : 768)")
    parser.add_argument(
        "--n-channels",
        type=int,
        default=3,
        help="number of channels in images (default : 3)",
    )
    parser.add_argument("--num-heads", type=int, default=12, help="(default : 12)")
    parser.add_argument("--num-encoders", type=int, default=12, help="number of encoders (default : 12)")
    parser.add_argument("--dropout", type=int, default=0.1, help="dropout value (default : 0.1)")
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="image size to be reshaped to (default : 256)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs (default : 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="base learning rate (default : 0.0001)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-2,
        help="weight decay value (default : 0.03)",
    )
    # TODO: HAVE TO RUN WITH BATCH SIZE ONE BECAUSE THE NUMBER OF FRAMES VARIES???
    parser.add_argument("--batch-size", type=int, default=1, help="batch size (default : 1)")
    parser.add_argument("--compile", action="store_true", help="compile the model")
    args = parser.parse_args()

    # For the HR images
    # target_transform = nn.Sequential(
    #     # TODO: stupid
    #     Lambda(lambda burst: bayer_mosaic_unpack(burst.unsqueeze(0)).squeeze(0)),
    #     ToImage(),
    #     # RandomPerspective(),
    #     # MUST unpack mosaic prior to random crop
    #     RandomCrop((args.img_size * 2, args.img_size * 2)),
    #     # ScaleJitter((512, 512)),
    #     ToDtype(torch.float16, scale=True),
    #     # Don't move the tensor to the GPU until after it's been transformed to save on memory
    #     Lambda(lambda burst: burst.to(device="cuda", memory_format=torch.channels_last)),
    # )

    # To generate the LR images
    # transform = nn.Sequential(
    #     Resize(size=(args.img_size, args.img_size), antialias=True),  # type: ignore[assignment]
    # )

    data_module = SRDataModule(
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )

    precision = TransformerEnginePrecision(
        weights_dtype=torch.bfloat16,
        # NOTE: Both of these require Hopper or newer.
        recipe=DelayedScaling(
            fp8_dpa=False,
            fp8_mha=False,
        ),
        replace_layers=True,
    )

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=args.epochs,
        plugins=[precision],
        # fast_dev_run=True,
        # precision="bf16-mixed",
        strategy="auto",
        # profiler="simple",
    )

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = WrappedViT(upscale_factor=2, n_channels=args.n_channels, lr=args.lr, weight_decay=args.weight_decay)
        # model = model.to(memory_format=torch.channels_last_3d)
    #     model = WrappedTEViT(
    #         dropout=args.dropout,
    #         latent_size=args.latent_size,
    #         lr=args.lr,
    #         n_channels=args.n_channels,
    #         n_frames=5,  # all of the sets of frames have maximum of five
    #         num_encoders=args.num_encoders,
    #         num_heads=args.num_heads,
    #         patch_size=args.patch_size,
    #         weight_decay=args.weight_decay,
    #     )

    # NOTE: didn't see a performance improvement with `fuse_wgrad_accumulation` on the 4090.
    # Did see a large decrease in training and validation accuracy.

    if args.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    trainer.fit(
        datamodule=data_module,
        model=model,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
