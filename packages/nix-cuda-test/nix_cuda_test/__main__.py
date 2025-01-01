import argparse
import warnings

import torch._inductor.config
import torch.backends.cuda
import torch.backends.cudnn
from pytorch_lightning.plugins import TransformerEnginePrecision
from pytorch_lightning.trainer.trainer import Trainer
from torchvision.transforms import Compose, Resize, ToTensor  # type: ignore[import]
from transformer_engine.common.recipe import DelayedScaling

from nix_cuda_test.cifar_data_module import CIFARDataModule
from nix_cuda_test.wrapped_te_vit import WrappedTEViT

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
    torch._inductor.config.cuda.version = "12.6"
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
        default=224,
        help="image size to be reshaped to (default : 224)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        # NOTE: CIFAR10 has 10 classes, but Transformer Engine requires we use a multiple of eight.
        default=16,
        help="number of classes in dataset (default : 16)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs (default : 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="base learning rate (default : 0.0001)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-2,
        help="weight decay value (default : 0.03)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size (default : 64)")
    parser.add_argument("--compile", action="store_true", help="compile the model")
    args = parser.parse_args()

    transforms = Compose([
        Resize(size=(args.img_size, args.img_size), antialias=True),  # type: ignore[assignment]
        ToTensor(),
    ])
    data_module = CIFARDataModule(
        batch_size=args.batch_size,
        data_dir="data",
        drop_last=True,
        num_workers=32,
        pin_memory=False,
        train_transforms=transforms,
        val_transforms=transforms,
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
        # precision="bf16-mixed",
        strategy="auto",
        # profiler="simple",
    )

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = WrappedTEViT(
            # model = WrappedViT(
            dropout=args.dropout,
            latent_size=args.latent_size,
            lr=args.lr,
            n_channels=args.n_channels,
            num_classes=args.num_classes,
            num_encoders=args.num_encoders,
            num_heads=args.num_heads,
            num_patches=(args.img_size // args.patch_size) ** 2,
            patch_size=args.patch_size,
            weight_decay=args.weight_decay,
        )

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
