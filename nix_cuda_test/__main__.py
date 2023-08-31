import argparse

import pytorch_lightning as pl
from torchvision.transforms import Compose, Resize, ToTensor  # type: ignore[import]

from nix_cuda_test.cifar_data_module import CIFARDataModule
from nix_cuda_test.wrapped_vit import WrappedViT


def main():
    from lightning_fabric.fabric import Fabric  # type: ignore

    Fabric.seed_everything(42, workers=True)

    import os

    os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

    import torch.backends.cuda

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    import torch.backends.cudnn

    torch.backends.cudnn.allow_tf32 = True

    # BF16 should be enough for our use case.
    # See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("medium")  # type: ignore

    import torch._dynamo.config

    # torch._dynamo.config.log_level = logging.DEBUG
    # torch._dynamo.config.verbose = True
    import torch._inductor.config

    torch._inductor.config.compile_threads = 1
    torch._inductor.config.dce = True
    torch._inductor.config.epilogue_fusion = True
    torch._inductor.config.permute_fusion = True
    torch._inductor.config.reordering = True
    # torch._inductor.config.shape_padding = True
    torch._inductor.config.size_asserts = False
    # torch._inductor.config.triton.cudagraphs = True
    # torch._inductor.config.tune_layout = True
    torch._dynamo.reset()

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
        default=16,
        help="number of classes in dataset (default : 16)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs (default : 10)")
    parser.add_argument("--lr", type=int, default=1e-2, help="base learning rate (default : 0.01)")
    parser.add_argument(
        "--weight-decay",
        type=int,
        default=3e-2,
        help="weight decay value (default : 0.03)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size (default : 64)")
    parser.add_argument("--compile", action="store_true", help="compile the model")
    args = parser.parse_args()

    transforms = Compose([Resize(size=(args.img_size, args.img_size), antialias=True), ToTensor()])
    data_module = CIFARDataModule(
        batch_size=args.batch_size,
        data_dir="data",
        drop_last=True,
        num_workers=32,
        pin_memory=False,
        train_transforms=transforms,
        val_transforms=transforms,
    )

    model = WrappedViT(
        batch_size=args.batch_size,
        dropout=args.dropout,
        latent_size=args.latent_size,
        lr=args.lr,
        n_channels=args.n_channels,
        num_classes=args.num_classes,
        num_encoders=args.num_encoders,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        weight_decay=args.weight_decay,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=args.epochs,
        precision="bf16-mixed",
        strategy="auto",
    )

    if args.compile:
        model = torch.compile(model)

    trainer.fit(
        datamodule=data_module,
        model=model,
    )


if __name__ == "__main__":
    main()


# class ViT(nn.Module):
#     def __init__(
#         self,
#         *,
#         image_size,
#         patch_size,
#         num_classes,
#         dim,
#         depth,
#         heads,
#         mlp_dim,
#         channels=3,
#     ):
#         super().__init__()
#         assert (
#             image_size % patch_size == 0
#         ), "image dimensions must be divisible by the patch size"
#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = channels * patch_size**2

#         self.patch_size = patch_size

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.transformer = nn.Transformer(dim, depth, heads, mlp_dim)

#         self.to_cls_token = nn.Identity()

#         self.mlp_head = nn.Sequential(
#             nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, num_classes)
#         )

#     def forward(self, img, mask=None):
#         p = self.patch_size

#         x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
#         x = self.patch_to_embedding(x)

#         cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding
#         x = self.transformer(x, mask)

#         x = self.to_cls_token(x[:, 0])
#         return self.mlp_head(x)
