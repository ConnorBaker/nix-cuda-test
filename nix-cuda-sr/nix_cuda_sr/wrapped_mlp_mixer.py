from collections.abc import Mapping

import optuna
import pytorch_lightning as pl
import schedulefree
import torch
import torchmetrics
import torchmetrics.functional.image
import torchmetrics.image
import torchvision.transforms.v2 as T
import transformer_engine.pytorch as te
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer


class SkipConnection(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.model = SkipConnection(te.LayerNormMLP(hidden_size, ffn_hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class WrappedMLPMixer(pl.LightningModule):
    def __init__(
        self,
        upscale_factor: int,
        embed_dim: int,  # TODO: Rename to embed dim scaling factor or something.
        patch_size: int,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.depth = 1

        self.patch_embed_dim = self.patch_size**2 * self.embed_dim
        self.scaled_patch_embed_dim = self.upscale_factor**2 * self.patch_embed_dim

        # NOTE: (from https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
        # Modular metrics contain internal states that should belong to only one DataLoader. In case you are using
        # multiple DataLoaders, it is recommended to initialize a separate modular metric instances for each DataLoader
        # and use them separately. The same holds for using separate metrics for training, validation and testing.
        self.train_metrics = torchmetrics.MetricCollection({
            # "train_dists": torchmetrics.image.DeepImageStructureAndTextureSimilarity(), # very slow
            # "train_ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(),
            # "train_msssim": torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(),
            # "train_psnr": torchmetrics.image.PeakSignalNoiseRatio(),
            "train_rmse_sw": torchmetrics.image.RootMeanSquaredErrorUsingSlidingWindow(),
        })
        self.val_metrics = torchmetrics.MetricCollection({
            "val_dists": torchmetrics.image.DeepImageStructureAndTextureSimilarity(),  # very slow
            "val_ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(),
            "val_msssim": torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(),
            "val_psnr": torchmetrics.image.PeakSignalNoiseRatio(),
            "val_rmse_sw": torchmetrics.image.RootMeanSquaredErrorUsingSlidingWindow(),
        })

        self.model = te.ops.Sequential(
            # Linear/MLP requires channel dimension be last.
            # TODO: Number of frames varies, so we cannot include it in the channel dimension at the end since the
            # embedding dimension is fixed.
            Rearrange("b c f (h hp) (w wp) -> b f h w (c hp wp)", c=4, hp=self.patch_size, wp=self.patch_size),
            # Linear transform to embed_dim
            te.Linear(4 * self.patch_size**2, self.patch_embed_dim),
            # MLP layers
            # TODO: LucidRains' implementation alternates between a convolution and a linear layer.
            # https://github.com/lucidrains/mlp-mixer-pytorch/blob/bcd54afa1d41afb0911f8c92657907d2ef95c6c6/mlp_mixer_pytorch/mlp_mixer_3d_pytorch.py#L37
            *[MLPBlock(self.patch_embed_dim, self.scaled_patch_embed_dim) for _ in range(self.depth)],
            # Linear transform and re-arrange to upscale and move channel dimension to front
            te.Linear(self.patch_embed_dim, self.scaled_patch_embed_dim),
            Rearrange(
                "b f h w (c hu hp wu wp) -> b c f (h hu hp) (w wu wp)",
                hu=self.upscale_factor,
                wu=self.upscale_factor,
                c=self.embed_dim,
                hp=self.patch_size,
                wp=self.patch_size,
            ),
            nn.Conv3d(self.embed_dim, 3, kernel_size=1),
        )

    @torch.compile(dynamic=True, mode="max-autotune")
    def forward(self, test_input: Tensor) -> Tensor:  # type: ignore[override]
        # B, C, F, H, W = test_input.shape
        # assert C == 4, f"Expected 4 channels, got {C} instead."
        # assert H % self.patch_size == 0, f"Height {H} is not divisible by patch size {self.patch_size}."
        # assert W % self.patch_size == 0, f"Width {W} is not divisible by patch size {self.patch_size}."

        # x = test_input
        # print(f"Input shape: {x.shape}")
        # for layer in self.model:
        #     print(f"Layer: {layer}")
        #     x = layer(x)
        #     print(f"Output shape: {x.shape}")

        x = self.model(test_input)

        return x

    def training_step(self, batches: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        unpacked_mosaicked_hr_imgs, demosaicked_hr_imgs = batches

        # Create the LR images
        unpacked_mosaicked_lr_imgs = T.functional.resize_image(
            unpacked_mosaicked_hr_imgs, size=[128, 128], antialias=True
        )

        # Superresolve
        demosaicked_resolved_hr_imgs: Tensor = self(unpacked_mosaicked_lr_imgs)

        # NOTE: the metrics don't know how to handle multiple frames -- they just consider them to be batched.
        demosaicked_resolved_hr_imgs = rearrange(demosaicked_resolved_hr_imgs, "b c f h w -> (b f) c h w")
        demosaicked_hr_imgs = rearrange(demosaicked_hr_imgs, "b c f h w -> (b f) c h w")

        metrics: Mapping[str, Tensor] = self.train_metrics(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)
        self.log_dict(metrics, prog_bar=True)
        return metrics["train_rmse_sw"]

    def validation_step(self, batches: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        unpacked_mosaicked_hr_imgs, demosaicked_hr_imgs = batches

        # Create the LR images
        unpacked_mosaicked_lr_imgs = T.functional.resize_image(
            unpacked_mosaicked_hr_imgs, size=[128, 128], antialias=True
        )

        # Superresolve
        demosaicked_resolved_hr_imgs: Tensor = self(unpacked_mosaicked_lr_imgs)

        # Save the first frame of the first batch to Tensorboard
        if batch_idx == 0:
            if self.current_epoch == 0:
                # These don't change between batches, so only log them once
                # TODO: Save a copy of the demoasicked lr imgs to verify visually that functions work as expected.
                self.logger.experiment.add_images(
                    "demosaicked_hr_imgs",
                    T.functional.to_dtype_image(demosaicked_hr_imgs[:, :, 0, :, :], dtype=torch.uint8, scale=True),
                    self.current_epoch,
                )
            self.logger.experiment.add_images(
                "demosaicked_resolved_hr_imgs",
                T.functional.to_dtype_image(demosaicked_resolved_hr_imgs[:, :, 0, :, :], dtype=torch.uint8, scale=True),
                self.current_epoch,
            )

        # NOTE: the metrics don't know how to handle multiple frames -- they just consider them to be batched.
        demosaicked_resolved_hr_imgs = rearrange(demosaicked_resolved_hr_imgs, "b c f h w -> (b f) c h w")
        demosaicked_hr_imgs = rearrange(demosaicked_hr_imgs, "b c f h w -> (b f) c h w")

        # Incremental update of the metrics
        # self.val_metrics.update(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)

        # Compute the metrics
        metrics: Mapping[str, Tensor] = self.val_metrics(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)
        self.log_dict(metrics, prog_bar=True)
        return metrics["val_dists"]

        # TODO: This should be a callback to prevent this kind of coupling.
        # if self.current_epoch > 0 and psnr < 0.0:
        #     raise optuna.TrialPruned("psnr is negative")

    # def on_validation_epoch_end(self):
    #     self.log_dict(self.val_metrics.compute(), prog_bar=True)
    #     self.val_metrics.reset()

    # def configure_optimizers(self) -> Optimizer:
    #     optimizer: Optimizer = schedulefree.RAdamScheduleFreeClosure(
    #         self.model.parameters(),
    #         lr=self.lr,
    #         weight_decay=self.weight_decay,
    #     )
    #     return optimizer

    def configure_optimizers(self) -> Optimizer:
        optimizer: Optimizer = te.optimizers.FusedAdam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
