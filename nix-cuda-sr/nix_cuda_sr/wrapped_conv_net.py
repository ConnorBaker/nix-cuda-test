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


class ConvNet(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int, upscale_factor: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

        self.model = te.ops.Sequential(
            nn.Conv3d(
                in_channels=4,
                out_channels=self.embed_dim,
                kernel_size=(1, self.patch_size, self.patch_size),
                stride=(1, self.patch_size, self.patch_size),
            ),
            # Linear/MLP requires channel dimension be last.
            Rearrange("b c f h w -> b f h w c"),
            nn.GELU(),
            te.LayerNormMLP(
                self.upscale_factor**0 * self.embed_dim,
                self.upscale_factor**1 * self.embed_dim,
            ),
            nn.GELU(),
            te.Linear(
                self.upscale_factor**0 * self.embed_dim,
                self.upscale_factor**1 * self.embed_dim,
            ),
            nn.GELU(),
            te.LayerNormMLP(
                self.upscale_factor**1 * self.embed_dim,
                self.upscale_factor**2 * self.embed_dim,
            ),
            nn.GELU(),
            te.Linear(
                self.upscale_factor**1 * self.embed_dim,
                self.upscale_factor**2 * self.embed_dim,
            ),
            # PixelShuffle requires channel dimension be next to H and W dimensions.
            Rearrange("b f h w c -> b f c h w"),
            # Scale by the upscale factor
            nn.GELU(),
            nn.PixelShuffle(self.upscale_factor),
            Rearrange("b f c h w -> b c f h w"),
            # Unpatchify
            nn.GELU(),
            nn.ConvTranspose3d(
                in_channels=self.embed_dim,
                out_channels=3,
                kernel_size=(1, self.patch_size, self.patch_size),
                stride=(1, self.patch_size, self.patch_size),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class WrappedConvNet(pl.LightningModule):
    def __init__(
        self,
        upscale_factor: int,
        embed_dim: int,
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

        # NOTE: (from https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
        # Modular metrics contain internal states that should belong to only one DataLoader. In case you are using
        # multiple DataLoaders, it is recommended to initialize a separate modular metric instances for each DataLoader
        # and use them separately. The same holds for using separate metrics for training, validation and testing.
        # metrics = torchmetrics.MetricCollection({
        #     "dists": torchmetrics.image.DeepImageStructureAndTextureSimilarity(),
        #     "psnr": torchmetrics.image.PeakSignalNoiseRatio(),
        #     "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(),
        #     "msssim": torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(),
        # })
        # self.loss_metric = "dists"
        # self.train_metrics = metrics.clone(prefix="train_")
        # self.val_metrics = metrics.clone(prefix="val_")

        self.model = ConvNet(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            upscale_factor=self.upscale_factor,
        )

    @torch.compile(dynamic=True, mode="max-autotune")
    def forward(self, test_input: Tensor) -> Tensor:  # type: ignore[override]
        return self.model(test_input)

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
        psnr = torchmetrics.functional.image.peak_signal_noise_ratio(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)
        self.log("train_psnr", psnr, prog_bar=True)
        return 1 / psnr  # NOTE: this is a loss, so we want to minimize it.
        # metrics: Mapping[str, Tensor] = self.train_metrics(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)
        # self.log_dict(metrics, prog_bar=True)
        # return 1 / metrics["train_psnr"]  # NOTE: this is a loss, so we want to minimize it.

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
        psnr = torchmetrics.functional.image.peak_signal_noise_ratio(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)
        self.log("val_psnr", psnr, prog_bar=True)
        # metrics: Mapping[str, Tensor] = self.val_metrics(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)
        # self.log_dict(metrics, prog_bar=True)

        # TODO: This should be a callback to prevent this kind of coupling.
        if self.current_epoch > 0 and psnr < 0.0:
            raise optuna.TrialPruned("psnr is negative")

        return 1 / psnr  # NOTE: this is a loss, so we want to minimize it.

    # NOTE: Is this necessary?
    # def on_validation_epoch_end(self):
    #     output = self.val_metrics.compute()
    #     self.log_dict(output)
    #     # remember to reset metrics at the end of the epoch
    #     self.val_metrics.reset()

    def configure_optimizers(self) -> Optimizer:
        optimizer: Optimizer = schedulefree.RAdamScheduleFreeClosure(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
