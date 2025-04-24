from collections.abc import Mapping
from dataclasses import dataclass, field

import optuna
import pytorch_lightning as pl
import schedulefree
import torch
import torchmetrics
import torchmetrics.image
import torchvision.transforms.v2 as T
import transformer_engine.pytorch as te
from einops import rearrange
from torch import Tensor
from torch.optim.optimizer import Optimizer

from nix_cuda_sr.multi_frame_raw_sr import MultiFrameRawSR

# def prep_hr_img(t: Tensor) -> Tensor:
#     # t = T.functional.resize_image(t, size=[300, 400], antialias=True)
#     # t = T.functional.center_crop_image(t, [256, 256])
#     t = T.functional.to_dtype_image(t, dtype=torch.float16, scale=True)
#     return t


@dataclass(kw_only=True, eq=False)
class WrappedViT(pl.LightningModule):
    # Args
    upscale_factor: int
    embed_dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    patch_size: int

    # Optimizer args
    lr: float
    weight_decay: float

    # Non-args
    loss_metric: str = field(init=False)
    train_metrics: torchmetrics.MetricCollection = field(init=False)
    val_metrics: torchmetrics.MetricCollection = field(init=False)
    model: MultiFrameRawSR = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        # NOTE: (from https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
        # Modular metrics contain internal states that should belong to only one DataLoader. In case you are using
        # multiple DataLoaders, it is recommended to initialize a separate modular metric instances for each DataLoader
        # and use them separately. The same holds for using separate metrics for training, validation and testing.
        metrics = torchmetrics.MetricCollection({
            "dists": torchmetrics.image.DeepImageStructureAndTextureSimilarity(),
            "psnr": torchmetrics.image.PeakSignalNoiseRatio(),
            "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(),
            "msssim": torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(),
        })
        self.loss_metric = "dists"
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.model = MultiFrameRawSR(
            embed_dim=self.embed_dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            patch_size=self.patch_size,
        )

    def forward(self, test_input: Tensor) -> Tensor:  # type: ignore[override]
        output = self.model(test_input)
        return output

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
        return metrics[f"train_{self.loss_metric}"]

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
        metrics: Mapping[str, Tensor] = self.val_metrics(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)
        self.log_dict(metrics, prog_bar=True)

        # TODO: This should be a callback to prevent this kind of coupling.
        if self.current_epoch > 0:
            for metric in ["psnr", "ssim", "msssim", "dists"]:
                if metrics[f"val_{metric}"] < 0.0:
                    raise optuna.TrialPruned(f"{metric} is negative")

        return metrics[f"val_{self.loss_metric}"]

    # NOTE: Is this necessary?
    def on_validation_epoch_end(self):
        output = self.val_metrics.compute()
        self.log_dict(output)
        # remember to reset metrics at the end of the epoch
        self.val_metrics.reset()

    # def configure_optimizers(self) -> tuple[list[Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    #     optimizer: Optimizer = te.optimizers.FusedAdam(
    #         self.model.parameters(),
    #         lr=self.lr,
    #         weight_decay=self.weight_decay,
    #         # exp_avg_dtype=torch.float16,
    #         # exp_avg_sq_dtype=torch.float16,
    #     )
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer, max_lr=1e-3, total_steps=int(self.trainer.estimated_stepping_batches)
    #     )
    #     return [optimizer], [scheduler]

    def configure_optimizers(self) -> Optimizer:
        optimizer: Optimizer = schedulefree.RAdamScheduleFreeClosure(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
