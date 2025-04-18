from collections.abc import Mapping
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torchmetrics
import torchmetrics.image
import torchvision.transforms.v2 as T
from einops import rearrange
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from nix_cuda_sr.bayer import bayer_mosaic_unpack
from nix_cuda_sr.multi_frame_raw_sr import MultiFrameRawSR

# Other metrics to consider
# https://lightning.ai/docs/torchmetrics/stable/image/arniqa.html
# https://lightning.ai/docs/torchmetrics/stable/image/error_relative_global_dimensionless_synthesis.html
# https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
# https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html
# https://lightning.ai/docs/torchmetrics/stable/image/mifid.html
# https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html
# https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html
# https://lightning.ai/docs/torchmetrics/stable/image/perceptual_path_length.html
# https://lightning.ai/docs/torchmetrics/stable/image/root_mean_squared_error_using_sliding_window.html
# https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
# https://lightning.ai/docs/torchmetrics/stable/image/universal_image_quality_index.html
# https://lightning.ai/docs/torchmetrics/stable/image/visual_information_fidelity.html


@dataclass(kw_only=True, eq=False)
class WrappedViT(pl.LightningModule):
    # Args
    upscale_factor: int
    n_channels: int

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
        self.model = MultiFrameRawSR()

    def forward(self, test_input: Tensor) -> Tensor:  # type: ignore[override]
        output = self.model(test_input)
        return output

    def training_step(self, imgs: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        mosaicked_hr_imgs, demosaicked_hr_imgs = imgs

        # Crop the images to 256x256
        mosaicked_hr_imgs = T.functional.to_dtype_image(
            T.functional.center_crop(bayer_mosaic_unpack(mosaicked_hr_imgs), [256, 256]),
            dtype=torch.float32,
            scale=True,
        )
        assert mosaicked_hr_imgs.shape == (
            expected := (mosaicked_hr_imgs.shape[0], 4, mosaicked_hr_imgs.shape[2], 256, 256)
        ), f"Output shape mismatch: {expected} vs. {mosaicked_hr_imgs.shape}"
        demosaicked_hr_imgs = T.functional.to_dtype_image(
            T.functional.center_crop(demosaicked_hr_imgs, [256, 256]),
            dtype=torch.float32,
            scale=True,
        )
        assert demosaicked_hr_imgs.shape == (
            expected := (mosaicked_hr_imgs.shape[0], 3, mosaicked_hr_imgs.shape[2], 256, 256)
        ), f"Output shape mismatch: {expected} vs. {demosaicked_hr_imgs.shape}"

        # Create the LR images
        mosaicked_lr_imgs = T.functional.resize_image(mosaicked_hr_imgs, size=[128, 128], antialias=True)
        assert mosaicked_lr_imgs.shape == (
            expected := (mosaicked_hr_imgs.shape[0], 4, mosaicked_hr_imgs.shape[2], 128, 128)
        ), f"Output shape mismatch: {expected} vs. {mosaicked_lr_imgs.shape}"

        # Superresolve
        resolved_hr_imgs: Tensor = self(mosaicked_lr_imgs)
        assert resolved_hr_imgs.shape == demosaicked_hr_imgs.shape, (
            f"Output shape mismatch: {demosaicked_hr_imgs.shape} vs. {resolved_hr_imgs.shape}"
        )

        # NOTE: the metrics don't know how to handle multiple frames -- they just consider them to be batched.
        resolved_hr_imgs = rearrange(resolved_hr_imgs, "b c f h w -> (b f) c h w")
        demosaicked_hr_imgs = rearrange(demosaicked_hr_imgs, "b c f h w -> (b f) c h w")
        metrics: Mapping[str, Tensor] = self.train_metrics(resolved_hr_imgs, demosaicked_hr_imgs)
        self.log_dict(metrics, prog_bar=True)
        return metrics[f"train_{self.loss_metric}"]

    def validation_step(self, imgs: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        mosaicked_hr_imgs, demosaicked_hr_imgs = imgs

        # Crop the images to 256x256
        mosaicked_hr_imgs = T.functional.to_dtype_image(
            T.functional.center_crop(bayer_mosaic_unpack(mosaicked_hr_imgs), [256, 256]),
            dtype=torch.float32,
            scale=True,
        )
        assert mosaicked_hr_imgs.shape == (
            expected := (mosaicked_hr_imgs.shape[0], 4, mosaicked_hr_imgs.shape[2], 256, 256)
        ), f"Output shape mismatch: {expected} vs. {mosaicked_hr_imgs.shape}"
        demosaicked_hr_imgs = T.functional.to_dtype_image(
            T.functional.center_crop(demosaicked_hr_imgs, [256, 256]),
            dtype=torch.float32,
            scale=True,
        )
        assert demosaicked_hr_imgs.shape == (
            expected := (mosaicked_hr_imgs.shape[0], 3, mosaicked_hr_imgs.shape[2], 256, 256)
        ), f"Output shape mismatch: {expected} vs. {demosaicked_hr_imgs.shape}"

        # Create the LR images
        mosaicked_lr_imgs = T.functional.resize_image(mosaicked_hr_imgs, size=[128, 128], antialias=True)
        assert mosaicked_lr_imgs.shape == (
            expected := (mosaicked_hr_imgs.shape[0], 4, mosaicked_hr_imgs.shape[2], 128, 128)
        ), f"Output shape mismatch: {expected} vs. {mosaicked_lr_imgs.shape}"

        # Superresolve
        resolved_hr_imgs: Tensor = self(mosaicked_lr_imgs)
        assert resolved_hr_imgs.shape == demosaicked_hr_imgs.shape, (
            f"Output shape mismatch: {demosaicked_hr_imgs.shape} vs. {resolved_hr_imgs.shape}"
        )

        # NOTE: the metrics don't know how to handle multiple frames -- they just consider them to be batched.
        resolved_hr_imgs = rearrange(resolved_hr_imgs, "b c f h w -> (b f) c h w")
        demosaicked_hr_imgs = rearrange(demosaicked_hr_imgs, "b c f h w -> (b f) c h w")
        metrics: Mapping[str, Tensor] = self.val_metrics(resolved_hr_imgs, demosaicked_hr_imgs)
        self.log_dict(metrics, prog_bar=True)
        return metrics[f"val_{self.loss_metric}"]

    # NOTE: Is this necessary?
    # def on_validation_epoch_end(self):
    #     output = self.val_metrics.compute()
    #     self.log_dict(output)
    #     # remember to reset metrics at the end of the epoch
    #     self.val_metrics.reset()

    def configure_optimizers(self) -> Optimizer:
        optimizer: Optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
