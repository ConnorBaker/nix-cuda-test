import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
import transformer_engine.pytorch as te
from einops import rearrange
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.functional.image.dists import DISTSNetwork
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from nix_cuda_sr.mlp_mixer.nn_mlp_mixer import NNMLPMixer
from nix_cuda_sr.mlp_mixer.te_mlp_mixer import TEMLPMixer


# TODO: Unclear if this is necessary; seen in LPIPS code.
class NoTrainDISTS(DISTSNetwork):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> "NoTrainDISTS":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)


class SkipConnection(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, fuse_wgrad_accumulation: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        # Declare the MLP layer and optionally fuse the weight gradient accumulation.
        mlp = te.LayerNormMLP(
            self.hidden_size,
            self.ffn_hidden_size,
            fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
        )
        if self.fuse_wgrad_accumulation:
            for param in mlp.parameters():
                param.grad = None
                param.__setattr__("main_grad", torch.zeros_like(param, dtype=torch.float32))

        self.model = SkipConnection(mlp)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MLPMixer(pl.LightningModule):
    def __init__(
        self,
        # Model
        model: NNMLPMixer | TEMLPMixer,
        # Optimizer parameters
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()

        # Model
        self.model = model

        # Optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # NOTE: (from https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
        # Modular metrics contain internal states that should belong to only one DataLoader. In case you are using
        # multiple DataLoaders, it is recommended to initialize a separate modular metric instances for each DataLoader
        # and use them separately. The same holds for using separate metrics for training, validation and testing.
        metrics = MetricCollection({
            "ssim": StructuralSimilarityIndexMeasure(),
            "msssim": MultiScaleStructuralSimilarityIndexMeasure(),
            "psnr": PeakSignalNoiseRatio(),
        })
        self.train_metrics = metrics.clone(prefix="train_").eval()
        self.val_metrics = metrics.clone(prefix="val_").eval()

        # TODO: Optimize the model with TensorRT.
        # NOTE: Since there's no internal state, we can use the same model for training and validation.
        dists: nn.Module = NoTrainDISTS(load_weights=True)
        dists = dists.eval()
        dists = dists.requires_grad_(False)
        if self.model.channels_last:
            dists = dists.to(memory_format=torch.channels_last)  # type: ignore
        if self.model.compile_model:
            # NOTE: For some reason, cannot use dynamic=True: we get an error "cannot determine truth value of
            # Relational"
            dists = torch.compile(dists, fullgraph=True, dynamic=False, mode="max-autotune")  # type: ignore
        self.dists = dists

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.model(x)

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

        # Incremental update of the metrics
        self.train_metrics.update(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)

        # Compute and log lpips and dists
        # NOTE: Must normalize images with LPIPS because it expects images to be in the range [-1, 1] and ours is
        # in the range [0, 1].
        # lpips = self.train_lpips(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs, normalize=True)
        dists: Tensor = self.dists(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs, require_grad=True)
        dists = dists.mean()
        self.log("train_dists", dists, prog_bar=True)

        # Use dists as the loss
        return dists

    # NOTE: Required otherwise metrics create memory leaks.
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

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
                self.logger.experiment.add_images(  # type: ignore
                    "demosaicked_hr_imgs",
                    T.functional.to_dtype_image(demosaicked_hr_imgs[:, :, 0, :, :], dtype=torch.uint8, scale=True),
                    self.current_epoch,
                )
            self.logger.experiment.add_images(  # type: ignore
                "demosaicked_resolved_hr_imgs",
                T.functional.to_dtype_image(demosaicked_resolved_hr_imgs[:, :, 0, :, :], dtype=torch.uint8, scale=True),
                self.current_epoch,
            )

        # NOTE: the metrics don't know how to handle multiple frames -- they just consider them to be batched.
        demosaicked_resolved_hr_imgs = rearrange(demosaicked_resolved_hr_imgs, "b c f h w -> (b f) c h w")
        demosaicked_hr_imgs = rearrange(demosaicked_hr_imgs, "b c f h w -> (b f) c h w")

        # Incremental update of the metrics
        self.val_metrics.update(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs)

        # Compute and log dists
        dists: Tensor = self.dists(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs, require_grad=True)
        dists = dists.mean()
        self.log("val_dists", dists, prog_bar=True)

        # Use dists as the loss
        return dists

        # TODO: This should be a callback to prevent this kind of coupling.
        # if self.current_epoch > 0 and psnr < 0.0:
        #     raise optuna.TrialPruned("psnr is negative")

    # NOTE: Required otherwise metrics create memory leaks.
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self) -> Optimizer:
        optimizer: Optimizer = te.optimizers.FusedAdam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
