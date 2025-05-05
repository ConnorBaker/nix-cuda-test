from collections.abc import Mapping, Sequence
from typing import Literal

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.functional.image.dists import DISTSNetwork
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.v2.functional import to_dtype_image
from transformer_engine.pytorch.optimizers import FusedAdam


# TODO: Unclear if this is necessary; seen in LPIPS code.
class NoTrainDISTS(DISTSNetwork):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> "NoTrainDISTS":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)


type Stage = Literal["training", "validation"]


class MLPMixer(pl.LightningModule):
    def __init__(
        self,
        # Model
        model: nn.Module,
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
            "psnr": PeakSignalNoiseRatio(),
        })
        self.metrics: Mapping[Stage, MetricCollection] = {
            stage: metrics.clone(prefix=f"{stage}_") for stage in ("training", "validation")
        }

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
            dists = torch.compile(
                dists, fullgraph=True, dynamic=False, mode="max-autotune"
            )  # type: ignore
        self.dists = dists

    def configure_optimizers(self) -> Optimizer:
        optimizer: Optimizer = FusedAdam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            master_weights=True,
            master_weight_dtype=torch.float32,
            store_param_remainders=True,
            # exp_avg_dtype=torch.uint8,
            # exp_avg_sq_dtype=torch.uint8,
        )
        # scheduler: LRScheduler = CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=10,
        #     T_mult=1,
        #     eta_min=0.0,
        #     last_epoch=-1,
        # )
        return optimizer

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.model(x)

    # Handle the transfer of the batch to the device and the correct memory format.
    def transfer_batch_to_device(
        self, batch: tuple[Tensor, Tensor], device: torch.device, dataloader_idx: int
    ) -> tuple[Tensor, Tensor]:
        lr_unpacked_mosaicked_imgs = batch[0].to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last_3d
            if self.model.channels_last
            else torch.preserve_format,
        )
        hr_demosaicked_img = batch[1].to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last
            if self.model.channels_last
            else torch.preserve_format,
        )
        return lr_unpacked_mosaicked_imgs, hr_demosaicked_img

    def stage_step(
        self, stage: Stage, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        lr_unpacked_mosaicked_imgs, hr_demosaicked_img = batch

        # Superresolve
        # torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/connorbaker/nix-cuda-sr/nix-cuda-sr/nix_cuda_sr/mlp_mixer/te_mlp_mixer.py", line 191, in torch_dynamo_resume_in_forward_at_191
        #     return self.model(x).clamp(0.0, 1.0). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
        hr_resolved_demosaicked_imgs: Tensor = self(lr_unpacked_mosaicked_imgs)

        # NOTE: Take just the first frame, assuming the RealBSR RAW dataset.
        hr_resolved_demosaicked_img: Tensor = hr_resolved_demosaicked_imgs[:, :, 0]

        # Compute dists
        dists: Tensor = self.dists(
            hr_resolved_demosaicked_img,
            hr_demosaicked_img,
            require_grad=stage == "training",
        ).mean()

        # Incremental update of the metrics and logging
        # Because we are using cudagraphs, we need to mark the step before metrics since they use mutation.
        # torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/nix/store/xsqqrgg81ir82b2hnvsi249sdhvna05h-python3.12-torchmetrics-1.7.1/lib/python3.12/site-packages/torchmetrics/metric.py", line 549, in wrapped_func
        #     update(*args, **kwargs)
        #   File "/nix/store/xsqqrgg81ir82b2hnvsi249sdhvna05h-python3.12-torchmetrics-1.7.1/lib/python3.12/site-packages/torchmetrics/image/psnr.py", line 137, in update
        #     self.min_target = torch.minimum(target.min(), self.min_target). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
        # torch.compiler.cudagraph_mark_step_begin()
        self.metrics[stage].update(hr_resolved_demosaicked_img, hr_demosaicked_img)
        self.log(f"{stage}_dists", dists, prog_bar=True)

        # Save the first frame of the first batch to Tensorboard
        if stage == "validation" and batch_idx == 0:
            if self.current_epoch == 0:
                # These don't change between batches, so only log them once
                # TODO: Save a copy of the demoasicked lr imgs to verify visually that functions work as expected.
                self.logger.experiment.add_images(  # type: ignore
                    "hr_demosaicked_img",
                    to_dtype_image(
                        hr_demosaicked_img ** (1 / 2.2), dtype=torch.uint8, scale=True
                    ),
                    self.current_epoch,
                )
            self.logger.experiment.add_images(  # type: ignore
                "hr_resolved_demosaicked_img",
                to_dtype_image(
                    hr_resolved_demosaicked_img ** (1 / 2.2),
                    dtype=torch.uint8,
                    scale=True,
                ),
                self.current_epoch,
            )

        # Use dists as the loss
        return dists

    # NOTE: Required otherwise metrics create memory leaks.
    def on_stage_epoch_end(self, stage: Stage) -> None:
        """Log the metrics at the end of the epoch."""
        self.log_dict(self.metrics[stage].compute(), prog_bar=True)
        self.metrics[stage].reset()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        return self.stage_step("training", batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        self.on_stage_epoch_end("training")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        return self.stage_step("validation", batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self.on_stage_epoch_end("validation")

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer
    ) -> None:  # noqa: PLR6301
        optimizer.zero_grad(set_to_none=True)
