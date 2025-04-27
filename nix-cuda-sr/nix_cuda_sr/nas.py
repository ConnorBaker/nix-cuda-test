from functools import partial
from pathlib import Path

import modelopt.torch.nas as mtn
import torch
import torchvision.transforms.v2 as T
from einops import rearrange
from pytorch_lightning import Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader

import nix_cuda_sr
from nix_cuda_sr.mlp_mixer.mlp_mixer import MLPMixer
from nix_cuda_sr.mlp_mixer.nn_mlp_mixer import NNMLPMixer
from nix_cuda_sr.sr_data_module import SRDataModule


def collect_func(batches: tuple[Tensor, Tensor]) -> Tensor:
    # raise NotImplementedError("This function is not implemented yet.")
    unpacked_mosaicked_hr_imgs, _demosaicked_hr_imgs = batches

    # Create the LR images
    unpacked_mosaicked_lr_imgs = T.functional.resize_image(unpacked_mosaicked_hr_imgs, size=[128, 128], antialias=True)

    return unpacked_mosaicked_lr_imgs.cuda()


def loss_func(dists_metric: nn.Module, demosaicked_resolved_hr_imgs: Tensor, batches: tuple[Tensor, Tensor]) -> Tensor:
    # raise NotImplementedError("This function is not implemented yet.")
    _unpacked_mosaicked_hr_imgs, demosaicked_hr_imgs = batches

    # Metric expects images of shape (N, C, H, W)
    demosaicked_resolved_hr_imgs = rearrange(demosaicked_resolved_hr_imgs, "b c f h w -> (b f) c h w").cuda()
    demosaicked_hr_imgs = rearrange(demosaicked_hr_imgs, "b c f h w -> (b f) c h w").cuda()

    # Compute the loss
    dists: Tensor = dists_metric(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs, require_grad=True)
    dists = dists.mean()

    # Return the loss
    return dists


def score_func(dists_metric: nn.Module, val_dataloader: DataLoader[tuple[Tensor, Tensor]], model: nn.Module) -> Tensor:
    # raise NotImplementedError("This function is not implemented yet.")
    # Get the first batch of data
    batches: tuple[Tensor, Tensor] = next(iter(val_dataloader))

    # Unpack the batches
    unpacked_mosaicked_hr_imgs, demosaicked_hr_imgs = batches
    unpacked_mosaicked_hr_imgs = unpacked_mosaicked_hr_imgs.cuda()
    demosaicked_hr_imgs = demosaicked_hr_imgs.cuda()

    # Create the LR images
    unpacked_mosaicked_lr_imgs = T.functional.resize_image(unpacked_mosaicked_hr_imgs, size=[128, 128], antialias=True)

    # Forward pass through the model
    demosaicked_resolved_hr_imgs: Tensor = model(unpacked_mosaicked_lr_imgs)

    # Metric expects images of shape (N, C, H, W)
    demosaicked_resolved_hr_imgs = rearrange(demosaicked_resolved_hr_imgs, "b c f h w -> (b f) c h w")
    demosaicked_hr_imgs = rearrange(demosaicked_hr_imgs, "b c f h w -> (b f) c h w")

    # Compute the loss
    dists: Tensor = dists_metric(demosaicked_resolved_hr_imgs, demosaicked_hr_imgs, require_grad=False)
    dists = dists.mean()

    # Score is the reciprocal of the loss
    score = 1.0 / dists

    # Return the score
    return score


def main() -> None:
    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=10,
        check_val_every_n_epoch=10,
        precision="bf16",
        strategy="auto",
    )

    impl = NNMLPMixer(
        # Model parameters
        upscale_factor=2,
        embed_dim=128,
        depth=16,
        patch_size=16,
        # Model optimization parameters
        compile_model=False,
        channels_last=False,
    )

    # init the model directly on the device and with parameters in half-precision
    # with trainer.init_module():
    model = MLPMixer(
        model=impl,
        lr=1e-3,
        weight_decay=1e-4,
    ).to(dtype=torch.bfloat16, device="cuda")

    datamodule = SRDataModule(
        root=Path("/home/connorbaker/Pictures/Tensor-256-ARW"),
        batch_size=16,
    )

    trainer.fit(
        model,  # type: ignore[arg-type]
        datamodule=datamodule,
    )

    # Generate the search space for AutoNAS
    converted_impl = mtn.convert(impl, mode="autonas").to(dtype=torch.bfloat16, device="cuda")

    # Looking for a subnet with at most 2 GFLOPs
    search_constraints = {"params": 3_000_000}

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    dists_metric = model.dists.cuda()

    # search_res (dict) contains state_dict / stats of the searcher
    searched_model, search_res = mtn.search(
        model=converted_impl,
        constraints=search_constraints,
        dummy_input=torch.randn(1, 4, 1, 256, 256).to(device="cuda", dtype=torch.bfloat16),
        config={
            # "checkpoint": "modelopt_search_checkpoint.pth",
            "verbose": True,
            "data_loader": train_dataloader,
            "collect_func": collect_func,
            "max_iter_data_loader": 1000,
            "score_func": partial(
                score_func, dists_metric, val_dataloader
            ),  # validation score is used to rank the subnets (value is maximized)
            "loss_func": partial(loss_func, dists_metric),  # loss function for the validation
            # checkpoint to store the search state and resume or re-run the search with different constraint
        },
    )
    # print(f"search_res: {json.dumps(search_res, indent=2, sort_keys=True)}")

    # Save the searched model for further fine-tuning
    # mto.save(searched_model, "modelopt_searched_model.pth")


if __name__ == "__main__":
    nix_cuda_sr._setup()
    main()
