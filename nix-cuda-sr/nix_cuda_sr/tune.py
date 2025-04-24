import logging
import sys
from pathlib import Path

import optuna
import torch
from pytorch_lightning.plugins import TransformerEnginePrecision
from pytorch_lightning.trainer.trainer import Trainer
from transformer_engine.common.recipe import DelayedScaling

import nix_cuda_sr
from nix_cuda_sr.sr_data_module import SRDataModule
from nix_cuda_sr.wrapped_te_vit import WrappedViT
from nix_cuda_sr.wrapped_conv_net import WrappedConvNet


def objective(trial: optuna.trial.Trial) -> float:
    hyperparameters = dict(
        # patch_size=trial.suggest_categorical("patch_size", [8, 16, 32]),
        embed_dim=trial.suggest_categorical("embed_dim", [128, 256, 512]),
        # ffn_dim=trial.suggest_categorical("ffn_dim", [128, 256, 512]),
        # num_heads=trial.suggest_categorical("num_heads", [2, 4, 8]),
        # num_layers=trial.suggest_categorical("num_layers", [4, 8]),
        # patch_size=16,  # 32 is too big for 256 images, H_p and W_p were smaller than window size
        patch_size=32,  # 32 is too big for 256 images, H_p and W_p were smaller than window size
        # embed_dim=512,
        # ffn_dim=512,
        # num_heads=8,
        # num_layers=8,
        lr=trial.suggest_float("lr", 1e-5, 1e-2),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2),
    )

    # precision = TransformerEnginePrecision(
    #     weights_dtype=torch.bfloat16,
    #     # NOTE: Both of these require Hopper or newer.
    #     recipe=DelayedScaling(
    #         fp8_dpa=False,
    #         fp8_mha=False,
    #     ),
    #     replace_layers=True,
    # )

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=400,
        # plugins=[precision],
        precision="transformer-engine",
        strategy="auto",
    )

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = WrappedConvNet(
            upscale_factor=2,
            **hyperparameters,
        )
        model = model.to(memory_format=torch.channels_last_3d)

    # NOTE: didn't see a performance improvement with `fuse_wgrad_accumulation` on the 4090.
    # Did see a large decrease in training and validation accuracy.

    # if args.compile:
    #     model = torch.compile(model)  # type: ignore[assignment]

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(
        datamodule=SRDataModule(
            root=Path("/home/connorbaker/Pictures/Tensor-256-ARW"),
            batch_size=16,
        ),
        model=model,  # type: ignore[arg-type]
    )

    return 1 / trainer.callback_metrics["val_psnr"].item()


if __name__ == "__main__":
    # NOTE:
    # Run with export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
    # See https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    nix_cuda_sr._setup()

    # from lightning_fabric.fabric import Fabric  # noqa: PLC0415
    # Fabric.seed_everything(42, workers=True)
    # torch._dynamo.reset()  # type: ignore[no-untyped-call]

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name="nix-cuda-sr-optuna", storage="sqlite:///nix-cuda-sr-optuna.db", direction="minimize"
    )
    study.optimize(objective, n_trials=1000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
