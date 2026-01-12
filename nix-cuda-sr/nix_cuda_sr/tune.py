import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import optuna
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner

import nix_cuda_sr
from nix_cuda_sr.mlp_mixer.mlp_mixer import MLPMixer
from nix_cuda_sr.mlp_mixer.te_mlp_mixer import TEMLPMixer
from nix_cuda_sr.realbsr_data_module import RealBSRDataModule
import torch


class MLPArgs(TypedDict):
    embed_dim: int
    hidden_sizes: Sequence[int]
    patch_size: int
    lr: float
    weight_decay: float


def objective(trial: optuna.trial.Trial) -> float:
    args: MLPArgs = MLPArgs(
        embed_dim=trial.suggest_int("embed_dim", 512, 2048, step=512),
        hidden_sizes=[
            trial.suggest_int(f"hidden_size_{i}", 512, 4096, step=512) for i in range(trial.suggest_int("depth", 1, 9))
        ],
        # Patch size must be a divisor of 128
        # patch_size=trial.suggest_categorical("patch_size", [2**i for i in range(3, 6)]),
        patch_size=20,
        lr=trial.suggest_float("lr", 1e-5, 1e-2),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2),
    )

    trainer = Trainer(
        accelerator="auto",
        # accumulate_grad_batches=8,
        # gradient_clip_val=0.75,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=3,
        # check_val_every_n_epoch=2,
        # num_sanity_val_steps=0,
        precision="transformer-engine",
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        strategy="auto",
    )

    datamodule = RealBSRDataModule(
        root=Path("/home/connorbaker/RealBSR/RAW"),
        batch_size=8,
    )

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = MLPMixer(
            model=TEMLPMixer(
                # Model parameters
                upscale_factor=4,
                embed_dim=args["embed_dim"],
                hidden_sizes=args["hidden_sizes"],
                patch_size=args["patch_size"],
                # Model optimization parameters
                compile_model=True,
                fuse_wgrad_accumulation=True,
                channels_last=True,
            ),
            lr=args["lr"],
            weight_decay=args["weight_decay"],
        )

    assert trainer.logger is not None
    trainer.logger.log_hyperparams(args)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["validation_dists"].item()


if __name__ == "__main__":
    # NOTE:
    # Run with export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
    # See https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    nix_cuda_sr._setup()

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name="nix-cuda-sr-optuna",
        storage="sqlite:///nix-cuda-sr-optuna.db",
        direction="minimize",
        load_if_exists=True,
    )

    # Catch CUDA out of memory errors
    study.optimize(objective, n_trials=100, gc_after_trial=True, catch=(RuntimeError,))

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
