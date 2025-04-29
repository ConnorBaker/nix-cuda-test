import logging
import sys
from pathlib import Path

import optuna
from pytorch_lightning.trainer.trainer import Trainer

import nix_cuda_sr
from nix_cuda_sr.mlp_mixer.mlp_mixer import MLPMixer
from nix_cuda_sr.mlp_mixer.te_mlp_mixer import TEMLPMixer
from nix_cuda_sr.sr_data_module import SRDataModule


def objective(trial: optuna.trial.Trial) -> float:
    hyperparameters = dict(
        embed_dim=trial.suggest_int("embed_dim", 16, 512, step=16),
        depth=trial.suggest_int("depth", 1, 5),
        # Patch size must be a divisor of 128
        patch_size=trial.suggest_categorical("patch_size", [2**i for i in range(3, 6)]),
        lr=trial.suggest_float("lr", 1e-5, 1e-2),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2),
    )

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        deterministic=False,
        devices="auto",
        max_epochs=100,
        check_val_every_n_epoch=100,
        precision="transformer-engine",
        strategy="auto",
    )

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = MLPMixer(
            model=TEMLPMixer(
                # Model parameters
                upscale_factor=2,
                embed_dim=hyperparameters["embed_dim"],
                depth=hyperparameters["depth"],
                patch_size=hyperparameters["patch_size"],
                # Model optimization parameters
                compile_model=True,
                fuse_wgrad_accumulation=True,
                channels_last=True,
            ),
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
        )

        datamodule = SRDataModule(
            root=Path("/home/connorbaker/Pictures/Tensor-256-ARW"),
            batch_size=16,
        )

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(
        model,  # type: ignore[arg-type]
        datamodule=datamodule,
    )

    return trainer.callback_metrics["val_dists"].item()


if __name__ == "__main__":
    # NOTE:
    # Run with export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
    # See https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    nix_cuda_sr._setup()

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name="nix-cuda-sr-optuna", storage="sqlite:///nix-cuda-sr-optuna.db", direction="minimize"
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
