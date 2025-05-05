from collections.abc import Sequence

import torch
import transformer_engine.pytorch as te
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from nix_cuda_sr.mlp_mixer.nn_mlp_mixer import SkipConnection

_PARAMS_DTYPE = torch.bfloat16  # type: ignore[assignment]

# TODO: Is this enough? Or do I need to set this attribute on *all* params,
# regardless of whether they're coming from a Transformer Engine model?
def maybe_fuse_wgrad_accumulation(model: nn.Module) -> nn.Module:
    """
    Fuses the weight gradient accumulation into the model if the model supports it.

    Should only be called on models that support weight gradient accumulation.
    """
    if model.fuse_wgrad_accumulation:
        for param in model.parameters():
            param.grad = None
            setattr(
                param,
                "main_grad",
                torch.zeros_like(
                    param,
                    # dtype=torch.float32,
                    dtype=_PARAMS_DTYPE,
                ),
            )

    return model


class TEMLPBlock(nn.Module):
    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, fuse_wgrad_accumulation: bool
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        self.model = te.ops.Sequential(
            maybe_fuse_wgrad_accumulation(
                te.Linear(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                    params_dtype=_PARAMS_DTYPE,
                )
            ),
            nn.GELU(),
            maybe_fuse_wgrad_accumulation(
                te.Linear(
                    self.ffn_hidden_size,
                    self.hidden_size,
                    fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                    params_dtype=_PARAMS_DTYPE,
                )
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        # for idx, m in enumerate(self.model):
        #     print(f"TEMLPBlock: pre-idx={idx} x.shape={x.shape} m={m}")
        #     x = m(x)
        #     print(f"TEMLPBlock: post-idx={idx} x.shape={x.shape} m={m}")
        # return x
        return self.model(x)


class TEMLPMixerLayer(nn.Module):
    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, fuse_wgrad_accumulation: bool
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        # TODO: Stupid ahead of time hardcoding of values.
        num_frames = 14
        num_channels = 4
        input_size = 80
        patch_size = 20
        num_patches = num_frames * num_channels * (input_size // patch_size) ** 2

        self.model = te.ops.Sequential(
            SkipConnection(
                te.ops.Sequential(
                    te.LayerNorm(
                        self.hidden_size,
                        params_dtype=_PARAMS_DTYPE,
                    ),
                    # NOTE: s = spatial dims (c f h w), e = embed_dim
                    Rearrange("b s e -> b e s"),
                    TEMLPBlock(
                        num_patches, self.ffn_hidden_size, self.fuse_wgrad_accumulation
                    ),
                    Rearrange("b e s -> b s e"),
                ),
            ),
            SkipConnection(
                maybe_fuse_wgrad_accumulation(
                    te.LayerNormMLP(
                        self.hidden_size,
                        self.ffn_hidden_size,
                        fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                        params_dtype=_PARAMS_DTYPE,
                    )
                )
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        # for idx, m in enumerate(self.model):
        #     print(f"TEMLPMixerLayer: pre-idx={idx} x.shape={x.shape} m={m}")
        #     x = m(x)
        #     print(f"TEMLPMixerLayer: post-idx={idx} x.shape={x.shape} m={m}")
        # return x
        return self.model(x)


class TEMLPMixer(nn.Module):
    def __init__(  # noqa: PLR0917
        self,
        # Model parameters
        upscale_factor: int,
        embed_dim: int,
        hidden_sizes: Sequence[int],
        patch_size: int,
        # Model optimization parameters
        compile_model: bool,
        channels_last: bool,
        fuse_wgrad_accumulation: bool,
    ) -> None:
        super().__init__()
        # Model parameters
        self.upscale_factor = 2 * upscale_factor  # doubled to account for demosaicking
        self.embed_dim = embed_dim
        self.hidden_sizes = hidden_sizes
        self.patch_size = patch_size

        # Model optimization parameters
        self.compile_model = compile_model
        self.channels_last = channels_last
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        model: nn.Module = te.ops.Sequential(
            # Linear/MLP requires channel dimension be last.
            # TODO: Number of frames varies, so we cannot include it in the channel dimension at the end since the
            # embedding dimension is fixed.
            Rearrange(
                "b c f (h hp) (w wp) -> b (c f h w) (hp wp)",
                c=4,
                hp=self.patch_size,
                wp=self.patch_size,
            ),
            # Linear transform to embed_dim
            maybe_fuse_wgrad_accumulation(
                te.Linear(
                    self.patch_size**2,
                    self.embed_dim,
                    fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                    params_dtype=_PARAMS_DTYPE,
                )
            ),
            # MLP layers
            # TODO: LucidRains' implementation alternates between a convolution and a linear layer.
            # https://github.com/lucidrains/mlp-mixer-pytorch/blob/bcd54afa1d41afb0911f8c92657907d2ef95c6c6/mlp_mixer_pytorch/mlp_mixer_3d_pytorch.py#L37
            *[
                TEMLPMixerLayer(
                    self.embed_dim, hidden_size, self.fuse_wgrad_accumulation
                )
                for hidden_size in self.hidden_sizes
            ],
            # Flatten the output for the final projection which performs upscaling
            # NOTE: s = other spatial dims (f h w), e = embed_dim
            Rearrange("b (c s) e -> b s (c e)", c=4),
            # Linear transform and re-arrange to upscale and move channel dimension to front
            maybe_fuse_wgrad_accumulation(
                te.Linear(
                    # NOTE: 4 channels in the input image
                    4 * self.embed_dim,
                    # NOTE: 3 channels in the output image
                    3 * (self.upscale_factor * self.patch_size) ** 2,
                    fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                    params_dtype=_PARAMS_DTYPE,
                )
            ),
            Rearrange(
                "b (f h w) (c hu hp wu wp) -> b c f (h hu hp) (w wu wp)",
                # TODO: OH GOD, DOES THIS NEED TO HAPPEN IN FORWARD?
                # NOTE: Recall h and w here are the number of patches, not the number of pixels.
                c=3,
                f=14,  # Assuming RealBSR RAW dataset
                h=4,  # Assuming RealBSR RAW dataset with patch size 20
                hu=self.upscale_factor,
                hp=self.patch_size,
                w=4,  # Assuming RealBSR RAW dataset with patch size 20
                wu=self.upscale_factor,
                wp=self.patch_size,
            ),
        )
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last_3d)  # type: ignore
        if self.compile_model:
            # NOTE: Transformer Engine doesn't support fullgraph=True yet.
            model = torch.compile(
                model, fullgraph=False, dynamic=True, mode="max-autotune"
            )  # type: ignore
        self.model = model

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # for idx, m in enumerate(self.model):
        #     print(f"TEMLPMixer: pre-idx={idx} x.shape={x.shape} m={m}")
        #     x = m(x)
        #     print(f"TEMLPMixer: post-idx={idx} x.shape={x.shape} m={m}")

        # return x.clamp(0.0, 1.0)
        return self.model(x).clamp(0.0, 1.0)
