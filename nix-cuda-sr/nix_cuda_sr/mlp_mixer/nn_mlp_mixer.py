import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torchvision.ops import MLP


class SkipConnection(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(self.ffn_hidden_size, self.hidden_size),
        )

        self.model = SkipConnection(mlp)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class NNMLPMixer(nn.Module):
    def __init__(  # noqa: PLR0917
        self,
        # Model parameters
        upscale_factor: int,
        embed_dim: int,
        depth: int,
        patch_size: int,
        # Model optimization parameters
        compile_model: bool,
        channels_last: bool,
    ) -> None:
        super().__init__()
        # Model parameters
        self.upscale_factor = upscale_factor
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size

        # Model optimization parameters
        self.compile_model = compile_model
        self.channels_last = channels_last

        from_patches_to_embed_dim = nn.Linear(
            # NOTE: 4 channels in the input image
            4 * self.patch_size**2,
            self.embed_dim,
        )

        from_embed_dim_to_upscaled_patches = nn.Linear(
            self.embed_dim,
            # NOTE: 3 channels in the output image
            3 * (self.upscale_factor * self.patch_size) ** 2,
        )

        model: nn.Module = nn.Sequential(
            # Linear/MLP requires channel dimension be last.
            # TODO: Number of frames varies, so we cannot include it in the channel dimension at the end since the
            # embedding dimension is fixed.
            Rearrange("b c f (h hp) (w wp) -> b f h w (c hp wp)", c=4, hp=self.patch_size, wp=self.patch_size),
            # Linear transform to embed_dim
            from_patches_to_embed_dim,
            # MLP layers
            # TODO: LucidRains' implementation alternates between a convolution and a linear layer.
            # https://github.com/lucidrains/mlp-mixer-pytorch/blob/bcd54afa1d41afb0911f8c92657907d2ef95c6c6/mlp_mixer_pytorch/mlp_mixer_3d_pytorch.py#L37
            *[
                # NOTE: 4 is an arbitrary expansion factor.
                MLPBlock(self.embed_dim, 4 * self.embed_dim)
                for _ in range(self.depth)
            ],
            # Linear transform and re-arrange to upscale and move channel dimension to front
            from_embed_dim_to_upscaled_patches,
            Rearrange(
                "b f h w (c hu hp wu wp) -> b c f (h hu hp) (w wu wp)",
                hu=self.upscale_factor,
                wu=self.upscale_factor,
                c=3,
                hp=self.patch_size,
                wp=self.patch_size,
            ),
        )
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last_3d)  # type: ignore
        if self.compile_model:
            # NOTE: Transformer Engine doesn't support fullgraph=True yet.
            model = torch.compile(model, fullgraph=False, dynamic=True, mode="max-autotune")  # type: ignore
        self.model = model

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.model(x).clamp(0.0, 1.0)
