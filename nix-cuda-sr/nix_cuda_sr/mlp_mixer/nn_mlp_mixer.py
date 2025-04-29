import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class SkipConnection(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


class NNMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, self.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(self.ffn_hidden_size, self.hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class NNMLPMixerLayer(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        # TODO: Stupid ahead of time hardcoding of values.
        num_frames = 14
        num_channels = 4
        input_size = 80
        patch_size = 20
        num_patches = num_frames * num_channels * (input_size // patch_size) ** 2

        self.model = nn.Sequential(
            SkipConnection(
                nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    # NOTE: s = spatial dims (c f h w), e = embed_dim
                    Rearrange("b s e -> b e s"),
                    NNMLPBlock(num_patches, self.ffn_hidden_size),
                    Rearrange("b e s -> b s e"),
                ),
            ),
            SkipConnection(
                nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    NNMLPBlock(self.hidden_size, self.ffn_hidden_size),
                ),
            ),
        )

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
        self.upscale_factor = 2 * upscale_factor  # doubled to account for demosaicking
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size

        # Model optimization parameters
        self.compile_model = compile_model
        self.channels_last = channels_last

        model: nn.Module = nn.Sequential(
            # Linear/MLP requires channel dimension be last.
            # TODO: Number of frames varies, so we cannot include it in the channel dimension at the end since the
            # embedding dimension is fixed.
            Rearrange("b c f (h hp) (w wp) -> b (c f h w) (hp wp)", c=4, hp=self.patch_size, wp=self.patch_size),
            # Linear transform to embed_dim
            nn.Linear(
                self.patch_size**2,
                self.embed_dim,
            ),
            # MLP layers
            # TODO: LucidRains' implementation alternates between a convolution and a linear layer.
            # https://github.com/lucidrains/mlp-mixer-pytorch/blob/bcd54afa1d41afb0911f8c92657907d2ef95c6c6/mlp_mixer_pytorch/mlp_mixer_3d_pytorch.py#L37
            *[
                # NOTE: 4 is an arbitrary expansion factor.
                NNMLPMixerLayer(self.embed_dim, 4 * self.embed_dim)
                for _ in range(self.depth)
            ],
            # Flatten the output for the final projection which performs upscaling
            # NOTE: s = other spatial dims (f h w), e = embed_dim
            Rearrange("b (c s) e -> b s (c e)", c=4),
            # Linear transform and re-arrange to upscale and move channel dimension to front
            nn.Linear(
                # NOTE: 4 channels in the input image
                4 * self.embed_dim,
                # NOTE: 3 channels in the output image
                3 * (self.upscale_factor * self.patch_size) ** 2,
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
            model = torch.compile(model, fullgraph=True, dynamic=False, mode="max-autotune")  # type: ignore
        self.model = model

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.model(x).clamp(0.0, 1.0)
