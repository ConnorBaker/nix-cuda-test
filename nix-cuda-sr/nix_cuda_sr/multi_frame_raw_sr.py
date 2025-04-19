import torch
import transformer_engine.pytorch as te  # NVIDIA Transformer Engine
from einops import rearrange, repeat
from torch import Tensor, nn


class BurstPatchEmbedding(nn.Module):
    """
    Convert a burst of 4-channel RAW frames into hierarchical patch embeddings for a transformer.
    Incorporates local window self-attention with relative positional encoding (spatial),
    and adds a temporal frame embedding.
    """

    def __init__(self, patch_size: int = 4, embed_dim: int = 64, window_size: int = 7, num_heads: int = 4):
        """
        Args:
            patch_size (int): Spatial patch size (P x P) to partition each frame.
            embed_dim (int): Embedding dimension for patch tokens (stage 1).
            window_size (int): Window size for local self-attention (spatial).
            num_heads (int): Number of attention heads for multi-head self-attention.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # 1. Patch embedding: use conv to extract patches and project to embed_dim.
        # Input has 4 channels; kernel_size = stride = patch_size gives non-overlapping patches.
        self.patch_embed = nn.Conv2d(in_channels=4, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Frame positional encoding: a learnable embedding for each frame index.
        # (We assume a reasonable maximum number of frames or dynamically generate as needed.)
        self.frame_embedding = nn.Embedding(num_embeddings=100, embedding_dim=embed_dim)
        # ^ num_embeddings can be set to max expected frames; here 100 is an arbitrary large value.

        # 3. Relative position bias for window attention (within a local window):
        M = window_size
        N = M * M  # number of patches in one window
        # Learnable table of size ((2*M-1)*(2*M-1), num_heads) for 2D relative bias.
        self.rel_pos_bias_table = nn.Parameter(torch.zeros((2 * M - 1) * (2 * M - 1), num_heads))
        # Pre-compute index matrix to map patch index pairs to bias table indices:
        coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij"))  # shape (2, M, M)
        coords_flatten = rearrange(coords, "c h w -> c (h w)")  # shape (2, N)
        # Compute relative coordinates (dy, dx) for each pair of patches:
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # shape (2, N, N)
        rel_coords = rearrange(rel_coords, "c x y -> x y c")  # shape (N, N, 2)
        # Shift to non-negative and convert to single index:
        rel_coords[..., 0] += M - 1  # vertical offset
        rel_coords[..., 1] += M - 1  # horizontal offset
        rel_coords[..., 0] *= 2 * M - 1
        rel_position_index = rel_coords[..., 0] + rel_coords[..., 1]  # shape (N, N)
        self.register_buffer("rel_position_index", rel_position_index.long())
        # ^ rel_position_index[p, q] gives index in bias table for relative position between patch p and q.

        # 4. Linear layers for QKV projection and output projection in local attention
        #    (Transformer Engine fused attention will be used for the actual attention computation).
        # TODO(@connorbaker): Check out setting params_dtype to FP8 for TE Linear layers.
        self.qkv = te.Linear(embed_dim, 3 * embed_dim)  # using Transformer Engine Linear for FP8 support
        self.proj = te.Linear(embed_dim, embed_dim)
        self.attn_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = (self.head_dim) ** -0.5  # scale for dot-product attention
        # Fused attention module (Transformer Engine DotProductAttention) for local window self-attention
        self.attn = te.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=self.head_dim,
            attention_dropout=0.0,
            qkv_format="bshd",  # query, key, value are provided as [batch, seq, head, dim] format
            attn_mask_type="no_mask",
        )
        # ^ Using TE's DotProductAttention to leverage fused kernels (FlashAttention/cuDNN) and FP8.
        #   Relative positional bias will be applied via core_attention_bias in forward.

        # 5. Patch merging layer (stage 2): merge 2x2 neighboring patches, doubling channel dim (embed_dim2 = 2*embed_dim).
        self.embed_dim2 = embed_dim * 2
        self.patch_merge_linear = te.Linear(
            4 * embed_dim, self.embed_dim2
        )  # linear projection after concatenating 2x2 patches

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor of shape (B, 4, F, H2, W2) from BayerMosaicUnpack, where H2 = H/2 and W2 = W/2.

        Returns:
            A tensor of patch embeddings with hierarchical structure.
            Shape: (B, N_patches, C_out), where C_out = embed_dim2, and N_patches = F * (H2/patch_size/2) * (W2/patch_size/2).
            (Equivalently, output can be viewed as (B, F, H2/(patch_size*2), W2/(patch_size*2), C_out).)
        """
        B, _, F, H2, W2 = x.shape  # input channels dimension is 4 (already separated Bayer channels)
        # Ensure spatial dimensions are divisible by patch_size and by 2 (for merging stage).
        assert H2 % self.patch_size == 0 and W2 % self.patch_size == 0, (
            f"H2 ({H2}), W2 ({W2}) must be multiples of patch_size ({self.patch_size})"
        )

        # Rearrange input to batch all frames for convolution: (B, 4, F, H2, W2) -> (B*F, 4, H2, W2)
        x_frames = rearrange(x, "b c f h w -> (b f) c h w")

        # 1. Patch embedding via conv:
        patch_feat = self.patch_embed(
            x_frames
        )  # shape: (B*F, embed_dim, H_p, W_p), H_p = H2/patch_size, W_p = W2/patch_size
        H_p, W_p = patch_feat.shape[2], patch_feat.shape[3]
        # Separate batch and frame, and move embed dim to last: (B*F, C, H_p, W_p) -> (B, F, H_p, W_p, C)
        C = self.embed_dim
        patch_feat = rearrange(patch_feat, "(b f) c h w -> b f h w c", b=B, f=F)
        # Now patch_feat[b, f, i, j, :] is the C-dimensional embedding of patch (i, j) in frame f.

        # 2. Add frame positional embeddings:
        frame_indices = torch.arange(F, device=x.device)  # shape (F,)
        frame_pos_emb = self.frame_embedding(frame_indices)  # shape (F, C)
        frame_pos_emb = rearrange(frame_pos_emb, "f c -> 1 f 1 1 c")  # shape (1, F, 1, 1, C)
        patch_feat += frame_pos_emb  # add frame-specific bias to each patch in that frame

        # 3. Window-based self-attention (within each frame) with relative positional bias:
        M = self.window_size
        N = M * M
        # For simplicity, assume H_p and W_p are multiples of M (so each frame can be partitioned into full MxM windows).
        assert H_p % M == 0 and W_p % M == 0, (
            f"For simplicity, H_p ({H_p}) and W_p ({W_p}) should be multiples of window_size ({M})"
        )

        # Partition patches into local windows:
        n_win_h = H_p // M
        n_win_w = W_p // M
        # total_windows = B * F * n_win_h * n_win_w
        # Shape windows as (total_windows, N, C) where N=M*M patches per window:
        windows = rearrange(
            patch_feat, "b f (nh m1) (nw m2) c -> (b f nh nw) (m1 m2) c", b=B, f=F, nh=n_win_h, nw=n_win_w, m1=M, m2=M
        )

        # TODO(@connorbaker): the dimensions here are wrong. the batch dimension is the number of windows, not the batch.

        # Linear projections for all windows at once (compute Q, K, V):
        qkv = self.qkv(windows)  # shape: (total_windows, N, 3*C)
        qkv = rearrange(qkv, "win seq (qkv c) -> qkv win seq c", qkv=3, c=C)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each shape: (total_windows, N, C)

        # Split heads for multi-head attention (then shape as batch-first for TE DotProductAttention):
        q = rearrange(
            q, "w n (h d) -> w n h d", h=self.attn_heads, d=self.head_dim
        )  # (total_windows, N, num_heads, head_dim)
        k = rearrange(k, "w n (h d) -> w n h d", h=self.attn_heads, d=self.head_dim)
        v = rearrange(v, "w n (h d) -> w n h d", h=self.attn_heads, d=self.head_dim)

        # Compute relative positional bias for this window (to be added to attention scores):
        bias_index = rearrange(self.rel_position_index[:N, :N], "p q -> (p q)")
        rel_bias = self.rel_pos_bias_table[bias_index]  # shape: (N*N, num_heads)
        rel_bias = rearrange(rel_bias, "(n1 n2) h -> 1 h n1 n2", n1=N, n2=N)
        # ^ rel_bias is of shape (1, num_heads, N, N) and will broadcast across all windows.

        # Fused scaled dot-product attention using Transformer Engine (with FP8 support):
        attn_out = self.attn(
            q,
            k,
            v,
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=rel_bias,
        )
        # ^ We use core_attention_bias to add the 2D relative positional bias to the attention logits
        #   (post_scale_bias ensures bias is added after scaling by 1/sqrt(d)).
        #   This leverages TE's fused attention kernel (FlashAttention/cuDNN).
        #   If TE did not support passing a bias tensor, we would have to fallback to manual attention computation for
        #   relative position encoding.

        # already (w, N, C)
        attn_out = self.proj(attn_out)  # project back to C dimensions per patch

        # Reshape back from local windows to original patch grid:
        attn_out = rearrange(
            attn_out, "(b f nh nw) (m1 m2) c -> b f (nh m1) (nw m2) c", b=B, f=F, nh=n_win_h, nw=n_win_w, m1=M, m2=M
        )
        # Now attn_out has shape (B, F, H_p, W_p, C), the refined patch features after local window attention.

        # 4. Patch merging stage:
        # Group 2x2 neighboring patches in each frame and merge channels:
        H_merge = H_p // 2
        W_merge = W_p // 2
        assert H_p % 2 == 0 and W_p % 2 == 0, f"H_p ({H_p}), W_p ({W_p}) must be even for patch merging"
        merged = rearrange(attn_out, "b f (hm p1) (wm p2) c -> b f hm wm (p1 p2 c)", hm=H_merge, wm=W_merge, p1=2, p2=2)
        # merged shape: (B, F, H_merge, W_merge, 4*C)
        # Flatten patches and apply linear projection to reduce 4*C -> embed_dim2:
        merged_flat = rearrange(merged, "b f hm wm c4 -> (b f hm wm) c4")
        merged_flat = self.patch_merge_linear(merged_flat)  # shape: (B*F*H_merge*W_merge, embed_dim2)
        merged_out = rearrange(
            merged_flat, "(b f hm wm) c2 -> b f hm wm c2", b=B, f=F, hm=H_merge, wm=W_merge, c2=self.embed_dim2
        )
        # merged_out[b, f, i, j, :] is the embedding vector for merged patch (i,j) in frame f.

        # 5. Output formatting: flatten frames and patches into sequence of length L = F * H_merge * W_merge.
        out = rearrange(merged_out, "b f hm wm c2 -> b (f hm wm) c2")  # shape: (B, N_patches, embed_dim2)
        return out


class MultiFrameRawSR(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        patch_size: int = 4,
        num_heads: int = 8,
        num_transformer_layers: int = 6,
        ffn_dim: int = 256,
    ):
        """
        MultiFrameRawSR: Joint demosaicing and super-resolution transformer for burst RAW images.
        :param embed_dim: Dimension of token embeddings in the transformer.
        :param patch_size: Spatial size of patches for embedding.
        :param num_heads: Number of attention heads in the transformer.
        :param num_transformer_layers: Number of transformer encoder layers.
        :param ffn_dim: Hidden dimension of feed-forward network in transformer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.ffn_dim = ffn_dim

        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert embed_dim % 2 == 0, (
            f"embed_dim ({embed_dim}) must be even for BurstPatchEmbedding (hierarchical embedding halves spatial size)"
        )

        # 1. Burst patch embedding: embed patches from all frames.
        self.patch_embed = BurstPatchEmbedding(
            embed_dim=embed_dim // 2,  # half of final embed_dim due to merging doubling channels
            patch_size=patch_size,
            num_heads=num_heads,
            window_size=8,
        )

        # 2. Transformer backbone (Transformer Engine for FP8):
        # Stack TransformerEngine's TransformerLayer for joint multi-frame feature extraction.
        # Using fused QKV parameters and batch-first attention format for maximal performance on H100.
        self.transformer_layers = te.ops.Sequential(*[
            te.TransformerLayer(
                hidden_size=embed_dim,
                ffn_hidden_size=ffn_dim,
                num_attention_heads=num_heads,
                attn_input_format="bshd",  # use batch-first (B, seq, num_heads, head_dim) internal format
                fuse_qkv_params=True,  # fuse QKV projections into one for efficiency
                self_attn_mask_type="no_mask",  # allow full self-attention (no mask)
                layernorm_epsilon=1e-5,
                layer_number=n + 1,
            )
            for n in range(num_transformer_layers)
        ])

        # Unembedding: ConvTranspose to reverse patch embedding (low-res feature map reconstruction).
        self.patch_unembed = nn.ConvTranspose2d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 3. Output upsampling and reconstruction:
        # We first upsample feature maps and then project to RGB.
        self.toRGB = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)

    def forward(self, burst: Tensor, scale: float = 2.0) -> Tensor:
        """
        Forward pass for burst super-resolution.
        :param burst: Tensor of shape (B, 4, F, H2, W2) from BayerMosaicUnpack, where H2 = H/2 and W2 = W/2.
        :param scale: Float scaling factor for super-resolution (e.g., 1.5 for 1.5x).
        :return: Tensor of shape (B, 3, F, H_out, W_out) with super-resolved RGB images.
        """
        B, _, F, H2, W2 = burst.shape  # burst has 4 channels (R, G1, G2, B) per frame

        # 1. Patch embedding: get patch tokens for transformer (shape: B x N_patches x embed_dim).
        tokens = self.patch_embed(burst)
        H_p = H2 // self.patch_size // 2  # patches per frame along height after merging (H2/patch_size/2)
        W_p = W2 // self.patch_size // 2  # patches per frame along width after merging
        N_patches_per_frame = H_p * W_p
        N_patches = F * N_patches_per_frame
        assert tokens.shape == (B, N_patches, self.embed_dim), (
            f"tokens shape mismatch: {(B, N_patches, self.embed_dim)} vs. {tokens.shape}"
        )

        # 2. Transformer encoding (joint attention across all frame tokens)
        # Using batch-first format for TransformerLayer input (attn_input_format='bshd'), so no need to transpose tokens.
        encoded = self.transformer_layers(tokens)  # shape: (B, N_patches, embed_dim)

        # After encoding, `encoded` shape is (B, N_patches, embed_dim). Now reshape tokens back to spatial feature maps.
        # 3. Reshape encoded tokens back to frame-wise feature maps (low-res):
        # Separate frames and patch grid, and put embed_dim as channel:
        encoded = rearrange(encoded, "b (f hp wp) c -> b f c hp wp", f=F, hp=H_p, wp=W_p)
        # Unpatchify: convert low-res patch tokens back to a spatial feature map per frame.
        lowres_feat = rearrange(encoded, "b f c hp wp -> (b f) c hp wp")
        feature_map = self.patch_unembed(lowres_feat)  # shape: (B*F, embed_dim, H2/2, W2/2)
        feature_map = rearrange(feature_map, "(b f) c h w -> b f c h w", b=B, f=F, h=H2 // 2, w=W2 // 2)

        # 4. Meta-upsampling to high resolution and RGB reconstruction:
        h_out = int(H2 * scale)
        w_out = int(W2 * scale)
        grid_y = torch.linspace(0, H2 // 2 - 1, h_out, device=feature_map.device) / ((H2 // 2) - 1) * 2 - 1
        grid_x = torch.linspace(0, W2 // 2 - 1, w_out, device=feature_map.device) / ((W2 // 2) - 1) * 2 - 1
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)  # shape: (h_out, w_out, 2)
        grid = repeat(grid, "i j xy -> b i j xy", b=B)  # repeat grid for all batches

        out_frames = []
        for f_idx in range(F):
            feat = feature_map[:, f_idx]  # (B, embed_dim, H2/2, W2/2) for frame f_idx
            # Upsample feature map by 'scale' using bilinear grid sampling (placeholder for meta-upscale):
            highres_feat = nn.functional.grid_sample(feat, grid, mode="bilinear", align_corners=True)
            # Convert high-res features to RGB image:
            rgb = self.toRGB(highres_feat)  # shape: (B, 3, h_out, w_out)
            out_frames.append(rgb)
        # Stack outputs for all frames:
        output = torch.stack(out_frames, dim=2)  # shape: (B, 3, F, H_out, W_out)
        return output
