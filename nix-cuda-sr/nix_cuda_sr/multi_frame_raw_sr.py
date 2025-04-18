import torch
from einops import rearrange
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
        # ^ num_embeddings can be set to max expected frames; here 100 is arbitrary large.

        # 3. Relative position bias for window attention:
        # Create a parameter table for 2D relative position bias within a window.
        M = window_size
        N = M * M  # number of patches in a window
        # Table of size (2*M-1)*(2*M-1) for each head (learnable biases).
        self.rel_pos_bias_table = nn.Parameter(torch.zeros((2 * M - 1) * (2 * M - 1), num_heads))
        # Pre-compute an index matrix to map each pair of patch indices in a window to a bias table index.
        coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij"))  # shape (2, M, M)
        assert coords.shape == (expected := (2, M, M)), f"coords shape mismatch: {expected} vs. {coords.shape}"
        coords_flatten = coords.view(2, -1)  # shape (2, N)
        assert coords_flatten.shape == (expected := (2, N)), (
            f"coords_flatten shape mismatch: {expected} vs. {coords_flatten.shape}"
        )
        # Compute relative coordinates (dy, dx) for each pair of patches in the window.
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # shape (2, N, N)
        assert rel_coords.shape == (expected := (2, N, N)), (
            f"rel_coords shape mismatch: {expected} vs. {rel_coords.shape}"
        )
        rel_coords = rel_coords.permute(1, 2, 0)  # shape (N, N, 2)
        assert rel_coords.shape == (expected := (N, N, 2)), (
            f"rel_coords permute shape mismatch: {expected} vs. {rel_coords.shape}"
        )
        # Shift values to make them non-negative:
        rel_coords[..., 0] += M - 1  # vertical offset range: [0, 2*M-2]
        rel_coords[..., 1] += M - 1  # horizontal offset range: [0, 2*M-2]
        rel_coords[..., 0] *= 2 * M - 1  # multiply y by (2*M-1) for unique indexing
        rel_position_index = rel_coords[..., 0] + rel_coords[..., 1]  # shape (N, N)
        assert rel_position_index.shape == (expected := (N, N)), (
            f"rel_position_index shape mismatch: {expected} vs. {rel_position_index.shape}"
        )
        self.register_buffer("rel_position_index", rel_position_index.long())
        # ^ rel_position_index[p, q] is the index in the bias table for relative position between patch p and q.

        # 4. Linear layers for QKV projection and output projection in attention.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)  # project token features to Q, K, V
        self.proj = nn.Linear(embed_dim, embed_dim)  # project output of attention back to embed_dim
        self.attn_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = (self.head_dim) ** -0.5  # scaling factor for dot-product attention

        # 5. Patch merging layer (stage 2): merge 2x2 neighboring patches to reduce resolution.
        # After merging, channel dimension will be doubled (embed_dim2 = 2 * embed_dim).
        self.embed_dim2 = embed_dim * 2
        # Linear layer to project concatenated 4*embed_dim features to embed_dim2.
        self.patch_merge_linear = nn.Linear(4 * embed_dim, self.embed_dim2)

    def forward(self, x) -> Tensor:
        """
        x: Tensor of shape (B, 4, F, H2, W2) from BayerMosaicUnpack, where H2 = H/2 and W2 = W/2.

        Returns:
            A tensor of patch embeddings with hierarchical structure.
            Shape: (B, N_patches, C_out), where C_out = embed_dim2, and N_patches = F * (H2/patch_size/2) * (W2/patch_size/2).
            (Alternatively, the output can be viewed as (B, F, H2/(patch_size*2), W2/(patch_size*2), C_out) in 5D form.)
        """
        B, _, F, H2, W2 = x.shape  # note: input channels dimension is 4 (already separated Bayer channels)
        # Ensure spatial dimensions are divisible by patch_size and by 2 (for merging).
        assert H2 % self.patch_size == 0 and W2 % self.patch_size == 0, (
            f"H2 ({H2}), W2 ({W2}) must be multiples of patch_size ({self.patch_size})"
        )

        # Rearrange input to batch of frames for convolution:
        # (B, 4, F, H2, W2) -> (B*F, 4, H2, W2)
        x_frames = rearrange(x, "b c f h w -> (b f) c h w", b=B, c=4, f=F, h=H2, w=W2)

        # 1. Patch embedding via conv:
        # After conv: shape = (B*F, embed_dim, H_p, W_p), where H_p = H2/patch_size, W_p = W2/patch_size.
        patch_feat = self.patch_embed(x_frames)  # project each patch to embed_dim features
        assert patch_feat.shape == (
            expected := (B * F, self.embed_dim, H2 // self.patch_size, W2 // self.patch_size)
        ), f"patch_feat shape mismatch: {expected} vs. {patch_feat.shape}"
        # Reshape to separate batch and frame dimensions again, and put embed_dim as last dim:
        # (B*F, C, H_p, W_p) -> (B, F, H_p, W_p, C)
        C = self.embed_dim
        H_p, W_p = patch_feat.shape[2], patch_feat.shape[3]
        patch_feat = rearrange(patch_feat, "(b f) c h w -> b f h w c", b=B, f=F, c=C, h=H_p, w=W_p)
        # Now patch_feat[b, f, i, j, :] is the embed_dim-dimensional feature of patch (i,j) in frame f.

        # 2. Add frame positional embeddings:
        # Get embeddings for frame indices 0..F-1 and add to patch features.
        frame_indices = torch.arange(F, device=x.device)  # shape (F,)
        assert frame_indices.shape == (expected := (F,)), (
            f"frame_indices shape mismatch: {expected} vs. {frame_indices.shape}"
        )
        frame_pos_emb = self.frame_embedding(frame_indices)  # shape (F, C)
        assert frame_pos_emb.shape == (expected := (F, C)), (
            f"frame_pos_emb shape mismatch: {expected} vs. {frame_pos_emb.shape}"
        )
        # reshape to (1, F, 1, 1, C) and broadcast over spatial patch grid:
        frame_pos_emb = rearrange(frame_pos_emb, "f c -> 1 f 1 1 c", f=F, c=C)  # shape (1, F, 1, 1, C)
        patch_feat += frame_pos_emb  # add the frame-specific bias to every patch in that frame

        # 3. Window-based self-attention (within each frame) with relative positional bias:
        # We will process each frame's patches in local windows of size window_size x window_size.
        M = self.window_size
        N = M * M  # number of patches in a window
        # For simplicity, assume H_p and W_p are multiples of M (if not, windows at edges can be smaller or padded).
        assert H_p % M == 0 and W_p % M == 0, (
            f"For simplicity, H_p ({H_p}) and W_p ({W_p}) should be multiples of window_size ({M})"
        )

        # Prepare to apply attention window by window.
        # - total_windows = B * F * n_win_h * n_win_w
        # - each window has N patches and feature dim C
        n_win_h = H_p // M
        n_win_w = W_p // M
        total_windows = B * F * n_win_h * n_win_w
        windows = rearrange(
            patch_feat,
            "b f (n_win_h m1) (n_win_w m2) c -> (b f n_win_h n_win_w) (m1 m2) c",
            b=B,
            f=F,
            n_win_h=n_win_h,
            m1=M,
            n_win_w=n_win_w,
            m2=M,
            c=C,
        )

        # Linear projections for Q, K, V for all windows at once:
        # windows shape (total_windows, N, C) -> qkv shape (total_windows, N, 3*C)
        qkv = self.qkv(windows)  # project to queries, keys, values
        assert qkv.shape == (expected := (total_windows, N, 3 * C)), f"qkv shape mismatch: {expected} vs. {qkv.shape}"
        qkv = rearrange(qkv, "n_win n (qkv c) -> qkv n_win n c", n_win=total_windows, n=N, qkv=3, c=C)
        # Now qkv is (3, total_windows, N, C). Split into Q, K, V:
        q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape (total_windows, N, C)
        # Split heads: reshape C = num_heads * head_dim
        q = q.view(total_windows, N, self.attn_heads, self.head_dim).transpose(1, 2)
        k = k.view(total_windows, N, self.attn_heads, self.head_dim).transpose(1, 2)
        v = v.view(total_windows, N, self.attn_heads, self.head_dim).transpose(1, 2)
        # Now Q, K, V have shape (total_windows, num_heads, N, head_dim)

        # Compute scaled dot-product attention for each window:
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # shape (total_windows, num_heads, N, N)
        assert attn_scores.shape == (expected := (total_windows, self.attn_heads, N, N)), (
            f"attn_scores shape mismatch: {expected} vs. {attn_scores.shape}"
        )
        # Add relative position bias:
        # Use precomputed index matrix to get bias for each relative position pair.
        bias_index = self.rel_position_index[:N, :N].reshape(-1)  # shape (N*N,)
        assert bias_index.shape == (expected := (N * N,)), (
            f"bias_index shape mismatch: {expected} vs. {bias_index.shape}"
        )
        rel_bias = self.rel_pos_bias_table[bias_index]  # shape (N*N, num_heads)
        assert rel_bias.shape == (expected := (N * N, self.attn_heads)), (
            f"rel_bias shape mismatch: {expected} vs. {rel_bias.shape}"
        )
        rel_bias = rearrange(rel_bias, "(n1 n2) num_heads -> num_heads n1 n2", n1=N, n2=N, num_heads=self.attn_heads)
        attn_scores += rel_bias.unsqueeze(0)  # broadcast to all windows (adds to each head's scores)

        # Softmax and weighted sum:
        attn_prob = attn_scores.softmax(dim=-1)  # shape (total_windows, num_heads, N, N)
        assert attn_prob.shape == (expected := (total_windows, self.attn_heads, N, N)), (
            f"attn_prob shape mismatch: {expected} vs. {attn_prob.shape}"
        )
        attn_out = attn_prob @ v  # shape (total_windows, num_heads, N, head_dim)
        assert attn_out.shape == (expected := (total_windows, self.attn_heads, N, self.head_dim)), (
            f"attn_out shape mismatch: {expected} vs. {attn_out.shape}"
        )
        attn_out = attn_out.transpose(1, 2).reshape(total_windows, N, C)  # (total_windows, N, C)
        attn_out = self.proj(attn_out)  # linear projection of attention outputs
        assert attn_out.shape == (expected := (total_windows, N, C)), (
            f"attn_out after proj shape mismatch: {expected} vs. {attn_out.shape}"
        )

        # Reshape back from windows to original patch grid:
        # attn_out shape: (total_windows, N, C)
        # Inverse of window partition:
        attn_out = attn_out.view(B * F, n_win_h, n_win_w, M, M, C)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous()  # shape (B*F, n_win_h, M, n_win_w, M, C)
        assert attn_out.shape == (expected := (B * F, n_win_h, M, n_win_w, M, C)), (
            f"attn_out permute shape mismatch: {expected} vs. {attn_out.shape}"
        )
        attn_out = attn_out.view(B, F, H_p, W_p, C)  # shape (B, F, H_p, W_p, C)

        # 4. Patch merging stage:
        # Group 2x2 neighboring patches in each frame to merge.
        # Input attn_out: (B, F, H_p, W_p, C). We assume H_p, W_p are even.
        assert H_p % 2 == 0 and W_p % 2 == 0, f"H_p ({H_p}), W_p ({W_p}) must be even for patch merging"
        H_merge = H_p // 2
        W_merge = W_p // 2
        # Reshape to group 2x2 patches:
        # new shape: (B, F, H_merge, 2, W_merge, 2, C)
        merged = attn_out.view(B, F, H_merge, 2, W_merge, 2, C)
        # Move the 2x2 grouping into the last dimension by concatenation:
        merged = merged.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # (B,F,H_merge,W_merge,2,2,C)
        assert merged.shape == (expected := (B, F, H_merge, W_merge, 2, 2, C)), (
            f"merged shape mismatch: {expected} vs. {merged.shape}"
        )
        merged = merged.view(B, F, H_merge, W_merge, 4 * C)  # (B,F,H_merge,W_merge,4*C)
        # Apply linear projection to reduce 4*C -> embed_dim2:
        merged = merged.view(B * F * H_merge * W_merge, 4 * C)  # flatten patches for linear layer
        merged = self.patch_merge_linear(merged)  # shape (B*F*H_merge*W_merge, embed_dim2)
        assert merged.shape == (expected := (B * F * H_merge * W_merge, self.embed_dim2)), (
            f"merged after linear shape mismatch: {expected} vs. {merged.shape}"
        )
        merged = merged.view(B, F, H_merge, W_merge, self.embed_dim2)
        # Now merged[b, f, i, j, :] is the embedding vector for the merged patch at (i,j) in frame f.

        # 5. Output formatting:
        # We can output as a flattened sequence of patch tokens per example, or keep the structured form.
        # Here we flatten each example's frames and patches into a sequence of length L = F * H_merge * W_merge.
        out = merged.view(B, F * H_merge * W_merge, self.embed_dim2)  # shape (B, L, embed_dim2)
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
        # embed_dim should be half of what what ours is, since it's hierarchical and scales by a factor of two.
        assert embed_dim % 2 == 0, f"embed_dim ({embed_dim}) must be even for BurstPatchEmbedding"

        # 1. Burst patch embedding: embeds patches from all frames into token embeddings.
        self.patch_embed = BurstPatchEmbedding(
            # TODO(@connorbaker): Does the patch size or number of heads need to match?
            embed_dim=embed_dim // 2,
            patch_size=patch_size,
            num_heads=num_heads,
            window_size=8,
        )

        # 2. Transformer backbone for joint multi-frame feature extraction.
        # We use PyTorch's TransformerEncoder for simplicity. This applies global self-attention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Unembedding: We need to unpatchify the output of the transformer.
        self.patch_unembed = nn.ConvTranspose2d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 3. Output upsampling and reconstruction:
        # - We first project transformer tokens back to a low-res feature map for each frame.
        # - Then apply a meta-upsampling to reach the high-resolution output.
        # We'll implement the projection as a convolution that produces a feature map per frame.
        # Compute number of patches along height and width for re-shaping tokens (depends on image size and patch_size).
        # Since image size is variable, we'll do re-shaping in forward.
        # Define a convolution to map embedding dim to output RGB channels (3) after upsampling.
        self.toRGB = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)  # applied after upsampling features

        # Alternatively, we could output to RGB at low-res and then upsample RGB, but we choose to upsample features
        # then convert to RGB.

    def forward(self, burst, scale: float = 2.0) -> Tensor:
        """
        Forward pass for burst super-resolution.
        :param burst: Tensor of shape (B, 4, F, H2, W2) from BayerMosaicUnpack, where H2 = H/2 and W2 = W/2.
        :param scale: Float scaling factor for super-resolution (e.g., 1.5 for 1.5x).
        :return: Tensor of shape (B, 3, F, H_out, W_out) with super-resolved RGB images.
        """
        B, _, F, H2, W2 = burst.shape  # burst is four-channel (R, G1, G2, B) with F frames
        # 1. Patch embedding: get patch tokens for transformer.
        # The BurstPatchEmbedding should handle multiple frames seamlessly. We might need to reshape back to include F.
        # For example, it could internally group frames. We'll assume it returns tokens for each frame concatenated.
        tokens = self.patch_embed(
            burst
        )  # shape: (B, N_patches, embed_dim), where N_patches = F * (H2/patch_size/2) * (W2/patch_size/2)
        H_p = H2 // self.patch_size // 2  # number of patches for height after merging
        W_p = W2 // self.patch_size // 2  # number of patches for width after merging
        N_patches_per_frame = H_p * W_p
        N_patches = F * N_patches_per_frame  # total number of patches across all frames
        assert tokens.shape == (expected := (B, N_patches, self.embed_dim)), (
            f"tokens shape mismatch: {expected} vs. {tokens.shape}"
        )

        # 1. Transformer encoding: joint attention across all frame tokens.
        # This will allow each token to attend to others (both within same frame and across frames).
        # PyTorch TransformerEncoder expects shape (batch, seq_len, d_model) since we used batch_first=True.
        encoded = self.transformer(tokens)  # shape: (B, N_patches, embed_dim)
        assert encoded.shape == (expected := (B, N_patches, self.embed_dim)), (
            f"encoded shape mismatch: {expected} vs. {encoded.shape}"
        )

        # 2. Reshape encoded tokens back to frame-wise feature maps.
        # First, separate frames and tokens into patches.
        encoded = encoded.view(B, F, H_p, W_p, self.embed_dim).permute(0, 1, 4, 2, 3)  # (B, F, embed_dim, H_p, W_p)
        assert encoded.shape == (expected := (B, F, self.embed_dim, H_p, W_p)), (
            f"encoded reshape shape mismatch: {expected} vs. {encoded.shape}"
        )
        # Now we need to invert the patch embedding and create spatial feature maps.
        # BurstPatchEmbedding used a patch size and stride = patch_size. We'll reconstruct accordingly.
        # We "unpatchify" by reversing the patch embedding (we can use a conv transpose or fold operation).
        feature_map = self.patch_unembed(encoded.view(B * F, self.embed_dim, H_p, W_p)).view(
            B, F, self.embed_dim, H2 // 2, W2 // 2
        )
        assert feature_map.shape == (expected := (B, F, self.embed_dim, H2 // 2, W2 // 2)), (
            f"feature_map shape mismatch: {expected} vs. {feature_map.shape}"
        )

        # 3. Meta-upsampling to high resolution.
        # We'll upsample each frame's feature map by the given scale factor. For arbitrary scales, we use grid_sample
        # as a placeholder for meta-learning.
        out_frames = []
        for f in range(F):
            feat = feature_map[:, f]  # shape (B, embed_dim, H2, W2) for frame f
            # Use bilinear interpolation on features as a simple way to handle fractional scale.
            # In practice, replace this with a learned Meta-Upscale module that predicts filters for the given scale [oai_citation_attribution:13‡openaccess.thecvf.com](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.pdf#:~:text=the%20Meta,as%20input%20and%20use%20these).
            h_out = int(H2 * scale)
            w_out = int(W2 * scale)
            # Normalize grid coordinates for sampling
            grid_y = torch.linspace(0, H2 - 1, h_out, device=feat.device) / (H2 - 1) * 2 - 1
            grid_x = torch.linspace(0, W2 - 1, w_out, device=feat.device) / (W2 - 1) * 2 - 1
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1)  # shape (h_out, w_out, 2)
            assert grid.shape == (expected := (h_out, w_out, 2)), f"grid shape mismatch: {expected} vs. {grid.shape}"
            grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # shape (B, h_out, w_out, 2)
            assert grid.shape == (expected := (B, h_out, w_out, 2)), f"grid shape mismatch: {expected} vs. {grid.shape}"
            # Sample the feature map at high-res coordinates
            highres_feat = nn.functional.grid_sample(feat, grid, mode="bilinear", align_corners=True)
            # Convert high-res features to RGB image
            rgb = self.toRGB(highres_feat)  # shape (B, 3, h_out, w_out)
            assert rgb.shape == (expected := (B, 3, h_out, w_out)), f"rgb shape mismatch: {expected} vs. {rgb.shape}"
            out_frames.append(rgb)
        # Stack outputs for all frames
        output = torch.stack(out_frames, dim=2)  # shape (B, 3, F, H_out, W_out)
        assert output.shape == (expected := (B, 3, F, h_out, w_out)), (
            f"output shape mismatch: {expected} vs. {output.shape}"
        )
        return output
