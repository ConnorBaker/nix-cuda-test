import torch
from torch import Tensor


def bayer_mosaic_unpack(x: Tensor) -> Tensor:
    """
    x: Tensor of shape (H, W) containing RAW Bayer mosaic images.
        Assumes an RGGB Bayer pattern (RGBG in a 2x2 block with two greens).

    Returns:
        Tensor of shape (4, H/2, W/2) where each frame is split into
        4 channels: R, G1, G2, B.
    """
    H, W = x.shape[-2:]
    # Ensure H and W are even (so H/2 and W/2 are integers)
    assert H % 2 == 0 and W % 2 == 0, f"Image height ({H}) and width ({W}) must be even for Bayer unpacking."

    # Extract sub-pixels for each Bayer color channel using slicing.
    # RGGB pattern (assuming top-left pixel is R):
    R_ = x[0::2, 0::2]  # R channel: pixels at even row, even col
    G1 = x[0::2, 1::2]  # G1 channel: even row, odd col
    G2 = x[1::2, 0::2]  # G2 channel: odd row, even col
    B_ = x[1::2, 1::2]  # B channel: pixels at odd row, odd col

    # Stack the four channels along a new dimension (size 4).
    # Resulting shape after stack: (4, H/2, W/2)
    out = torch.stack([R_, G1, G2, B_])
    return out


def bayer_mosaic_pack(x: Tensor) -> Tensor:
    """
    x: Tensor of shape (B, 4, F, H/2, W/2) containing unpacked Bayer mosaic images.
        Assumes the channels are ordered as: R, G1, G2, B.

    Returns:
        Tensor of shape (B, F, H, W) where each frame is packed into a
        single channel using the Bayer pattern.
    """
    H2, W2 = x.shape[-2:]

    # Create an empty tensor to hold the packed image.
    packed = torch.empty((x.shape[0], x.shape[2], H2 * 2, W2 * 2), dtype=x.dtype, device=x.device)

    # Fill in the packed image using the unpacked channels.
    packed[..., 0::2, 0::2] = x[:, 0]  # R channel
    packed[..., 0::2, 1::2] = x[:, 1]  # G1 channel
    packed[..., 1::2, 0::2] = x[:, 2]  # G2 channel
    packed[..., 1::2, 1::2] = x[:, 3]  # B channel

    return packed
