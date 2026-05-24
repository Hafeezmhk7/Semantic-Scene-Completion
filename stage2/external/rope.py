"""
stage2/external/rope.py
========================
Rotary Positional Embedding (RoPE) utilities for 1D and 3D variants.

Current situation in Can3Tok Stage 2
-------------------------------------
GeometryDiTA/D currently uses a *learned absolute positional embedding* (APE):
    self.pos_embed = nn.Parameter(torch.zeros(1, 496, hidden_size))
    h_g = self.zg_embedder(x) + self.pos_embed   # added once at input
This is standard ViT/BERT style. It has two limitations:
  1. Positional information is injected only at the input; deeper layers
     see no relative position signal in their Q·K dot products.
  2. It gives no inductive bias for the 3D spatial structure of the scene.

What RoPE provides
-------------------
RoPE (Su et al. 2021, arXiv:2104.09864) modifies the attention score
between positions i and j as:
    score(i, j) = (R_i q_i)^T (R_j k_j) = q_i^T R_{j-i} k_j
where R_p is a rotation matrix parameterised by position p.  This means
the Q·K dot product automatically encodes relative position (j-i) — at
every layer, not just the first.

1D RoPE  —  applies to token sequence index (0 → N_ZG-1 = 495)
3D RoPE  —  decomposes head_dim into three axis bands (x, y, z) and
             applies independent 1D RoPE per axis using grid coordinates
             derived from a 3D arrangement of the 496 geometry tokens.

3D grid assignment for 496 geometry tokens
--------------------------------------------
The 496 z_g tokens come from the Perceiver encoder which cross-attends
over 40,000 Gaussians.  There is no explicit spatial ordering in the
token sequence, but nearby Gaussians tend to pool into nearby tokens
(the Perceiver uses learned cross-attention, not random pooling).  We
therefore assign 3D coordinates by linearising a 8×8×8=512 grid and
taking the first 496 positions in row-major order (x-fastest):
    token i  →  (px, py, pz) = (i%8, (i//8)%8, i//64)
This is an approximation — it encodes relative position within an
implicit spatial cube, not exact Gaussian xyz.  Whether this structural
inductive bias helps or hurts is what the ablation measures.

head_dim decomposition for GeometryDiTA-B (hidden=384, heads=12)
------------------------------------------------------------------
    head_dim = 384 / 12 = 32
    1D: all 32 dims  →  16 frequency pairs encode sequence position
    3D: 32 = d_x(10) + d_y(10) + d_z(12)
        first 10 dims:  5 frequency pairs encode px (0..7)
        next  10 dims:  5 frequency pairs encode py (0..7)
        last  12 dims:  6 frequency pairs encode pz (0..7)

References
----------
[RoFormer]    Su et al. "RoFormer: Enhanced Transformer with Rotary
              Position Embedding." arXiv:2104.09864 (2021).
[LLaMA]       Touvron et al. "LLaMA: Open and Efficient Foundation
              Language Models." arXiv:2302.13971 (2023).
[VideoRoPE]   "RoPE-3D / VideoRoPE" — temporal+spatial axis factorisation
              for video transformers (2024).
[RoPE-3D]     Wang et al. "RoPE3D: Road Segmentation and Free Space
              Detection in Bird's-Eye View Using 3D Rotary Embeddings."
              (concept adapted here for scene tokens.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# Low-level helpers
# ============================================================================

def _build_freqs(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Build frequency matrix.

    For dim-dimensional RoPE applied to a sequence of seq_len positions:
        freqs[p, i] = p / theta^(2i / dim)   for i in 0 .. dim//2 - 1

    Returns [seq_len, dim//2]
    """
    half  = dim // 2
    inv_f = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t     = torch.arange(seq_len, dtype=torch.float32)
    return torch.outer(t, inv_f)   # [seq_len, dim//2]


def _cos_sin(freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, each [N, half_dim], each duplicated to [N, dim]."""
    c = torch.cos(freqs)
    s = torch.sin(freqs)
    # Duplicate so the full head_dim can be handled with a single multiply.
    # The standard "split-half and rotate" trick:
    #   x = [x0, x1]  →  x' = [x0*cos - x1*sin, x0*sin + x1*cos]
    # is equivalent to: x' = x * cos_full + rotate_half(x) * sin_full
    # where cos_full = sin_full = repeated along the dim axis.
    return torch.cat([c, c], dim=-1), torch.cat([s, s], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[-x1, x0] rotation partner used in the 'split-half' RoPE formula."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE rotation to a Q or K tensor.

    x   : [B, num_heads, N, head_dim]
    cos : [N, head_dim]   pre-expanded to full dim (cos repeated twice)
    sin : [N, head_dim]

    Returns [B, num_heads, N, head_dim]
    """
    # broadcast over batch and head dims
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, N, D]
    sin = sin.unsqueeze(0).unsqueeze(0)   # [1, 1, N, D]
    return x * cos + _rotate_half(x) * sin


# ============================================================================
# 1-D RoPE  (sequence position 0 → N-1)
# ============================================================================

class RoPE1D(nn.Module):
    """
    Standard 1-D Rotary Positional Embedding.

    Encodes the 1-D sequence index of each token (0 → N_ZG-1 = 495 for
    geometry tokens, 0 → N_ZS-1 = 15 for semantic tokens).

    Pre-computes cos/sin tables up to max_seq_len and stores as buffers
    (never updated by the optimiser, compatible with DDP/FSDP).

    Usage
    -----
        rope = RoPE1D(head_dim=32, max_seq_len=512)
        cos, sin = rope(seq_len=496)     # → both [496, 32]
        q = apply_rope(q, cos, sin)      # q: [B, H, N, D]
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        freqs     = _build_freqs(head_dim, max_seq_len, theta)
        cos, sin  = _cos_sin(freqs)    # each [max_seq_len, head_dim]
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cos, sin: each [seq_len, head_dim]."""
        return (
            self.cos_cache[:seq_len],   # type: ignore[index]
            self.sin_cache[:seq_len],   # type: ignore[index]
        )


# ============================================================================
# 3-D RoPE  (xyz grid coordinates)
# ============================================================================

def _grid_coords_8x8x8(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign 3D grid coordinates to the first seq_len positions of an
    8×8×8 = 512 cube in row-major (x-fastest) order.

    token i  →  (px, py, pz) = (i % 8,  (i // 8) % 8,  i // 64)

    For seq_len=496 the last 16 positions of the z=7 face are unused.
    This is intentional — token 495 = (7, 5, 7) is the last valid position.

    Returns px, py, pz: each [seq_len], integer tensor with values in [0, 7].
    """
    assert seq_len <= 512, f"8×8×8 grid supports at most 512 tokens, got {seq_len}"
    ids = torch.arange(seq_len, dtype=torch.long)
    px  = ids % 8
    py  = (ids // 8) % 8
    pz  = ids // 64
    return px, py, pz


def _build_3d_cos_sin(
    head_dim:  int,
    seq_len:   int,
    theta:     float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build 3D RoPE cos/sin tables for seq_len tokens.

    head_dim is split into three bands:
        d_x = head_dim // 3
        d_y = head_dim // 3
        d_z = head_dim - 2 * (head_dim // 3)   ← absorbs remainder

    For GeometryDiTA-B (head_dim=32): d_x=10, d_y=10, d_z=12.

    The coordinate for each axis is the grid index (0-7) treated as a
    position for 1D RoPE within that axis's frequency band.

    Returns cos_3d, sin_3d: each [seq_len, head_dim]
    """
    d_x = head_dim // 3
    d_y = head_dim // 3
    d_z = head_dim - 2 * d_x        # absorbs remainder (≥ d_x, d_y)

    px, py, pz = _grid_coords_8x8x8(seq_len)   # each [seq_len], int

    # Per-axis frequency tables indexed by coordinate value (0–7)
    # freqs_ax: [8, d_ax // 2]  — precompute for all possible coords
    freqs_x = _build_freqs(d_x, 8, theta)   # [8, d_x//2]
    freqs_y = _build_freqs(d_y, 8, theta)
    freqs_z = _build_freqs(d_z, 8, theta)

    # Look up the frequency row for each token's coordinate
    fx = freqs_x[px]    # [seq_len, d_x//2]
    fy = freqs_y[py]
    fz = freqs_z[pz]

    # Build full cos/sin for each axis band
    cos_x, sin_x = _cos_sin(fx)   # each [seq_len, d_x]
    cos_y, sin_y = _cos_sin(fy)
    cos_z, sin_z = _cos_sin(fz)

    # Concatenate along head_dim
    cos_3d = torch.cat([cos_x, cos_y, cos_z], dim=-1)   # [seq_len, head_dim]
    sin_3d = torch.cat([sin_x, sin_y, sin_z], dim=-1)
    return cos_3d, sin_3d


def apply_rope_3d(
    x:       torch.Tensor,
    cos_3d:  torch.Tensor,
    sin_3d:  torch.Tensor,
) -> torch.Tensor:
    """
    Apply 3D RoPE to Q or K.

    head_dim is split into three bands (x, y, z) and each band is
    rotated independently using that axis's cos/sin values.  The
    _rotate_half trick still works per-band as long as each band has
    even dimension.

    x      : [B, num_heads, N, head_dim]
    cos_3d : [N, head_dim]
    sin_3d : [N, head_dim]

    Returns [B, num_heads, N, head_dim]
    """
    head_dim = x.shape[-1]
    d_x      = head_dim // 3
    d_y      = head_dim // 3

    # Decompose into axis bands
    x_x = x[..., :d_x]                    # x-axis band
    x_y = x[..., d_x : d_x + d_y]         # y-axis band
    x_z = x[..., d_x + d_y :]             # z-axis band

    c_x = cos_3d[:, :d_x];        s_x = sin_3d[:, :d_x]
    c_y = cos_3d[:, d_x:d_x+d_y]; s_y = sin_3d[:, d_x:d_x+d_y]
    c_z = cos_3d[:, d_x+d_y:];    s_z = sin_3d[:, d_x+d_y:]

    # Apply 1D RoPE rotation independently per band
    def _rot(v, c, s):
        c = c.unsqueeze(0).unsqueeze(0)   # [1, 1, N, band]
        s = s.unsqueeze(0).unsqueeze(0)
        return v * c + _rotate_half(v) * s

    return torch.cat([_rot(x_x, c_x, s_x),
                      _rot(x_y, c_y, s_y),
                      _rot(x_z, c_z, s_z)], dim=-1)


class RoPE3D(nn.Module):
    """
    3-D Rotary Positional Embedding for geometry tokens.

    Tokens are mapped to an 8×8×8 grid, and each of the three head_dim
    bands encodes one spatial axis.  All cos/sin tables are pre-computed
    and stored as buffers.

    Usage
    -----
        rope = RoPE3D(head_dim=32, seq_len=496)
        cos3d, sin3d = rope()           # → each [496, 32]
        q = apply_rope_3d(q, cos3d, sin3d)
    """

    def __init__(self, head_dim: int, seq_len: int = 496, theta: float = 10000.0):
        super().__init__()
        cos_3d, sin_3d = _build_3d_cos_sin(head_dim, seq_len, theta)
        self.register_buffer("cos_cache", cos_3d, persistent=False)
        self.register_buffer("sin_cache", sin_3d, persistent=False)
        self.head_dim = head_dim
        self.seq_len  = seq_len

        # Log decomposition for transparency
        d_x = head_dim // 3
        d_y = head_dim // 3
        d_z = head_dim - 2 * d_x
        print(f"  [RoPE3D] head_dim={head_dim}: "
              f"d_x={d_x}, d_y={d_y}, d_z={d_z}  "
              f"(grid 8×8×8, {seq_len} tokens)")

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cos_3d, sin_3d: each [seq_len, head_dim]."""
        return self.cos_cache, self.sin_cache   # type: ignore[return-value]