"""
stage2/models/geometry_dit512.py
================================
Stage 2b geometry generator for the SPLIT-latent experiments (6, 7), i.e.
local_disentangle=True.

In these experiments the geometry latent z_g is the FULL 512-token decoder
sequence (not the old 496), and z_s is a SEPARATE 16-token stochastic layout code
produced from the CLS. This differs from the old Strategy A (where z_s lived inside
the 512 as a prefix) and from B1 (where z_s was deterministic).

GeometryDiT512 generates z_g [B, 512, 32] conditioned on z_s [B, 16, 32] via
cross-attention at every block, mirroring the Stage 1 local-disentangle decoder
(zs_cond_decoder_B: 512 geometry tokens, z_s as cross-attention K/V per layer).
This is the GeometryDiTD architecture with the geometry token count generalised to
512 and z_s kept as a separate context rather than a prefix.

forward(x, t, z_s_clean) -> velocity [B, 512, 32]
"""

import torch
import torch.nn as nn

from ..external.dit_block import (
    CrossAttnDiTBlock, TimestepEmbedder, TokenFinalLayer, init_dit_weights,
)

N_ZG      = 512    # geometry tokens (full decoder sequence under local_disentangle)
N_ZS      = 16     # separate z_s layout tokens
TOKEN_DIM = 32
_VALID_ROPE = ('learned_ape', '1d', '3d')


def _make_rope(rope_type: str, head_dim: int):
    """Return (rope_zg, rope_zs). rope_zg drives self-attention over the 512 z_g
    tokens (and Q in cross-attention); rope_zs (1D) drives the 16 z_s keys."""
    if rope_type == 'learned_ape':
        return None, None
    from ..external.rope import RoPE1D, RoPE3D
    if rope_type == '1d':
        return (RoPE1D(head_dim=head_dim, max_seq_len=N_ZG + N_ZS),
                RoPE1D(head_dim=head_dim, max_seq_len=N_ZS))
    if rope_type == '3d':
        # 512 tokens fill the 8x8x8 grid exactly; row-major grid is an approximation
        # of the Hilbert order. z_s keys stay 1D (no 3D structure).
        return (RoPE3D(head_dim=head_dim, seq_len=N_ZG),
                RoPE1D(head_dim=head_dim, max_seq_len=N_ZS))
    raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")


class GeometryDiT512(nn.Module):
    """
    Cross-attention geometry DiT for local_disentangle Stage 1 checkpoints.

    z_g_noisy [B, 512, 32] is the denoising sequence.
    z_s_clean [B,  16, 32] conditions every block via cross-attention K/V.

    forward(x, t, z_s_clean) -> [B, 512, 32]
    """

    def __init__(self, hidden_size: int = 384, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, rope_type: str = 'learned_ape'):
        super().__init__()
        if rope_type not in _VALID_ROPE:
            raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")
        self.hidden_size = hidden_size
        self.rope_type   = rope_type

        self.zg_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.zs_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.t_embedder  = TimestepEmbedder(hidden_size)

        head_dim         = hidden_size // num_heads
        rope_zg, rope_zs = _make_rope(rope_type, head_dim)

        if rope_type == 'learned_ape':
            self.pos_embed = nn.Parameter(torch.zeros(1, N_ZG, hidden_size))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_module('rope_zg', rope_zg)
            self.register_module('rope_zs', rope_zs)
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                rope_type=rope_type, rope_module=rope_zg, rope_kv=rope_zs)
            for _ in range(depth)
        ])
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)
        init_dit_weights(self)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                z_s_clean: torch.Tensor) -> torch.Tensor:
        h_g = self.zg_embedder(x)
        if self.pos_embed is not None:
            h_g = h_g + self.pos_embed
        h_s = self.zs_embedder(z_s_clean)     # [B, 16, D]
        c   = self.t_embedder(t)              # [B, D]
        for block in self.blocks:
            h_g = block(h_g, c, h_s)
        return self.final_layer(h_g, c)       # [B, 512, 32]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (f"GeometryDiT512(rope={self.rope_type}, hidden={self.hidden_size}, "
                f"depth={len(self.blocks)}, params={self.num_params()/1e6:.2f}M)")


def GeometryDiT512_S(**kw): return GeometryDiT512(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiT512_B(**kw): return GeometryDiT512(hidden_size=384, depth=12, num_heads=12, **kw)  # default
def GeometryDiT512_L(**kw): return GeometryDiT512(hidden_size=512, depth=16, num_heads=16, **kw)

GeometryDiT512_models = {
    "GeometryDiT512-S": GeometryDiT512_S,
    "GeometryDiT512-B": GeometryDiT512_B,
    "GeometryDiT512-L": GeometryDiT512_L,
}