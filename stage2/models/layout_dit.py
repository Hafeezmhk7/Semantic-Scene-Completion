"""
stage2/models/layout_dit.py
===========================
Stage 2a: LayoutDiT

Generates z_s tokens [B, 16, 32] from Gaussian noise via flow matching.
Used by Strategy A and Strategy D (both share this same Stage 2a model).

Architecture
------------
    z_s_noisy [B, 16, 32]
        → LinearEmbed  [B, 16, hidden_size]  + learned PE [1, 16, hidden_size]
        → N × DiTBlock(hidden_size, num_heads)   conditioned on t_embed [B, D]
        → TokenFinalLayer                         → [B, 16, 32]
        → velocity prediction v_s

Flow matching target: v_s = z_s_clean − z_s_noise
"""

import torch
import torch.nn as nn
from ..external.dit_block import (
    DiTBlock, TimestepEmbedder, TokenFinalLayer, init_dit_weights
)

# ── Constants matching Stage 1 ─────────────────────────────────────────────
N_ZS      = 16   # number of z_s tokens
TOKEN_DIM = 32   # embed_dim in Stage 1 (shapevae-256.yaml)


class LayoutDiT(nn.Module):
    """
    Stage 2a DiT for z_s generation.

    Parameters
    ----------
    hidden_size : int   transformer width             (default 256)
    depth       : int   number of DiT blocks          (default 6)
    num_heads   : int   attention heads               (default 8)
    mlp_ratio   : float FFN expansion ratio           (default 4.0)

    Input / output
    --------------
    forward(z_s_noisy, t) → velocity [B, 16, 32]
        z_s_noisy : [B, 16, 32]  noisy layout tokens
        t         : [B]          timestep in [0, 1]
    """

    def __init__(
        self,
        hidden_size: int   = 256,
        depth:       int   = 6,
        num_heads:   int   = 8,
        mlp_ratio:   float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projection: token_dim (32) → hidden_size
        self.token_embedder = nn.Linear(TOKEN_DIM, hidden_size)

        # Timestep conditioning
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Learned positional embedding — 16 positions
        self.pos_embed = nn.Parameter(torch.zeros(1, N_ZS, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Output: [B, 16, hidden_size] → [B, 16, 32]
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)

        init_dit_weights(self)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 16, 32]  noisy z_s tokens (x_t in flow matching notation)
        t : [B]          timestep ∈ [0, 1]

        Returns
        -------
        [B, 16, 32]  predicted velocity v_θ(x_t, t)
        """
        x = self.token_embedder(x) + self.pos_embed   # [B, 16, D]
        c = self.t_embedder(t)                         # [B, D]
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)                  # [B, 16, 32]

    # ── Convenience ────────────────────────────────────────────────────────

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"LayoutDiT(hidden={self.hidden_size}, "
            f"depth={len(self.blocks)}, "
            f"heads={self.blocks[0].attn.num_heads}, "
            f"params={self.num_params()/1e6:.2f}M)"
        )


# ── Size presets ─────────────────────────────────────────────────────────────

def LayoutDiT_S(**kw):  return LayoutDiT(hidden_size=128, depth=4,  num_heads=4, **kw)
def LayoutDiT_B(**kw):  return LayoutDiT(hidden_size=256, depth=6,  num_heads=8, **kw)  # default
def LayoutDiT_L(**kw):  return LayoutDiT(hidden_size=384, depth=8,  num_heads=12, **kw)

LayoutDiT_models = {
    "LayoutDiT-S": LayoutDiT_S,
    "LayoutDiT-B": LayoutDiT_B,
    "LayoutDiT-L": LayoutDiT_L,
}