"""
stage2/models/geometry_dit.py
==============================
Stage 2b: geometry token generation for Strategy A and Strategy D.

Two models sharing the same latent target (z_g) but differing in how z_s
conditions the generation:

  GeometryDiTA  (Strategy A)
      z_s is prepended as the first 16 tokens of the sequence (prefix).
      All 512 tokens attend to each other via self-attention.
      Loss computed only on the last 496 (z_g) positions.

  GeometryDiTD  (Strategy D)
      z_g [B, 496, 32] is the denoising sequence.
      z_s [B,  16, 32] conditions every transformer layer via cross-attention.
      This mirrors the VAE decoder's ZSCondTransformerDecoder architecture,
      so the DiT is learning to invert the exact conditioning structure used
      during Stage 1 decoding.
"""

import torch
import torch.nn as nn
from ..external.dit_block import (
    modulate, Mlp,
    DiTBlock, TimestepEmbedder, TokenFinalLayer, init_dit_weights,
)

# ── Constants matching Stage 1 ─────────────────────────────────────────────
N_ZS      = 16    # semantic tokens
N_ZG      = 496   # geometry tokens
N_TOTAL   = 512   # N_ZS + N_ZG
TOKEN_DIM = 32    # embed_dim from Stage 1


# ============================================================================
# CrossAttnDiTBlock — new building block for Strategy D
# ============================================================================

class CrossAttnDiTBlock(nn.Module):
    """
    DiT block with self-attention on x and cross-attention to a context sequence.

    Extends DiTBlock by inserting a cross-attention sublayer between self-attention
    and FFN.  The AdaLN-Zero conditioning from the timestep acts on self-attn and
    FFN only — the cross-attention reads the context tokens with a plain LayerNorm.

    forward(x, c, context):
        x       : [B, N, D]  denoising sequence (z_g tokens)
        c       : [B, D]     timestep conditioning vector
        context : [B, M, D]  conditioning sequence (z_s tokens, projected)
    returns [B, N, D]
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # ── Self-attention (AdaLN-Zero) ────────────────────────────────────
        self.norm1     = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=True)

        # ── Cross-attention (plain LN — context is external) ──────────────
        self.norm_ca   = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=True)

        # ── FFN (AdaLN-Zero) ──────────────────────────────────────────────
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = Mlp(hidden_size, int(hidden_size * mlp_ratio))

        # 6 values: shift/scale/gate for self-attn + shift/scale/gate for FFN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x:       torch.Tensor,
        c:       torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention with AdaLN modulation
        x_mod       = modulate(self.norm1(x), shift_msa, scale_msa)
        sa_out, _   = self.self_attn(x_mod, x_mod, x_mod)
        x           = x + gate_msa.unsqueeze(1) * sa_out

        # Cross-attention: x queries context (z_s or z_layout)
        ca_out, _   = self.cross_attn(self.norm_ca(x), context, context)
        x           = x + ca_out   # no gate — cross-attn is always active

        # FFN with AdaLN modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================================
# GeometryDiTA  —  Strategy A (prefix conditioning)
# ============================================================================

class GeometryDiTA(nn.Module):
    """
    Stage 2b for Strategy A.

    z_s_clean is prepended as a prefix to the z_g_noisy sequence.
    All 512 tokens participate in self-attention together.
    Output velocity is extracted from the last 496 positions (z_g).

    forward(x, t, z_s_clean) → [B, 496, 32]
        x         : [B, 496, 32]  noisy z_g tokens
        t         : [B]           timestep
        z_s_clean : [B,  16, 32]  clean z_s context (no noise added)
    """

    def __init__(
        self,
        hidden_size: int   = 384,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Single embedder for all 512 tokens (z_s and z_g are both token_dim=32)
        self.token_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.t_embedder     = TimestepEmbedder(hidden_size)

        # Learned PE over full 512-token sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, N_TOTAL, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Output: project all 512 positions, extract z_g positions in forward()
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)

        init_dit_weights(self)

    def forward(
        self,
        x:          torch.Tensor,
        t:          torch.Tensor,
        z_s_clean:  torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x         : [B, 496, 32]  noisy z_g  (this is x_t for the ODE)
        t         : [B]
        z_s_clean : [B, 16, 32]   prefix context (not denoised, no noise)
        returns     [B, 496, 32]  velocity for z_g positions
        """
        # Build 512-token sequence: [z_s_clean | z_g_noisy]
        seq = torch.cat([z_s_clean, x], dim=1)   # [B, 512, 32]
        h   = self.token_embedder(seq) + self.pos_embed  # [B, 512, D]
        c   = self.t_embedder(t)                          # [B, D]
        for block in self.blocks:
            h = block(h, c)
        out = self.final_layer(h, c)               # [B, 512, 32]
        return out[:, N_ZS:, :]                    # [B, 496, 32]  z_g positions only

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiTA(hidden={self.hidden_size}, "
            f"depth={len(self.blocks)}, "
            f"params={self.num_params()/1e6:.2f}M)"
        )


# ============================================================================
# GeometryDiTD  —  Strategy D (cross-attention conditioning)
# ============================================================================

class GeometryDiTD(nn.Module):
    """
    Stage 2b for Strategy D.

    z_g_noisy [B, 496, 32] is the denoising sequence.
    z_s_clean [B,  16, 32] conditions every block via cross-attention (K, V).

    This directly mirrors the ZSCondTransformerDecoder used in the Stage 1
    VAE decoder for Strategy D: the same conditioning structure in both stages
    means the Stage 2 DiT is learning to invert the exact function the Stage 1
    decoder uses.

    forward(x, t, z_s_clean) → [B, 496, 32]
        x         : [B, 496, 32]  noisy z_g tokens
        t         : [B]           timestep
        z_s_clean : [B,  16, 32]  clean z_s (K/V in cross-attention)
    """

    def __init__(
        self,
        hidden_size: int   = 384,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Separate projectors: z_g (denoised sequence) and z_s (context)
        self.zg_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.zs_embedder = nn.Linear(TOKEN_DIM, hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)

        # PE for z_g positions only (496)
        self.pos_embed = nn.Parameter(torch.zeros(1, N_ZG, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CrossAttnDiTBlock at every layer
        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)

        init_dit_weights(self)

    def forward(
        self,
        x:          torch.Tensor,
        t:          torch.Tensor,
        z_s_clean:  torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x         : [B, 496, 32]  noisy z_g
        t         : [B]
        z_s_clean : [B, 16, 32]   clean z_s conditioning (K/V)
        returns     [B, 496, 32]  velocity prediction
        """
        h_g  = self.zg_embedder(x) + self.pos_embed   # [B, 496, D]
        h_s  = self.zs_embedder(z_s_clean)             # [B,  16, D]
        c    = self.t_embedder(t)                       # [B, D]
        for block in self.blocks:
            h_g = block(h_g, c, h_s)
        return self.final_layer(h_g, c)               # [B, 496, 32]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiTD(hidden={self.hidden_size}, "
            f"depth={len(self.blocks)}, "
            f"params={self.num_params()/1e6:.2f}M)"
        )


# ── Size presets ─────────────────────────────────────────────────────────────

def GeometryDiTA_S(**kw):  return GeometryDiTA(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiTA_B(**kw):  return GeometryDiTA(hidden_size=384, depth=12, num_heads=12, **kw)  # default
def GeometryDiTA_L(**kw):  return GeometryDiTA(hidden_size=512, depth=16, num_heads=16, **kw)

def GeometryDiTD_S(**kw):  return GeometryDiTD(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiTD_B(**kw):  return GeometryDiTD(hidden_size=384, depth=12, num_heads=12, **kw)  # default
def GeometryDiTD_L(**kw):  return GeometryDiTD(hidden_size=512, depth=16, num_heads=16, **kw)

GeometryDiT_models = {
    "GeometryDiTA-S": GeometryDiTA_S,  "GeometryDiTA-B": GeometryDiTA_B,  "GeometryDiTA-L": GeometryDiTA_L,
    "GeometryDiTD-S": GeometryDiTD_S,  "GeometryDiTD-B": GeometryDiTD_B,  "GeometryDiTD-L": GeometryDiTD_L,
}