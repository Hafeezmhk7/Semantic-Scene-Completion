"""
stage2/models/geometry_dit.py
==============================
Stage 2b: geometry token generation for Strategy A and Strategy D.

Positional encoding options (ablation)
---------------------------------------
Three options are available via the rope_type constructor argument.
The flag is also exposed in the SLURM job script as ROPE_TYPE.

  'learned_ape'  (DEFAULT — current behaviour)
      Learned absolute positional embedding, added once at the input:
          h_g = zg_embedder(x) + pos_embed   [1, 496, D]
      Standard ViT/BERT style.  Positional information is only present at
      the first layer.  No relative position in Q·K dot products.

  '1d'
      1-D Rotary Positional Embedding applied to Q and K at every block.
      pos_embed is removed; no learned positional parameters added.
      Encodes relative token sequence index (0 → 495) in the Q·K product.
      Based on RoFormer (Su et al., arXiv:2104.09864).

  '3d'
      3-D Rotary Positional Embedding.  head_dim=32 is split into
      three axis bands (d_x=10, d_y=10, d_z=12).  Each geometry token
      is mapped to an (x, y, z) coordinate in an 8×8×8 grid:
          token i → (i%8, (i//8)%8, i//64)
      Applied to Q and K at every block (self-attention and cross-attention).
      Encodes spatial proximity between tokens — tokens that aggregate
      geometrically nearby Gaussians share similar 3D positions.
      Based on VideoRoPE / RoPE-3D (2024).

Both RoPE variants share pre-computed cos/sin tables across all blocks
(stored as non-trainable buffers in the model, injected once at init).
This avoids redundant recomputation and is DDP/FSDP-safe.

THREE model classes
--------------------
  GeometryDiTA   Strategy A, cross-attention z_s conditioning
  GeometryDiTD   Strategy D, cross-attention z_s conditioning (same arch)
  GeometryDiT_adaLN  ABLATION — pooled adaLN, no cross-attention

Each accepts rope_type: str = 'learned_ape' as a constructor argument.
"""

import torch
import torch.nn as nn
from ..external.dit_block import (
    modulate, Mlp,
    DiTBlock, CrossAttnDiTBlock,
    TimestepEmbedder, TokenFinalLayer, init_dit_weights,
)

# ── Constants matching Stage 1 ─────────────────────────────────────────────
N_ZS      = 16    # semantic tokens
N_ZG      = 496   # geometry tokens
TOKEN_DIM = 32    # embed_dim from Stage 1

_VALID_ROPE = ('learned_ape', '1d', '3d')


def _make_rope(rope_type: str, head_dim: int) -> tuple:
    """
    Construct RoPE modules for z_g self-attention (rope_zg) and
    z_s cross-attention keys (rope_zs).

    Returns (rope_zg, rope_zs) where each is an nn.Module or None.

    rope_zg : applied to Q and K in self-attention over the 496 z_g tokens,
              and to Q in cross-attention.
    rope_zs : applied to K in cross-attention over the 16 z_s tokens
              (always 1D — z_s tokens have no 3D spatial structure).

    When rope_type='learned_ape', both are None and pos_embed is used instead.
    """
    if rope_type == 'learned_ape':
        return None, None
    if rope_type not in _VALID_ROPE:
        raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")

    from ..external.rope import RoPE1D, RoPE3D

    if rope_type == '1d':
        rope_zg = RoPE1D(head_dim=head_dim, max_seq_len=N_ZG + 16)
        rope_zs = RoPE1D(head_dim=head_dim, max_seq_len=N_ZS)
    else:  # '3d'
        rope_zg = RoPE3D(head_dim=head_dim, seq_len=N_ZG)
        rope_zs = RoPE1D(head_dim=head_dim, max_seq_len=N_ZS)  # z_s stays 1D

    return rope_zg, rope_zs


# ============================================================================
# GeometryDiTA  —  Strategy A  (cross-attention conditioning)
# ============================================================================

class GeometryDiTA(nn.Module):
    """
    Stage 2b for Strategy A — cross-attention z_s conditioning.

    z_g_noisy [B, 496, 32] is the denoising sequence.
    z_s_clean [B,  16, 32] conditions every block via cross-attention K/V.

    Positional encoding is controlled by rope_type (see module docstring).
    When rope_type='learned_ape' (default), behaviour is identical to the
    previous version.

    forward(x, t, z_s_clean) → [B, 496, 32]
    """

    def __init__(
        self,
        hidden_size: int   = 384,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        rope_type:   str   = 'learned_ape',
    ):
        super().__init__()
        if rope_type not in _VALID_ROPE:
            raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")
        self.hidden_size = hidden_size
        self.rope_type   = rope_type

        self.zg_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.zs_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.t_embedder  = TimestepEmbedder(hidden_size)

        head_dim  = hidden_size // num_heads
        rope_zg, rope_zs = _make_rope(rope_type, head_dim)

        # Learned APE — only when rope_type='learned_ape'
        if rope_type == 'learned_ape':
            self.pos_embed = nn.Parameter(torch.zeros(1, N_ZG, hidden_size))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Register as buffers so state_dict contains the RoPE tables
            # (makes checkpoints self-contained and DDP-safe)
            self.register_module('rope_zg', rope_zg)
            self.register_module('rope_zs', rope_zs)
            self.pos_embed = None   # explicit None for clarity

        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                rope_type=rope_type,
                rope_module=rope_zg,
                rope_kv=rope_zs,
            )
            for _ in range(depth)
        ])
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)
        init_dit_weights(self)

    def forward(
        self,
        x:         torch.Tensor,
        t:         torch.Tensor,
        z_s_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        x         : [B, 496, 32]  noisy z_g
        t         : [B]           timestep ∈ [0, 1]
        z_s_clean : [B,  16, 32]  clean z_s from LayoutDiT (K/V at every block)
        returns     [B, 496, 32]  velocity prediction
        """
        h_g = self.zg_embedder(x)
        if self.pos_embed is not None:
            h_g = h_g + self.pos_embed   # only for learned_ape

        h_s = self.zs_embedder(z_s_clean)    # [B, 16, D]
        c   = self.t_embedder(t)              # [B, D]

        for block in self.blocks:
            h_g = block(h_g, c, h_s)

        return self.final_layer(h_g, c)   # [B, 496, 32]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiTA(rope={self.rope_type}, "
            f"hidden={self.hidden_size}, depth={len(self.blocks)}, "
            f"params={self.num_params()/1e6:.2f}M)"
        )


# ============================================================================
# GeometryDiTD  —  Strategy D  (cross-attention conditioning)
# ============================================================================

class GeometryDiTD(nn.Module):
    """
    Stage 2b for Strategy D — cross-attention conditioning.

    Architecturally identical to GeometryDiTA. Kept as a separate class
    because it pairs with a Strategy D Stage 1 checkpoint
    (decoder_zs_cross_attn=True).

    forward(x, t, z_s_clean) → [B, 496, 32]
    """

    def __init__(
        self,
        hidden_size: int   = 384,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        rope_type:   str   = 'learned_ape',
    ):
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
                rope_type=rope_type,
                rope_module=rope_zg,
                rope_kv=rope_zs,
            )
            for _ in range(depth)
        ])
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)
        init_dit_weights(self)

    def forward(
        self,
        x:         torch.Tensor,
        t:         torch.Tensor,
        z_s_clean: torch.Tensor,
    ) -> torch.Tensor:
        h_g = self.zg_embedder(x)
        if self.pos_embed is not None:
            h_g = h_g + self.pos_embed
        h_s = self.zs_embedder(z_s_clean)
        c   = self.t_embedder(t)
        for block in self.blocks:
            h_g = block(h_g, c, h_s)
        return self.final_layer(h_g, c)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiTD(rope={self.rope_type}, "
            f"hidden={self.hidden_size}, depth={len(self.blocks)}, "
            f"params={self.num_params()/1e6:.2f}M)"
        )


# ============================================================================
# GeometryDiT_adaLN  —  ABLATION  (pooled adaLN conditioning)
# ============================================================================

class GeometryDiT_adaLN(nn.Module):
    """
    ABLATION: GeometryDiT with pooled adaLN conditioning instead of
    cross-attention.  Use with --zs_conditioning adaLN.

    rope_type applies to the self-attention DiTBlocks.
    Since there is no cross-attention, rope_zs is not used.

    forward(x, t, z_s_clean) → [B, 496, 32]
    """

    def __init__(
        self,
        hidden_size: int   = 384,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        rope_type:   str   = 'learned_ape',
    ):
        super().__init__()
        if rope_type not in _VALID_ROPE:
            raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")
        self.hidden_size = hidden_size
        self.rope_type   = rope_type

        self.zg_embedder  = nn.Linear(TOKEN_DIM, hidden_size)
        self.zs_pool_proj = nn.Linear(TOKEN_DIM, hidden_size)
        self.t_embedder   = TimestepEmbedder(hidden_size)

        head_dim          = hidden_size // num_heads
        rope_zg, _        = _make_rope(rope_type, head_dim)  # no cross-attn → zs unused

        if rope_type == 'learned_ape':
            self.pos_embed = nn.Parameter(torch.zeros(1, N_ZG, hidden_size))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_module('rope_zg', rope_zg)
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                rope_type=rope_type,
                rope_module=rope_zg,
            )
            for _ in range(depth)
        ])
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)
        init_dit_weights(self)

    def forward(
        self,
        x:         torch.Tensor,
        t:         torch.Tensor,
        z_s_clean: torch.Tensor,
    ) -> torch.Tensor:
        h          = self.zg_embedder(x)
        if self.pos_embed is not None:
            h = h + self.pos_embed
        c          = self.t_embedder(t)
        z_s_pooled = z_s_clean.mean(dim=1)
        c          = c + self.zs_pool_proj(z_s_pooled)
        for block in self.blocks:
            h = block(h, c)
        return self.final_layer(h, c)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiT_adaLN(rope={self.rope_type}, "
            f"hidden={self.hidden_size}, depth={len(self.blocks)}, "
            f"params={self.num_params()/1e6:.2f}M)"
        )


# ============================================================================
# Size presets and model registries
# ============================================================================

def GeometryDiTA_S(**kw): return GeometryDiTA(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiTA_B(**kw): return GeometryDiTA(hidden_size=384, depth=12, num_heads=12, **kw)
def GeometryDiTA_L(**kw): return GeometryDiTA(hidden_size=512, depth=16, num_heads=16, **kw)

def GeometryDiTD_S(**kw): return GeometryDiTD(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiTD_B(**kw): return GeometryDiTD(hidden_size=384, depth=12, num_heads=12, **kw)
def GeometryDiTD_L(**kw): return GeometryDiTD(hidden_size=512, depth=16, num_heads=16, **kw)

def GeometryDiT_adaLN_S(**kw): return GeometryDiT_adaLN(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiT_adaLN_B(**kw): return GeometryDiT_adaLN(hidden_size=384, depth=12, num_heads=12, **kw)
def GeometryDiT_adaLN_L(**kw): return GeometryDiT_adaLN(hidden_size=512, depth=16, num_heads=16, **kw)

GeometryDiT_models = {
    "GeometryDiTA-S": GeometryDiTA_S,
    "GeometryDiTA-B": GeometryDiTA_B,
    "GeometryDiTA-L": GeometryDiTA_L,
    "GeometryDiTD-S": GeometryDiTD_S,
    "GeometryDiTD-B": GeometryDiTD_B,
    "GeometryDiTD-L": GeometryDiTD_L,
}

GeometryDiT_adaLN_models = {
    "GeometryDiT_adaLN-S": GeometryDiT_adaLN_S,
    "GeometryDiT_adaLN-B": GeometryDiT_adaLN_B,
    "GeometryDiT_adaLN-L": GeometryDiT_adaLN_L,
}