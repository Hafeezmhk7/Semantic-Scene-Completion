"""
stage2/models/geometry_dit.py
==============================
Stage 2b: geometry token generation for Strategy A and Strategy D.

THREE models are provided:

  GeometryDiTA  (Strategy A, --zs_conditioning cross_attn)
      z_g [B, 496, 32] is the denoising sequence.
      z_s [B,  16, 32] conditions every block via cross-attention (K, V).
      Architecturally identical to GeometryDiTD — kept as a separate class
      so thesis checkpoints are unambiguously labelled by strategy.

  GeometryDiTD  (Strategy D, --zs_conditioning cross_attn)
      Same architecture as GeometryDiTA.
      The naming distinction reflects which Stage 1 checkpoint it pairs with:
      a Strategy D Stage 1 checkpoint (decoder_zs_cross_attn=True) vs
      a Strategy A Stage 1 checkpoint.

  GeometryDiT_adaLN  (Strategy A or D, --zs_conditioning adaLN)
      ABLATION MODEL — tests how much per-token structure of z_s matters.
      z_s [B, 16, 32] → mean_pool → [B, 32] → Linear(32→D) → added to
      the timestep conditioning vector c before every block.
      No cross-attention sublayer. z_g tokens only self-attend.

Why cross-attention is the default
------------------------------------
Cross-attention has a dedicated softmax over only 16 z_s keys per z_g query,
giving the conditioning signal 100% of the softmax budget.  In the previous
prefix concatenation design, the 16 z_s keys competed with 496 z_g keys in
the same softmax, diluting the conditioning to ~3% of the budget.
The adaLN ablation collapses per-token structure entirely — it tests whether
the structured token split (colour / semantic / spatial) in z_s provides any
benefit over a single pooled layout vector.
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
TOKEN_DIM = 32    # embed_dim from Stage 1


# ============================================================================
# CrossAttnDiTBlock
# ============================================================================

class CrossAttnDiTBlock(nn.Module):
    """
    DiT block with self-attention on x and cross-attention to a context sequence.

    Sublayer order:  self-attention (adaLN-Zero)
                     → cross-attention (plain LN on query side, no gate)
                     → FFN (adaLN-Zero)

    The timestep conditioning c controls self-attention and FFN via adaLN-Zero.
    Cross-attention intentionally carries no adaLN gate — z_s conditioning is
    always fully active regardless of timestep.

    forward(x, c, context):
        x       : [B, N, D]  denoising sequence (z_g tokens)
        c       : [B, D]     timestep conditioning vector
        context : [B, M, D]  conditioning sequence (projected z_s tokens)
    returns [B, N, D]
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # Self-attention — AdaLN-Zero
        self.norm1      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=True)

        # Cross-attention — plain LN on query side, no adaLN modulation
        self.norm_ca    = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=True)

        # FFN — AdaLN-Zero
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = Mlp(hidden_size, int(hidden_size * mlp_ratio))

        # 6 adaLN params: shift/scale/gate for self-attn + shift/scale/gate for FFN
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

        # Self-attention with AdaLN-Zero
        x_mod     = modulate(self.norm1(x), shift_msa, scale_msa)
        sa_out, _ = self.self_attn(x_mod, x_mod, x_mod)
        x         = x + gate_msa.unsqueeze(1) * sa_out

        # Cross-attention: z_g queries z_s — no gate, always active
        ca_out, _ = self.cross_attn(self.norm_ca(x), context, context)
        x         = x + ca_out

        # FFN with AdaLN-Zero
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================================
# GeometryDiTA  —  Strategy A  (cross-attention conditioning)
# ============================================================================

class GeometryDiTA(nn.Module):
    """
    Stage 2b for Strategy A — cross-attention conditioning.

    z_g_noisy [B, 496, 32] is the denoising sequence.
    z_s_clean [B,  16, 32] conditions every block via cross-attention K/V.

    Architecturally identical to GeometryDiTD. The two classes are kept
    separate so training checkpoints are unambiguously labelled by which
    Stage 1 strategy they pair with.

    Note on the previous design: GeometryDiTA previously used prefix
    concatenation (z_s prepended as 16 tokens, 512-token self-attention).
    This was changed to cross-attention because the 16 z_s keys competed
    with 496 z_g keys in the same softmax, structurally diluting the
    conditioning signal to ~3% of the softmax budget.

    forward(x, t, z_s_clean) → [B, 496, 32]
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

        self.zg_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.zs_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.t_embedder  = TimestepEmbedder(hidden_size)

        # PE over z_g positions only (496)
        self.pos_embed = nn.Parameter(torch.zeros(1, N_ZG, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
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
        h_g = self.zg_embedder(x) + self.pos_embed  # [B, 496, D]
        h_s = self.zs_embedder(z_s_clean)            # [B,  16, D]
        c   = self.t_embedder(t)                      # [B, D]
        for block in self.blocks:
            h_g = block(h_g, c, h_s)
        return self.final_layer(h_g, c)              # [B, 496, 32]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiTA(cross_attn, hidden={self.hidden_size}, "
            f"depth={len(self.blocks)}, params={self.num_params()/1e6:.2f}M)"
        )


# ============================================================================
# GeometryDiTD  —  Strategy D  (cross-attention conditioning)
# ============================================================================

class GeometryDiTD(nn.Module):
    """
    Stage 2b for Strategy D — cross-attention conditioning.

    Architecturally identical to GeometryDiTA. Kept as a separate class
    because it pairs with a Strategy D Stage 1 checkpoint
    (decoder_zs_cross_attn=True), whose ZSCondTransformerDecoder uses
    the same cross-attention structure. The Stage 2 DiT therefore learns
    to invert the exact conditioning function the Stage 1 decoder uses.

    forward(x, t, z_s_clean) → [B, 496, 32]
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

        self.zg_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.zs_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.t_embedder  = TimestepEmbedder(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, N_ZG, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
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
        h_g = self.zg_embedder(x) + self.pos_embed
        h_s = self.zs_embedder(z_s_clean)
        c   = self.t_embedder(t)
        for block in self.blocks:
            h_g = block(h_g, c, h_s)
        return self.final_layer(h_g, c)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiTD(cross_attn, hidden={self.hidden_size}, "
            f"depth={len(self.blocks)}, params={self.num_params()/1e6:.2f}M)"
        )


# ============================================================================
# GeometryDiT_adaLN  —  ABLATION  (pooled adaLN conditioning)
# ============================================================================

class GeometryDiT_adaLN(nn.Module):
    """
    ABLATION: GeometryDiT with pooled adaLN conditioning instead of
    cross-attention.  Use with --zs_conditioning adaLN.

    Conditioning mechanism
    ----------------------
        z_s [B, 16, 32]
            → mean_pool                 → [B, 32]
            → zs_pool_proj: Linear(32→D)→ [B, D]
            → c = t_embed + zs_pool_proj→ [B, D]   ← same c used by all blocks

    No cross-attention sublayer. Every DiTBlock receives a combined
    conditioning vector c that carries both timestep and pooled layout
    information via the same adaLN_modulation (shift/scale/gate) path.

    What the ablation measures
    --------------------------
    Cross-attention preserves per-token selectivity: each z_g token
    independently attends to the z_s token most relevant to it
    (e.g. a floor region attending to spatial layout tokens 9–15).
    Mean pooling broadcasts identical layout information to every z_g token.
    If generation quality is substantially lower than cross-attention, it
    confirms that the token-level specialisation of z_s (enforced by Stage 1
    LayNCE, PoolNCE, and the structured split) is actively exploited during
    geometry generation — not just noise.

    Works for both --strategy A and --strategy D.

    forward(x, t, z_s_clean) → [B, 496, 32]
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

        self.zg_embedder  = nn.Linear(TOKEN_DIM, hidden_size)
        self.zs_pool_proj = nn.Linear(TOKEN_DIM, hidden_size)  # 32 → D
        self.t_embedder   = TimestepEmbedder(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, N_ZG, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Standard DiTBlocks — no cross-attention sublayer
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
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
        z_s_clean : [B,  16, 32]  clean z_s — mean-pooled and added to t_embed
        returns     [B, 496, 32]  velocity prediction
        """
        h          = self.zg_embedder(x) + self.pos_embed    # [B, 496, D]
        c          = self.t_embedder(t)                       # [B, D]
        z_s_pooled = z_s_clean.mean(dim=1)                    # [B, 32]
        c          = c + self.zs_pool_proj(z_s_pooled)        # [B, D]
        for block in self.blocks:
            h = block(h, c)
        return self.final_layer(h, c)                         # [B, 496, 32]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GeometryDiT_adaLN(hidden={self.hidden_size}, "
            f"depth={len(self.blocks)}, params={self.num_params()/1e6:.2f}M)"
        )


# ============================================================================
# Size presets and model registries
# ============================================================================

# Cross-attention (default)
def GeometryDiTA_S(**kw): return GeometryDiTA(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiTA_B(**kw): return GeometryDiTA(hidden_size=384, depth=12, num_heads=12, **kw)
def GeometryDiTA_L(**kw): return GeometryDiTA(hidden_size=512, depth=16, num_heads=16, **kw)

def GeometryDiTD_S(**kw): return GeometryDiTD(hidden_size=256, depth=8,  num_heads=8,  **kw)
def GeometryDiTD_B(**kw): return GeometryDiTD(hidden_size=384, depth=12, num_heads=12, **kw)
def GeometryDiTD_L(**kw): return GeometryDiTD(hidden_size=512, depth=16, num_heads=16, **kw)

# adaLN ablation
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