# ---------------------------------------------------------------
# Building blocks extracted from DiT (Peebles & Xie, ICCV 2023)
# https://github.com/facebookresearch/DiT  —  Apache 2.0 License
#
# Changes from the original:
#   * Replaced timm.Attention / timm.Mlp with pure-PyTorch equivalents
#     so there is no timm dependency in Stage 2.
#   * Replaced the image-specific FinalLayer with TokenFinalLayer that
#     outputs token_dim directly — suitable for any latent token sequence.
#   * Removed everything image-specific: LabelEmbedder, DiT, PatchEmbed,
#     2-D sinusoidal positional embedding utilities, and model configs.
#   * modulate() and TimestepEmbedder are kept verbatim from DiT.
#
# Can3Tok additions (RoPE ablation, May 2026):
#   * _RoPEAttention: drop-in replacement for nn.MultiheadAttention that
#     applies 1D or 3D RoPE to Q and K before scaled-dot-product attention.
#   * DiTBlock and CrossAttnDiTBlock now accept rope_type and rope_module
#     kwargs to switch between learned-APE (original), 1D-RoPE, and 3D-RoPE.
# ---------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers (verbatim from DiT)
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift.  [B,N,D] x [B,D] → [B,N,D]"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """Two-layer MLP with GELU.  Replaces timm.Mlp to remove dependency."""
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU(approximate="tanh")
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# TimestepEmbedder  (verbatim from DiT)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """t ∈ [0, 1] → sinusoidal frequencies → 2-layer MLP → [B, hidden_size]"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half      = dim // 2
        freqs     = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32) / half
        ).to(device=t.device)
        args      = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


# ---------------------------------------------------------------------------
# _RoPEAttention  — nn.MultiheadAttention replacement with RoPE support
# ---------------------------------------------------------------------------

class _RoPEAttention(nn.Module):
    """
    Multi-head attention with optional 1D or 3D Rotary Positional Embedding.

    This replaces nn.MultiheadAttention when rope_type != 'learned_ape'.
    It manually projects Q, K, V, applies RoPE to Q and K, then uses
    F.scaled_dot_product_attention (FlashAttention-compatible on H100).

    Weight layout matches nn.MultiheadAttention for easy checkpointing:
        in_proj_weight  : [3*D, D]  (Q, K, V stacked)
        in_proj_bias    : [3*D]
        out_proj.weight : [D, D]
        out_proj.bias   : [D]

    rope_module is a RoPE1D or RoPE3D instance. For self-attention the same
    rope is applied to both Q and K. For cross-attention (Q from z_g,
    K/V from z_s) the caller should pass rope_kv for K/V if the context
    has a different positional scheme (defaults to rope if not given).

    Parameters
    ----------
    embed_dim  : total hidden dimension D
    num_heads  : number of attention heads H
    rope_type  : 'none' | '1d' | '3d'
    rope_module: RoPE1D or RoPE3D instance (required when rope_type != 'none')
    rope_kv    : optional separate RoPE for K/V (used in cross-attention);
                 if None, rope_module is used for Q and plain (unrotated) for K/V
    """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        rope_type:   str                        = 'none',
        rope_module: Optional[nn.Module]        = None,
        rope_kv:     Optional[nn.Module]        = None,
        bias:        bool                       = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.rope_type  = rope_type
        self.rope_module = rope_module
        self.rope_kv     = rope_kv      # if None, K/V are not rotated in cross-attn

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialise like nn.MultiheadAttention
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project x [B, N, D] → Q, K, V each [B, H, N, head_dim]."""
        B, N, D  = x.shape
        qkv      = F.linear(x, self.in_proj_weight, self.in_proj_bias)   # [B, N, 3D]
        q, k, v  = qkv.chunk(3, dim=-1)                                  # each [B, N, D]
        def reshape(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        return reshape(q), reshape(k), reshape(v)   # each [B, H, N, head_dim]

    def _project_cross(
        self,
        query:   torch.Tensor,   # [B, N_q, D]
        context: torch.Tensor,   # [B, N_c, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project query → Q and context → K, V (cross-attention)."""
        B, N_q, D = query.shape
        B, N_c, _ = context.shape

        wq, wk, wv = self.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = (None, None, None)
        if self.in_proj_bias is not None:
            bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)

        def reshape(t, N):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(F.linear(query,   wq, bq), N_q)   # [B, H, N_q, hd]
        k = reshape(F.linear(context, wk, bk), N_c)   # [B, H, N_c, hd]
        v = reshape(F.linear(context, wv, bv), N_c)   # [B, H, N_c, hd]
        return q, k, v

    def _get_rope_tables(self, seq_len: int, is_3d: bool):
        """Retrieve cos/sin from the rope_module."""
        if is_3d:
            cos, sin = self.rope_module()   # [seq_len, head_dim]
        else:
            cos, sin = self.rope_module(seq_len)
        return cos, sin

    def forward_self(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self-attention: Q, K, V all from x.
        RoPE applied to both Q and K using the same position sequence.
        """
        B, N, D = x.shape
        q, k, v = self._project(x)   # each [B, H, N, hd]

        if self.rope_type != 'none' and self.rope_module is not None:
            is_3d    = (self.rope_type == '3d')
            cos, sin = self._get_rope_tables(N, is_3d)

            if is_3d:
                from .rope import apply_rope_3d
                q = apply_rope_3d(q, cos, sin)
                k = apply_rope_3d(k, cos, sin)
            else:
                from .rope import apply_rope
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)

        # FlashAttention-2 via torch sdpa (available on H100 with PyTorch ≥ 2.0)
        out = F.scaled_dot_product_attention(q, k, v)   # [B, H, N, hd]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

    def forward_cross(
        self,
        query:   torch.Tensor,   # z_g  [B, N_q, D]
        context: torch.Tensor,   # z_s  [B, N_c, D]
    ) -> torch.Tensor:
        """
        Cross-attention: Q from query (z_g), K/V from context (z_s).

        RoPE strategy for cross-attention:
          — Q (z_g tokens, 496): apply 1D or 3D RoPE via rope_module
          — K (z_s tokens,  16): apply 1D RoPE via rope_kv (if provided)
                                 otherwise leave unrotated.

        Rationale: z_s tokens are semantic (not spatially ordered in the
        same 3D grid as z_g). Using a separate short 1D RoPE for z_s keys
        lets the model encode that z_s token 5 is "closer" to z_s token 6
        than to z_s token 15, without imposing the 3D spatial assumption.
        """
        B, N_q, D = query.shape
        B, N_c, _ = context.shape
        q, k, v   = self._project_cross(query, context)

        if self.rope_type != 'none' and self.rope_module is not None:
            is_3d    = (self.rope_type == '3d')
            cos_q, sin_q = self._get_rope_tables(N_q, is_3d)

            if is_3d:
                from .rope import apply_rope_3d, apply_rope
                q = apply_rope_3d(q, cos_q, sin_q)
            else:
                from .rope import apply_rope
                q = apply_rope(q, cos_q, sin_q)

            # K/V: use rope_kv (1D) if provided
            if self.rope_kv is not None:
                cos_k, sin_k = self.rope_kv(N_c)
                from .rope import apply_rope
                k = apply_rope(k, cos_k, sin_k)
            # If rope_kv is None, K is left unrotated (relative to Q the
            # positions are "undefined" — equivalent to a global context token)

        out = F.scaled_dot_product_attention(q, k, v)   # [B, H, N_q, hd]
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        return self.out_proj(out)

    def forward(
        self,
        query:   torch.Tensor,
        key:     torch.Tensor,
        value:   torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        nn.MultiheadAttention-compatible interface.

        When query == key == value: self-attention.
        When query != key:          cross-attention.
        Returns (output, None) — None replaces the attn_weights for
        drop-in compatibility with the existing block code.
        """
        if query is key:
            return self.forward_self(query), None
        else:
            return self.forward_cross(query, key), None


# ---------------------------------------------------------------------------
# DiTBlock  — adaLN-Zero conditioning, with optional RoPE
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Source: DiT (Peebles & Xie, 2023).

    rope_type    : 'learned_ape' (default, no RoPE, same as original DiT)
                   '1d'          (1D RoPE on self-attention Q and K)
                   '3d'          (3D RoPE on self-attention Q and K)
    rope_module  : RoPE1D or RoPE3D instance.  MUST be provided when
                   rope_type in ('1d', '3d').  Injected by the parent model
                   so ALL blocks share the same pre-computed tables.

    forward(x, c):
        x  [B, N, D]  token sequence
        c  [B, D]     conditioning vector (timestep embedding)
    returns [B, N, D]
    """
    def __init__(
        self,
        hidden_size:  int,
        num_heads:    int,
        mlp_ratio:    float            = 4.0,
        rope_type:    str              = 'learned_ape',
        rope_module:  Optional[nn.Module] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        rope_active = rope_type in ('1d', '3d')
        if rope_active and rope_module is None:
            raise ValueError(f"DiTBlock: rope_type='{rope_type}' requires rope_module.")

        if rope_active:
            self.attn = _RoPEAttention(hidden_size, num_heads,
                                       rope_type=rope_type,
                                       rope_module=rope_module)
        else:
            # Original behaviour — plain nn.MultiheadAttention
            self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                              batch_first=True, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x_mod       = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x           = x + gate_msa.unsqueeze(1) * attn_out
        x           = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ---------------------------------------------------------------------------
# CrossAttnDiTBlock  — self-attn + cross-attn + FFN, with optional RoPE
# ---------------------------------------------------------------------------

class CrossAttnDiTBlock(nn.Module):
    """
    DiT block with self-attention on x and cross-attention to z_s context.

    Sublayer order:  self-attention (adaLN-Zero)
                     → cross-attention (plain LN, no gate)
                     → FFN (adaLN-Zero)

    rope_type, rope_module : same semantics as DiTBlock.
    rope_kv                : optional RoPE1D for z_s key positions in
                             cross-attention (default: 1D RoPE with seq=16).
                             Set to None to leave z_s keys unrotated.

    All three RoPE tables are injected once from the parent model and
    shared across all blocks (no per-block state, all in buffers).

    forward(x, c, context):
        x       : [B, N, D]  z_g denoising tokens
        c       : [B, D]     timestep conditioning
        context : [B, M, D]  projected z_s tokens (M=16)
    returns [B, N, D]
    """

    def __init__(
        self,
        hidden_size:  int,
        num_heads:    int,
        mlp_ratio:    float               = 4.0,
        rope_type:    str                 = 'learned_ape',
        rope_module:  Optional[nn.Module] = None,
        rope_kv:      Optional[nn.Module] = None,
    ):
        super().__init__()
        self.norm1   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_ca = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp     = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        rope_active = rope_type in ('1d', '3d')
        if rope_active and rope_module is None:
            raise ValueError(f"CrossAttnDiTBlock: rope_type='{rope_type}' requires rope_module.")

        if rope_active:
            # Self-attention: Q=K=z_g, apply rope_module to both
            self.self_attn = _RoPEAttention(hidden_size, num_heads,
                                            rope_type=rope_type,
                                            rope_module=rope_module)
            # Cross-attention: Q=z_g (rotated), K=z_s (rope_kv or unrotated)
            self.cross_attn = _RoPEAttention(hidden_size, num_heads,
                                             rope_type=rope_type,
                                             rope_module=rope_module,
                                             rope_kv=rope_kv)
        else:
            self.self_attn  = nn.MultiheadAttention(hidden_size, num_heads,
                                                    batch_first=True, bias=True)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads,
                                                    batch_first=True, bias=True)

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

        # Cross-attention: no adaLN gate — z_s conditioning always fully active
        ca_out, _ = self.cross_attn(self.norm_ca(x), context, context)
        x         = x + ca_out

        # FFN with AdaLN-Zero
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ---------------------------------------------------------------------------
# TokenFinalLayer  (replaces image-specific FinalLayer, unchanged)
# ---------------------------------------------------------------------------

class TokenFinalLayer(nn.Module):
    """
    Final layer for token-based DiT models.
    Projects each token [D] → [token_dim] with AdaLN modulation.

    forward(x, c):
        x  [B, N, D]     token sequence after all DiT blocks
        c  [B, D]        conditioning vector
    returns [B, N, token_dim]
    """
    def __init__(self, hidden_size: int, token_dim: int):
        super().__init__()
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear           = nn.Linear(hidden_size, token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


# ---------------------------------------------------------------------------
# Weight initialisation (mirrors DiT's initialize_weights, unchanged)
# ---------------------------------------------------------------------------

def init_dit_weights(model: nn.Module) -> None:
    """
    Initialise weights following the DiT recipe.
    Also handles _RoPEAttention in_proj_weight (Xavier uniform).
    """
    def _basic_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(_basic_init)

    # Zero-init adaLN-Zero modulation outputs
    for m in model.modules():
        if hasattr(m, 'adaLN_modulation') and isinstance(m.adaLN_modulation, nn.Sequential):
            last = m.adaLN_modulation[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.weight, 0)
                nn.init.constant_(last.bias,   0)

    # Zero-init final layer output projection
    for name, m in model.named_modules():
        if 'final_layer' in name and isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias,   0)

    # Normal init for TimestepEmbedder MLP
    for m in model.modules():
        if isinstance(m, TimestepEmbedder):
            nn.init.normal_(m.mlp[0].weight, std=0.02)
            nn.init.normal_(m.mlp[2].weight, std=0.02)

    # Xavier init for _RoPEAttention in_proj_weight
    for m in model.modules():
        if isinstance(m, _RoPEAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)