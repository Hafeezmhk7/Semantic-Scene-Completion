"""
Token-local decoder for Can3Tok.

REPLACES the flat GS_decoder MLP (~777M params, 1024-d bottleneck) with a
shared per-token MLP (~2M params, no bottleneck).

WHY:
  The flat GS_decoder compresses [B, 512, 384] → flatten → [B, 196608] → Linear → [B, 1024].
  This 1024-d bottleneck must hold ALL information for ALL 40,000 Gaussians of ALL
  scenes simultaneously. At 300 scenes this works via memorization. At 3800 scenes
  it fails because 1024 dims cannot encode enough scene-specific position information.

  The token-local decoder gives each Gaussian its own LOCAL 384-d context (its
  decoder token), with 512 such contexts running in parallel. Total information
  bandwidth is 512 × 384 = 196,608 dimensions instead of 1024. The transformer
  before the decoder already mixed information across tokens via self-attention,
  so each token's 384-d vector contains the relevant local geometry summary.

ARCHITECTURE:
  Input:  H_out [B, 512, 384]   — transformer output (one vector per decoder token)
  Per-token MLP applied independently (weights SHARED across tokens):
            Linear(384 → 512) → LN → GELU                                (h1)
            Linear(512 → 512) → LN → GELU                                (h2: per-token semantic context)
            Linear(512 → GAUSSIANS_PER_TOKEN × 14)                       (raw Gaussian attrs)
  Reshape and crop to 40,000 Gaussians, apply activations.

OUTPUTS (matching original GS_decoder interface):
  output_flat        : [B, 40000 × 14]   reconstructed Gaussians (flat).
  per_gaussian_feats : [B, 40000, feat_dim]   per-Gaussian features for InfoNCE.
                       Each Gaussian's feature comes from its parent token's h2,
                       optionally combined with a per-Gaussian positional embedding
                       so that 79 Gaussians from the same token get DISTINCT features.
  hidden (legacy)    : [B, 1024]   pooled hidden for backward-compat with code paths
                       that expect a global hidden vector (e.g. shape_embed-based
                       semantic head). Derived by mean-pooling h2 + projection.

PER-GAUSSIAN FEATURES (the important detail for InfoNCE):
  Naive approach: just repeat each token's h2 → 79 copies. But then all 79 Gaussians
  inside one token get IDENTICAL features, defeating per-Gaussian InfoNCE.

  Better approach: add a learned per-position embedding indexed by intra-token slot
  (0..78). So feature[i] = h2[token_of(i)] + pos_emb[slot_of(i)]. This gives every
  Gaussian a unique semantic feature while keeping per-token coherence.

  This is implemented below.

PARAMETER COUNT:
  in_linear   (384→512):   197,120
  in_norm                  1,024
  mid_linear  (512→512):   262,656
  mid_norm                 1,024
  out_linear  (512→79·14): 567,378
  pos_emb     (79×D_feat): 79·feat_dim ≈ 10K (with feat_dim=128)
  feat_proj   (512→D_feat): ~65K
  hidden_proj (512→1024):  525,312     (only used in legacy hidden path)
  ─────────────────────────
  Total:                   ~1.6M params, 500× reduction vs 777M GS_decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLocalDecoder(nn.Module):
    """
    Per-token decoder applied independently to each of NUM_TOKENS decoder tokens.

    Forward returns (output_flat, per_gaussian_features [, hidden]) depending on flags.

    Parameters
    ----------
    width : int
        Per-token feature dim from the transformer (default 384).
    hidden_dim : int
        Hidden width of the per-token MLP (default 512).
    num_tokens : int
        Number of decoder tokens (default 512). Must match upstream transformer.
    num_gaussians : int
        Total Gaussians to output per scene (default 40,000).
    color_residual : bool
        If True, colors are residuals (no clamp). If False, colors are clamped to [0,1].
    per_gaussian_feat_dim : int
        Output dim of per-Gaussian features used by downstream InfoNCE losses.
        Default 128. Set to 0 to disable per-Gaussian features (returns None for that slot).
    """

    NUM_OUT_PER_GAUSSIAN = 14   # pos[3] + color[3] + opacity[1] + scale[3] + quat[4]
    LEGACY_HIDDEN_DIM    = 1024 # for backward-compat with code expecting [B, 1024]

    def __init__(self, width=384, hidden_dim=512, num_tokens=512,
                 num_gaussians=40_000, color_residual=False,
                 per_gaussian_feat_dim=128):
        super().__init__()
        self.width                 = width
        self.hidden_dim            = hidden_dim
        self.num_tokens            = num_tokens
        self.num_gaussians         = num_gaussians
        self.color_residual        = color_residual
        self.per_gaussian_feat_dim = per_gaussian_feat_dim

        # ceil(40000 / 512) = 79 → 512 × 79 = 40,448 total outputs.
        self.g_per_token  = math.ceil(num_gaussians / num_tokens)
        self.total_output = self.num_tokens * self.g_per_token

        # ─── Per-token MLP (weights SHARED across tokens via broadcast) ─────────
        self.in_linear  = nn.Linear(width, hidden_dim)
        self.in_norm    = nn.LayerNorm(hidden_dim)
        self.mid_linear = nn.Linear(hidden_dim, hidden_dim)
        self.mid_norm   = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, self.g_per_token * self.NUM_OUT_PER_GAUSSIAN)

        # Small init on the output projection so initial Gaussians have near-zero
        # positions and near-unit scales (exp(0)=1). Prevents explosion at init.
        nn.init.trunc_normal_(self.out_linear.weight, std=0.01)
        nn.init.zeros_(self.out_linear.bias)

        # ─── Per-Gaussian semantic feature head ─────────────────────────────────
        # Produces a feat_dim-d feature for EVERY Gaussian (not just every token).
        # Mechanism: project h2 (per-token, [B, T, hidden_dim]) to feat_dim, then
        # add a learned intra-token position embedding indexed by slot in 0..g_per_token-1.
        # This gives every Gaussian a distinct feature while keeping per-token coherence.
        if per_gaussian_feat_dim > 0:
            self.feat_proj = nn.Linear(hidden_dim, per_gaussian_feat_dim)
            # Learned positional embedding for the g_per_token slots within a token.
            self.intra_token_pos_emb = nn.Parameter(
                torch.zeros(self.g_per_token, per_gaussian_feat_dim)
            )
            nn.init.trunc_normal_(self.intra_token_pos_emb, std=0.02)
        else:
            self.feat_proj = None
            self.intra_token_pos_emb = None

        # ─── Legacy global hidden (for code paths expecting [B, 1024]) ──────────
        # When return_hidden=True, we mean-pool h2 across tokens and project to 1024.
        # This is for backward-compat with SemanticProjectionHead-style heads that
        # operate on a global hidden. NOT used for per-Gaussian InfoNCE — that uses
        # per_gaussian_features above.
        self.hidden_proj = nn.Linear(hidden_dim, self.LEGACY_HIDDEN_DIM)
        nn.init.trunc_normal_(self.hidden_proj.weight, std=0.02)
        nn.init.zeros_(self.hidden_proj.bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[TokenLocalDecoder] {num_tokens} tokens × {width} → "
              f"{self.g_per_token} Gaussians/token × {self.NUM_OUT_PER_GAUSSIAN} attrs "
              f"({'residuals' if color_residual else 'clamp(0,1)'}) | "
              f"per_gauss_feat_dim={per_gaussian_feat_dim} | "
              f"{n_params/1e6:.2f}M params")

    def forward(self, H_out_or_flat, return_hidden=False, return_per_gaussian=False):
        """
        Forward pass.

        Parameters
        ----------
        H_out_or_flat : Tensor
            Either [B, num_tokens, width] (token-shaped) or
            [B, num_tokens * width] (flat — backward compat with existing call sites).
        return_hidden : bool
            If True, also returns a [B, 1024] pooled hidden (legacy compat).
        return_per_gaussian : bool
            If True, also returns per-Gaussian features [B, N, feat_dim] for
            future per-Gaussian InfoNCE paths that bypass SemanticProjectionHead.
            DEFAULT IS FALSE so that calling dec(x) / dec(x, return_hidden=True)
            is a drop-in replacement for the original GS_decoder (which returns
            `out` or `(out, hidden)`). The existing SemanticProjectionHead
            pipeline only needs the [B, 1024] hidden, not per-Gaussian feats.

        Returns
        -------
        Depending on flags, returns one of:
            output_flat                                                    if both False (DROP-IN)
            (output_flat, hidden)                                          if only return_hidden  (DROP-IN)
            (output_flat, per_gaussian_features)                           if only return_per_gaussian
            (output_flat, per_gaussian_features, hidden)                   if both True
        """
        # Accept both shapes for backward compat with existing decode() code paths.
        if H_out_or_flat.dim() == 2:
            B = H_out_or_flat.shape[0]
            H = H_out_or_flat.reshape(B, self.num_tokens, self.width)
        else:
            H = H_out_or_flat
            B = H.shape[0]
            assert H.shape[1] == self.num_tokens and H.shape[2] == self.width, (
                f"TokenLocalDecoder expected [B, {self.num_tokens}, {self.width}], "
                f"got {tuple(H.shape)}"
            )

        # ─── Per-token MLP. Linear applies over last dim, so [B, T, W] runs the
        # MLP independently on each of the B×T vectors with shared weights. ─────
        h1 = F.gelu(self.in_norm(self.in_linear(H)))        # [B, T, hidden_dim]
        h2 = F.gelu(self.mid_norm(self.mid_linear(h1)))     # [B, T, hidden_dim]
        raw = self.out_linear(h2)                           # [B, T, g_per_token * 14]
        raw = raw.reshape(B, self.total_output, self.NUM_OUT_PER_GAUSSIAN)
        raw = raw[:, :self.num_gaussians, :]                # [B, 40000, 14]

        # ─── Apply activations (mirror current GS_decoder behaviour exactly) ───
        pos   = raw[:, :, 0:3]
        color = (raw[:, :, 3:6] if self.color_residual
                 else raw[:, :, 3:6].clamp(0.0, 1.0))
        opac  = torch.sigmoid(raw[:, :, 6:7])
        scale = torch.exp(raw[:, :, 7:10])
        quat  = F.normalize(raw[:, :, 10:14], p=2, dim=-1)

        output = torch.cat([pos, color, opac, scale, quat], dim=-1)
        output_flat = output.reshape(B, -1)                 # [B, 40000*14] flat

        # ─── Per-Gaussian features for InfoNCE ─────────────────────────────────
        per_gaussian_features = None
        if return_per_gaussian and self.feat_proj is not None:
            # Project h2 to feat_dim: [B, T, hidden_dim] → [B, T, feat_dim]
            token_feats = self.feat_proj(h2)                # [B, T, feat_dim]

            # Repeat each token g_per_token times along a new dim:
            # [B, T, feat_dim] → [B, T, g_per_token, feat_dim]
            token_feats_repeated = token_feats.unsqueeze(2).expand(
                -1, -1, self.g_per_token, -1
            )

            # Add intra-token positional embedding (broadcast over B and T):
            # intra_token_pos_emb: [g_per_token, feat_dim] → [1, 1, g_per_token, feat_dim]
            # token_feats_repeated: [B, T, g_per_token, feat_dim]
            per_gauss_4d = token_feats_repeated + self.intra_token_pos_emb.view(
                1, 1, self.g_per_token, self.per_gaussian_feat_dim
            )

            # Flatten to [B, T * g_per_token, feat_dim] and crop to 40000:
            per_gaussian_features = per_gauss_4d.reshape(
                B, self.total_output, self.per_gaussian_feat_dim
            )[:, :self.num_gaussians, :]                    # [B, 40000, feat_dim]

        # ─── Legacy global hidden for SemanticProjectionHead compat ────────────
        hidden = None
        if return_hidden:
            pooled = h2.mean(dim=1)                         # [B, hidden_dim]
            hidden = self.hidden_proj(pooled)               # [B, 1024]

        # ─── Return shape depends on flags ─────────────────────────────────────
        if return_per_gaussian and return_hidden:
            return output_flat, per_gaussian_features, hidden
        elif return_per_gaussian:
            return output_flat, per_gaussian_features
        elif return_hidden:
            return output_flat, hidden
        else:
            return output_flat