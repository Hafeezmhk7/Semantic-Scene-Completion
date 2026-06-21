"""
Token-local decoder for Can3Tok.

REPLACES the flat GS_decoder MLP (~777M params, 1024-d bottleneck) with a
shared per-token MLP (~2M params, no bottleneck).

WHY (original motivation, still true):
  The flat GS_decoder compresses [B, 512, 384] -> flatten -> [B, 196608] -> Linear -> [B, 1024].
  This 1024-d bottleneck must hold ALL information for ALL 40,000 Gaussians of ALL
  scenes simultaneously. At 300 scenes this works via memorization. At 3800 scenes
  it fails because 1024 dims cannot encode enough scene-specific position information.

  The token-local decoder gives each Gaussian its own LOCAL 384-d context (its
  decoder token), with 512 such contexts running in parallel. Total information
  bandwidth is 512 x 384 = 196,608 dimensions instead of 1024. The transformer
  before the decoder already mixed information across tokens via self-attention,
  so each token's 384-d vector contains the relevant local geometry summary.

================================================================================
GRADIENT STARVATION FIX (separate_heads / head_cross_stitch -- NEW)
================================================================================
DIAGNOSIS (from V5/V6/V7 training logs):
  In the original (legacy SHARED) mode, ALL 14 output channels (pos[3], color[3],
  opacity[1], scale[3], quat[4]) are produced by ONE shared `out_linear`:
  `nn.Linear(hidden_dim, g_per_token * 14)`, initialised with trunc_normal_(std=0.01)
  and a zero bias. At init this makes every Gaussian/axis start at exp(0)=1 (a
  uniform isotropic sphere) by construction, and because position's raw L2
  magnitude dominates the other channels by 2-3 orders of magnitude (Pos ~500-7000
  vs Scl ~74 in observed runs), the rows of `out_linear` feeding scale/rotation get
  almost no gradient relative to the rows feeding position. The shared weight
  matrix is a literal gradient bottleneck: d(total_loss)/dW is a sum over ALL
  attribute losses through the SAME W, so the position loss's gradient dominates
  the direction the upstream shared layers get pushed toward, while scale/rotation
  sit in a near-flat, near-unmoving basin for hundreds of epochs (observed: Scl
  plateaus ~74-75 from epoch ~10 onward; Rot stuck ~1420-1485 throughout).

FIX -- separate_heads=True:
  Keep the shared trunk (in_linear -> mid_linear -> h2) EXACTLY as before -- h2
  still carries the per-token local geometry summary the transformer produced.
  Replace the single `out_linear` with FIVE independent small per-attribute
  heads (position, color, opacity, scale, quaternion), each its own
  Linear(hidden_dim -> head_hidden) -> GELU -> Linear(head_hidden -> g_per_token*k).
  Because each head's final projection is now a SEPARATE parameter tensor, the
  position loss's gradient can no longer write into the weights that produce
  scale or rotation (or vice versa) -- there is no shared matrix left to compete
  over. This directly removes the starvation mechanism rather than reweighting
  around it. (Literature: this is the standard "shared trunk, task-specific
  heads" MTL pattern; recent mechanistic work on multi-task grokking shows
  task-specific heads naturally evolve near-orthogonal weight subspaces once
  separated, which is why interference disappears on its own without needing
  any cross-talk mechanism in many cases -- see head_cross_stitch below for the
  fallback if that doesn't hold here.)

FIX (optional, layered on top) -- head_cross_stitch=True:
  If separate_heads alone causes the 5 heads to drift apart (e.g. predicted
  scale/rotation becomes locally inconsistent with predicted position -- splats
  whose orientation doesn't track local surface geometry), add a small amount of
  CONTROLLED interaction between the heads, following Cross-Stitch networks
  (Misra et al., CVPR 2016). Each head's pre-final-projection hidden vector
  (5 vectors of width head_hidden, one per attribute) is mixed via a LEARNED
  5x5 linear combination ("cross-stitch matrix") before the final per-head
  projection. The cross-stitch matrix is initialised to the identity, so at the
  start of training there is ZERO extra coupling (mathematically identical to
  separate_heads=True with no cross-stitch); the optimiser can then learn to
  couple any pair of heads only if doing so reduces the loss. This is the
  minimal-complexity way to let training itself decide where on the
  shared <-> independent spectrum each head should sit, rather than hand-fixing
  it. head_cross_stitch=True requires separate_heads=True (it is a refinement
  of the per-head architecture, not an independent option).

ARCHITECTURE (legacy SHARED mode, separate_heads=False -- UNCHANGED, default):
  Input:  H_out [B, 512, 384]   -- transformer output (one vector per decoder token)
  Per-token MLP applied independently (weights SHARED across tokens):
            Linear(384 -> 512) -> LN -> GELU                                (h1)
            Linear(512 -> 512) -> LN -> GELU                                (h2: per-token semantic context)
            Linear(512 -> GAUSSIANS_PER_TOKEN x 14)                        (raw Gaussian attrs)
  Reshape and crop to 40,000 Gaussians, apply activations.

ARCHITECTURE (separate_heads=True, head_cross_stitch=False):
  Same shared trunk -> h2. Then FIVE independent heads off h2:
            pos_head  : Linear(512 -> head_hidden) -> GELU -> Linear(head_hidden -> g*3)
            color_head: Linear(512 -> head_hidden) -> GELU -> Linear(head_hidden -> g*3)
            opa_head  : Linear(512 -> head_hidden) -> GELU -> Linear(head_hidden -> g*1)
            scale_head: Linear(512 -> head_hidden) -> GELU -> Linear(head_hidden -> g*3)
            quat_head : Linear(512 -> head_hidden) -> GELU -> Linear(head_hidden -> g*4)
  No parameter is shared between heads past h2.

ARCHITECTURE (separate_heads=True, head_cross_stitch=True):
  Same 5 heads, but their head_hidden-width "pre-projection" activations are
  stacked into [B, T, 5, head_hidden] and mixed by a learned 5x5 matrix
  (init = identity) along the head dimension BEFORE the final per-head linear
  projection. See CrossStitchUnit below.

OUTPUTS (matching original GS_decoder interface, unchanged regardless of mode):
  output_flat        : [B, 40000 x 14]   reconstructed Gaussians (flat).
  per_gaussian_feats : [B, 40000, feat_dim]   per-Gaussian features for InfoNCE.
                       Each Gaussian's feature comes from its parent token's h2,
                       optionally combined with a per-Gaussian positional embedding
                       so that 79 Gaussians from the same token get DISTINCT features.
  hidden (legacy)    : [B, 1024]   pooled hidden for backward-compat with code paths
                       that expect a global hidden vector (e.g. shape_embed-based
                       semantic head). Derived by mean-pooling h2 + projection.

PER-GAUSSIAN FEATURES (the important detail for InfoNCE) -- UNCHANGED:
  Naive approach: just repeat each token's h2 -> 79 copies. But then all 79 Gaussians
  inside one token get IDENTICAL features, defeating per-Gaussian InfoNCE.

  Better approach: add a learned per-position embedding indexed by intra-token slot
  (0..78). So feature[i] = h2[token_of(i)] + pos_emb[slot_of(i)]. This gives every
  Gaussian a unique semantic feature while keeping per-token coherence.

  This always reads h2 (the shared trunk output), so it is IDENTICAL regardless
  of separate_heads / head_cross_stitch -- the InfoNCE pipeline is untouched.

PARAMETER COUNT:
  legacy shared mode (separate_heads=False):     ~1.6M params (unchanged from before)
  separate_heads=True, head_hidden=128:          ~1.7M params (5 small heads instead
                                                   of 1 big shared out_linear; trunk
                                                   identical, so total is similar)
  + head_cross_stitch=True:                      +25 params (a single 5x5 matrix) --
                                                   negligible parameter cost.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Attribute order and output-channel widths. Order matters: it defines the
# concatenation order of the final 14-dim Gaussian vector (pos,color,opa,scale,quat),
# which must match the rest of the codebase (GEOMETRIC_INDICES, PARAM_SLICES, etc.).
_ATTR_SPECS = [
    ("position", 3),
    ("color",    3),
    ("opacity",  1),
    ("scale",    3),
    ("quat",     4),
]


class CrossStitchUnit(nn.Module):
    """
    Learned linear mixing across N parallel per-task hidden vectors, following
    Cross-Stitch Networks (Misra et al., CVPR 2016).

    Input:  stack of N task-specific hidden vectors, each [B, T, dim]
            -> stacked as [B, T, N, dim]
    Output: same shape, where output[..., n, :] = sum_m alpha[n, m] * input[..., m, :]

    The NxN mixing matrix `alpha` is initialised to the identity matrix, so at
    init this module is exactly a no-op (output == input, zero extra coupling).
    Training can then learn any amount of coupling between any pair of heads,
    from 0 (fully independent, recovers separate_heads with no cross-stitch) up
    to full sharing, whichever reduces the loss.
    """
    def __init__(self, n_tasks, dim):
        super().__init__()
        self.n_tasks = n_tasks
        self.dim     = dim
        # Parameterised as an explicit NxN matrix initialised to identity.
        self.alpha = nn.Parameter(torch.eye(n_tasks))
        print(f"[CrossStitchUnit] {n_tasks} heads x dim={dim} | "
              f"{n_tasks * n_tasks} params | init=identity (no-op at start)")

    def forward(self, stacked):
        # stacked: [B, T, N, dim] -> mix along the N (task) dimension.
        # einsum: out[b,t,n,d] = sum_m alpha[n,m] * stacked[b,t,m,d]
        return torch.einsum('nm,btmd->btnd', self.alpha, stacked)


class TokenLocalDecoder(nn.Module):
    """
    Per-token decoder applied independently to each of NUM_TOKENS decoder tokens.

    Forward returns (output_flat, per_gaussian_features [, hidden]) depending on flags.

    Parameters
    ----------
    width : int
        Per-token feature dim from the transformer (default 384).
    hidden_dim : int
        Hidden width of the shared trunk MLP (default 512).
    num_tokens : int
        Number of decoder tokens (default 512). Must match upstream transformer.
    num_gaussians : int
        Total Gaussians to output per scene (default 40,000).
    color_residual : bool
        If True, colors are residuals (no clamp). If False, colors are clamped to [0,1].
    per_gaussian_feat_dim : int
        Output dim of per-Gaussian features used by downstream InfoNCE losses.
        Default 128. Set to 0 to disable per-Gaussian features (returns None for that slot).
    separate_heads : bool
        STEP 2 FIX. If True, replace the single shared out_linear with five
        independent per-attribute heads (position/color/opacity/scale/quat), each
        with its own small Linear(hidden_dim -> head_hidden) -> GELU ->
        Linear(head_hidden -> g_per_token * k). Removes the shared-weight gradient
        bottleneck that starves scale/rotation. Default False (= legacy, backward
        compatible).
    head_hidden : int
        Hidden width of each per-attribute head's first projection (only used when
        separate_heads=True). Default 128. Five heads at 128 are still far smaller
        in aggregate than the original 777M flat decoder, and close in size to the
        legacy single out_linear (so this is mostly a re-routing of capacity, not a
        big expansion).
    head_cross_stitch : bool
        STEP 3 FIX (requires separate_heads=True). If True, mix the five heads'
        pre-projection hidden vectors with a learned 5x5 Cross-Stitch matrix
        (init = identity, i.e. zero coupling at the start of training) before each
        head's final linear projection. Use this only if separate_heads=True alone
        causes the predicted attributes to visibly drift apart (e.g. inconsistent
        scale/rotation vs. local position/geometry). Default False.
    """

    NUM_OUT_PER_GAUSSIAN = 14   # pos[3] + color[3] + opacity[1] + scale[3] + quat[4]
    LEGACY_HIDDEN_DIM    = 1024 # for backward-compat with code expecting [B, 1024]

    def __init__(self, width=384, hidden_dim=512, num_tokens=512,
                 num_gaussians=40_000, color_residual=False,
                 per_gaussian_feat_dim=128,
                 separate_heads=False, head_hidden=128,
                 head_cross_stitch=False):
        super().__init__()
        self.width                 = width
        self.hidden_dim            = hidden_dim
        self.num_tokens            = num_tokens
        self.num_gaussians         = num_gaussians
        self.color_residual        = color_residual
        self.per_gaussian_feat_dim = per_gaussian_feat_dim
        self.separate_heads        = separate_heads
        self.head_hidden           = head_hidden
        self.head_cross_stitch     = head_cross_stitch

        if head_cross_stitch and not separate_heads:
            raise ValueError(
                "head_cross_stitch=True requires separate_heads=True (cross-stitch "
                "mixes the per-attribute heads' hidden states; with separate_heads=False "
                "there are no separate heads to mix). Enable separate_heads.")

        # ceil(40000 / 512) = 79 -> 512 x 79 = 40,448 total outputs.
        self.g_per_token  = math.ceil(num_gaussians / num_tokens)
        self.total_output = self.num_tokens * self.g_per_token

        # --- Shared trunk (IDENTICAL in both modes) ---
        # Per-token MLP (weights SHARED across tokens via broadcast over [B,T,*]).
        self.in_linear  = nn.Linear(width, hidden_dim)
        self.in_norm    = nn.LayerNorm(hidden_dim)
        self.mid_linear = nn.Linear(hidden_dim, hidden_dim)
        self.mid_norm   = nn.LayerNorm(hidden_dim)

        if not separate_heads:
            # --- LEGACY: single shared output projection ---
            self.out_linear = nn.Linear(hidden_dim, self.g_per_token * self.NUM_OUT_PER_GAUSSIAN)
            # Small init so initial Gaussians have near-zero positions and
            # near-unit scales (exp(0)=1). Prevents explosion at init.
            nn.init.trunc_normal_(self.out_linear.weight, std=0.01)
            nn.init.zeros_(self.out_linear.bias)
            self.head_in_proj  = None
            self.head_out_proj = None
            self.cross_stitch  = None
        else:
            # --- SEPARATE HEADS (Step 2 fix) ---
            # One small "head_in_proj" (hidden_dim -> head_hidden) per attribute,
            # producing the per-head hidden vector that (optionally) gets mixed by
            # the cross-stitch unit, followed by a per-attribute final projection
            # "head_out_proj" (head_hidden -> g_per_token * k). Each attribute gets
            # its OWN parameters end-to-end past h2 -- no shared final layer.
            self.out_linear = None
            self.head_in_proj  = nn.ModuleDict()
            self.head_out_proj = nn.ModuleDict()
            for name, k in _ATTR_SPECS:
                in_proj  = nn.Linear(hidden_dim, head_hidden)
                out_proj = nn.Linear(head_hidden, self.g_per_token * k)
                # Same small-init convention as the legacy out_linear, applied
                # per-head, so every attribute starts near its activation's
                # "neutral" point (pos~0, color~0.5 after clamp, opacity~0.5
                # after sigmoid, scale~1 after exp, quat~arbitrary unit after norm).
                nn.init.trunc_normal_(out_proj.weight, std=0.01)
                nn.init.zeros_(out_proj.bias)
                self.head_in_proj[name]  = in_proj
                self.head_out_proj[name] = out_proj

            if head_cross_stitch:
                self.cross_stitch = CrossStitchUnit(n_tasks=len(_ATTR_SPECS), dim=head_hidden)
            else:
                self.cross_stitch = None

        # --- Per-Gaussian semantic feature head (UNCHANGED, reads h2 directly) ---
        # Produces a feat_dim-d feature for EVERY Gaussian (not just every token).
        # Mechanism: project h2 (per-token, [B, T, hidden_dim]) to feat_dim, then
        # add a learned intra-token position embedding indexed by slot in 0..g_per_token-1.
        # This gives every Gaussian a distinct feature while keeping per-token coherence.
        # Untouched by separate_heads / head_cross_stitch since it never reads
        # out_linear / head_out_proj -- only h2.
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

        # --- Legacy global hidden (for code paths expecting [B, 1024]) ---
        # When return_hidden=True, we mean-pool h2 across tokens and project to 1024.
        # This is for backward-compat with SemanticProjectionHead-style heads that
        # operate on a global hidden. NOT used for per-Gaussian InfoNCE -- that uses
        # per_gaussian_features above. Reads h2 only, so unaffected by either flag.
        self.hidden_proj = nn.Linear(hidden_dim, self.LEGACY_HIDDEN_DIM)
        nn.init.trunc_normal_(self.hidden_proj.weight, std=0.02)
        nn.init.zeros_(self.hidden_proj.bias)

        n_params = sum(p.numel() for p in self.parameters())
        _mode = ("SEPARATE-HEADS"
                 + (" + CROSS-STITCH" if head_cross_stitch else "")) if separate_heads \
                else "SHARED (legacy)"
        print(f"[TokenLocalDecoder] {num_tokens} tokens x {width} -> "
              f"{self.g_per_token} Gaussians/token x {self.NUM_OUT_PER_GAUSSIAN} attrs "
              f"({'residuals' if color_residual else 'clamp(0,1)'}) | "
              f"mode={_mode}"
              f"{f' head_hidden={head_hidden}' if separate_heads else ''} | "
              f"per_gauss_feat_dim={per_gaussian_feat_dim} | "
              f"{n_params/1e6:.2f}M params")

    def _forward_legacy_out(self, h2, B):
        """Legacy shared out_linear path. Returns raw [B, total_output, 14]."""
        raw = self.out_linear(h2)                           # [B, T, g_per_token * 14]
        return raw.reshape(B, self.total_output, self.NUM_OUT_PER_GAUSSIAN)

    def _forward_separate_heads(self, h2, B):
        """
        Separate per-attribute heads path. Returns raw [B, total_output, 14] with
        channels concatenated in the SAME order as _ATTR_SPECS (pos,color,opa,scale,quat),
        which matches NUM_OUT_PER_GAUSSIAN's documented layout.
        """
        # Step 1: per-head first projection (hidden_dim -> head_hidden), GELU.
        # h_in[name]: [B, T, head_hidden]
        h_in = {name: F.gelu(self.head_in_proj[name](h2)) for name, _ in _ATTR_SPECS}

        # Step 2 (optional): Cross-Stitch mixing across the 5 heads' hidden states.
        if self.cross_stitch is not None:
            # Stack in the fixed _ATTR_SPECS order -> [B, T, 5, head_hidden]
            stacked = torch.stack([h_in[name] for name, _ in _ATTR_SPECS], dim=2)
            mixed   = self.cross_stitch(stacked)             # [B, T, 5, head_hidden]
            h_in    = {name: mixed[:, :, idx, :]
                       for idx, (name, _) in enumerate(_ATTR_SPECS)}

        # Step 3: per-head final projection -> [B, T, g_per_token * k] each.
        outs = []
        for name, k in _ATTR_SPECS:
            raw_k = self.head_out_proj[name](h_in[name])     # [B, T, g_per_token * k]
            raw_k = raw_k.reshape(B, self.num_tokens, self.g_per_token, k)
            outs.append(raw_k)

        # Concatenate along the channel dim -> [B, T, g_per_token, 14], matching
        # the documented (pos,color,opa,scale,quat) layout, then reshape to the
        # same [B, total_output, 14] shape the legacy path produces.
        raw = torch.cat(outs, dim=-1)                        # [B, T, g_per_token, 14]
        return raw.reshape(B, self.total_output, self.NUM_OUT_PER_GAUSSIAN)

    def forward(self, H_out_or_flat, return_hidden=False, return_per_gaussian=False):
        """
        Forward pass.

        Parameters
        ----------
        H_out_or_flat : Tensor
            Either [B, num_tokens, width] (token-shaped) or
            [B, num_tokens * width] (flat -- backward compat with existing call sites).
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

        # --- Shared trunk. Linear applies over last dim, so [B, T, W] runs the
        # MLP independently on each of the B x T vectors with shared weights. ---
        h1 = F.gelu(self.in_norm(self.in_linear(H)))        # [B, T, hidden_dim]
        h2 = F.gelu(self.mid_norm(self.mid_linear(h1)))     # [B, T, hidden_dim]

        if self.separate_heads:
            raw = self._forward_separate_heads(h2, B)        # [B, total_output, 14]
        else:
            raw = self._forward_legacy_out(h2, B)             # [B, total_output, 14]

        raw = raw[:, :self.num_gaussians, :]                # [B, 40000, 14]

        # --- Apply activations (IDENTICAL in both modes -- only the path that
        # produced `raw` differs) ---
        pos   = raw[:, :, 0:3]
        color = (raw[:, :, 3:6] if self.color_residual
                 else raw[:, :, 3:6].clamp(0.0, 1.0))
        opac  = torch.sigmoid(raw[:, :, 6:7])
        scale = torch.exp(raw[:, :, 7:10])
        quat  = F.normalize(raw[:, :, 10:14], p=2, dim=-1)

        output = torch.cat([pos, color, opac, scale, quat], dim=-1)
        output_flat = output.reshape(B, -1)                 # [B, 40000*14] flat

        # --- Per-Gaussian features for InfoNCE (UNCHANGED -- always reads h2) ---
        per_gaussian_features = None
        if return_per_gaussian and self.feat_proj is not None:
            # Project h2 to feat_dim: [B, T, hidden_dim] -> [B, T, feat_dim]
            token_feats = self.feat_proj(h2)                # [B, T, feat_dim]

            # Repeat each token g_per_token times along a new dim:
            # [B, T, feat_dim] -> [B, T, g_per_token, feat_dim]
            token_feats_repeated = token_feats.unsqueeze(2).expand(
                -1, -1, self.g_per_token, -1
            )

            # Add intra-token positional embedding (broadcast over B and T):
            # intra_token_pos_emb: [g_per_token, feat_dim] -> [1, 1, g_per_token, feat_dim]
            # token_feats_repeated: [B, T, g_per_token, feat_dim]
            per_gauss_4d = token_feats_repeated + self.intra_token_pos_emb.view(
                1, 1, self.g_per_token, self.per_gaussian_feat_dim
            )

            # Flatten to [B, T * g_per_token, feat_dim] and crop to 40000:
            per_gaussian_features = per_gauss_4d.reshape(
                B, self.total_output, self.per_gaussian_feat_dim
            )[:, :self.num_gaussians, :]                    # [B, 40000, feat_dim]

        # --- Legacy global hidden for SemanticProjectionHead compat (UNCHANGED) ---
        hidden = None
        if return_hidden:
            pooled = h2.mean(dim=1)                         # [B, hidden_dim]
            hidden = self.hidden_proj(pooled)                # [B, 1024]

        # --- Return shape depends on flags ---
        if return_per_gaussian and return_hidden:
            return output_flat, per_gaussian_features, hidden
        elif return_per_gaussian:
            return output_flat, per_gaussian_features
        elif return_hidden:
            return output_flat, hidden
        else:
            return output_flat