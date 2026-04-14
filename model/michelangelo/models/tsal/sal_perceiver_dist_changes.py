# -*- coding: utf-8 -*-
"""
sal_perceiver_dist_changes.py  —  Can3Tok VAE
==============================================

DECODER STRATEGIES (controlled by flags, all backward-compatible):

  A  latent_disentangle=True, decoder_zs_cross_attn=False  [BEST PERFORMANCE]
       Z = [z_layout(0:16) | z_geo(16:512)] → all 512 tokens in decoder sequence
       Self-attention mixes layout & geometry freely at every layer

  B1 decoder_layout_cross_attn=True   [NEW — 512 geom + cross-attn conditioning]
       Z [B,512,32] = full geometry  (latent_disentangle not required)
       z_layout [B,16,32] from Layout16Projector(shape_embed) — SEPARATE from Z
       Decoder: 512 geometry tokens in sequence
       Conditioning: z_layout as cross-attn K/V at every transformer layer
       Key difference from failed decoder_zs_cross_attn: geometry = FULL 512 not 496

  B2 decoder_layout_additive=True   [NEW — 512 geom + additive bias conditioning]
       Same z_layout as B1
       Conditioning: flatten(z_layout) → MLP → [B,384] added as broadcast bias
       before the transformer (once, not per-layer)
       Simpler than B1; lower capacity but faster and more stable

  B3 Both B1+B2 simultaneously: additive bias before stack + cross-attn per layer
       Strongest conditioning; test if additive + cross-attn compound

  C  baseline (no flags): 512 geometry tokens, no layout conditioning

  D  decoder_zs_cross_attn=True [FAILED — kept for reference]
       496 geom tokens + 16 z_s as cross-attn K/V → worst performance

ALL B strategies:
  z_layout [B,16,32] gets structure losses: MeanColorHead, SemanticHead, LayoutHead
  InfoNCE: flatten(z_layout) [B,512] → MLP → [B,128] L2-norm → prototype NCE
  PCA vis: zs_tokens_epoch_NNN.ply  (B×16 points, colored by dominant category)

MAIN NEW IDEA: --decoder_zs_cross_attn
  z_g tokens [B, 496, 32] are the ONLY decoder input sequence.
  z_s tokens [B,  16, 32] condition every decoder transformer layer via cross-attention.

  OLD DESIGN (decoder_zs_cross_attn=False, backward compatible):
    Z [B, 512, 32] → post_kl → transformer(self-attn only) → GS_decoder(512×384)

  NEW DESIGN (decoder_zs_cross_attn=True):
    z_g [B, 496, 32] → post_kl_g [B, 496, 384] → FourierPE
    z_s [B,  16, 32] → post_kl_s [B,  16, 384]
    For each of the 12 transformer layers:
        H = self_attn(H_g)                           z_g attends to z_g
        H = cross_attn(Q=H, K=H_s, V=H_s)           z_g reads from z_s
        H = FFN(H)
    GS_decoder(flatten(H))  — input dim 496×384 = 190,464

  WHY CROSS-ATTN NOT ADALN:
    AdaLN makes every affine in the decoder depend on z_s → Run 1 showed >400× swap ratio.
    Cross-attention is a soft gate: if z_g is self-sufficient, attention weights → 0.
    The decoder consults z_s when helpful; it is never forced to depend on it for geometry.
    This preserves disentanglement while still passing semantic context.

  The L_cross_recon / L_ortho losses are still supported (now optional since the
  architecture itself enforces the separation) and active by default to additionally
  strengthen z_g geometry-sufficiency.

GRADIENT PATHS
  PATH 1 — Reconstruction:      L_recon → GS_decoder → ZSCondDecoder → post_kl_g → z_g → encoder
  PATH 2 — KL:                  L_KL → mu, log_var → encoder
  PATH 3 — Mean Color:          L_color → MeanColorHead → z[:,0,:]
  PATH 4 — Scene Semantic:      L_sem_kl → SceneSemanticHead → z[:,1:16,:]
  PATH 5 — Layout Centroids:    L_layout → SceneLayoutHead → z[:,1:16,:]
  PATH 6 — Cross-attn gradient: L_recon flows through cross-attn weights → post_kl_s → z_s → encoder
  PATH 7 — Per-Gaussian InfoNCE:L_infonce → SemanticProjectionHead → decoder hidden
  PATH 8 — Scene z_s InfoNCE:   L_z_s_nce → SemanticTokenInfoNCEHead → z_s → encoder
  PATH 9 — Cross-recon:         L_cross_recon → decoder(z_cross) → z_g (geometry-sufficiency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
import numpy as np

from model.michelangelo.models.modules import checkpoint
from model.michelangelo.models.modules.embedder import FourierEmbedder
from model.michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from model.michelangelo.models.modules.transformer_blocks import (
    ResidualCrossAttentionBlock, Transformer)
from .tsal_base import ShapeAsLatentModule


# ============================================================================
# AUXILIARY HEADS
# ============================================================================

class MeanColorHead(nn.Module):
    def __init__(self, in_dim=32):
        super().__init__()
        hidden = max(32, min(64, in_dim))
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 3), nn.Sigmoid())
        print(f"[MeanColorHead] [{in_dim}]→[3] | "
              f"{sum(p.numel() for p in self.parameters()):,} params")
    def forward(self, x): return self.head(x)


class SceneSemanticHead(nn.Module):
    NUM_LABELS = 72
    def __init__(self, in_dim=480):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, self.NUM_LABELS))
        print(f"[SceneSemanticHead] [{in_dim}]→[72] | "
              f"{sum(p.numel() for p in self.parameters()):,} params")
    def forward(self, x): return F.softmax(self.head(x), dim=-1)


class SceneLayoutHead(nn.Module):
    NUM_CATS = 72
    def __init__(self, in_dim=480):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, self.NUM_CATS * 3))
        print(f"[SceneLayoutHead] [{in_dim}]→[72,3] | "
              f"{sum(p.numel() for p in self.parameters()):,} params")
    def forward(self, x):
        return self.head(x).reshape(x.shape[0], self.NUM_CATS, 3)


# ============================================================================
# Z_S INFONCE HEAD
# ============================================================================

class SemanticTokenInfoNCEHead(nn.Module):
    """
    z_s [B, semantic_dims] → L2-norm [B, 128].
    No LayerNorm between linears to preserve gradient magnitude reaching z_s.
    """
    def __init__(self, in_dim=512, proj_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, proj_dim))
        print(f"[SemanticTokenInfoNCEHead] [{in_dim}]→[{proj_dim}] L2-norm | "
              f"{sum(p.numel() for p in self.parameters()):,} params")
    def forward(self, z_s_flat):
        return F.normalize(self.head(z_s_flat), p=2, dim=-1)


# ============================================================================
# STRATEGY B: LAYOUT CONDITIONING COMPONENTS
# ============================================================================

class Layout16Projector(nn.Module):
    """
    Projects shape_embed [B, width] → z_layout [B, n_tokens, embed_dim].

    z_layout is SEPARATE from the main latent Z — it is an additional global
    conditioning signal derived from shape_embed, not part of the VAE latent.
    This means the reconstruction path (Z) is completely unmodified.

    Enables Strategy B1 (cross-attn) and B2 (additive), or both simultaneously.

    At second-stage inference:
      Stage 1 DiT generates z_layout [B, 15, 32] conditioned on text/class.
      Token 0 = color token; tokens 1-15 = layout tokens.
      Stage 2 DiT denoises z_geo [B, 496, 32] conditioned on z_layout.
    """
    def __init__(self, in_dim=384, n_tokens=16, token_dim=32):
        super().__init__()
        self.n_tokens  = n_tokens
        self.token_dim = token_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, n_tokens * token_dim))
        print(f"[Layout16Projector] [{in_dim}]→[{n_tokens},{token_dim}] | "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, shape_embed):
        B = shape_embed.shape[0]
        return self.proj(shape_embed).reshape(B, self.n_tokens, self.token_dim)


class LayoutAdditiveConditioner(nn.Module):
    """
    Strategy B2: projects z_layout to a single broadcast bias vector.

    flatten(z_layout) [B, n_tokens*token_dim=512]
      → Linear(512→width) → LayerNorm → ReLU → Linear(width→width)
      → [B, width]  (broadcast over all 512 decoder positions)

    Applied ONCE before the transformer, unlike cross-attention which applies
    per layer. Lower capacity but more stable and faster.

    Can be combined with cross-attention (Strategy B3):
      H = H + additive_bias          (once, before stack)
      for each layer: H += cross_attn(Q=H, K=z_layout, V=z_layout)
    """
    def __init__(self, n_tokens=16, token_dim=32, width=384):
        super().__init__()
        in_dim = n_tokens * token_dim   # 512
        self.proj = nn.Sequential(
            nn.Linear(in_dim, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width))
        print(f"[LayoutAdditiveConditioner] [{in_dim}]→[{width}] broadcast bias | "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, z_layout):
        B = z_layout.shape[0]
        flat = z_layout.reshape(B, -1)   # [B, 512]
        return self.proj(flat)           # [B, width]


# ============================================================================
# FOURIER DECODER PE
# ============================================================================

class FourierDecoderPE(nn.Module):
    """3D Fourier PE over 8×8×8 scaffold grid. Handles 512 or 496 tokens."""
    SCAFFOLD_DIMS = 8

    def __init__(self, fourier_embedder, width, num_tokens=512):
        super().__init__()
        S = self.SCAFFOLD_DIMS
        coords = []
        for i in range(S):
            for j in range(S):
                for k in range(S):
                    coords.append([(2*i/(S-1))-1, (2*j/(S-1))-1, (2*k/(S-1))-1])
        all_coords = torch.tensor(coords, dtype=torch.float32)  # [512, 3]
        # Skip the first (512 - num_tokens) voxels, which are the z_s voxels.
        # Works for any num_tokens, not just the hardcoded 496 case.
        if num_tokens != 512:
            n_skip = 512 - num_tokens
            all_coords = all_coords[n_skip:]  # [num_tokens, 3]
        self.register_buffer('voxel_coords', all_coords)
        assert all_coords.shape[0] == num_tokens, \
            f"FourierDecoderPE: expected {num_tokens} coords, got {all_coords.shape[0]}"
        self.fourier_embedder = fourier_embedder
        self.proj = nn.Linear(fourier_embedder.out_dim, width)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        print(f"[FourierDecoderPE] {num_tokens} tokens | "
              f"{sum(p.numel() for p in self.proj.parameters()):,} params")

    def forward(self, B, device):
        fourier = self.fourier_embedder(self.voxel_coords.to(device).unsqueeze(0))
        return self.proj(fourier).expand(B, -1, -1)


# ============================================================================
# MAIN NEW IDEA: Z_S CROSS-ATTENTION CONDITIONED DECODER TRANSFORMER
# ============================================================================

class ZSCondTransformerBlock(nn.Module):
    """
    Transformer block: self-attention on z_g + cross-attention with z_s.

    x   = z_g sequence [B, 496, width]
    z_s = semantic tokens [B, 16, width] — key/value for cross-attn
    """
    def __init__(self, width, heads):
        super().__init__()
        self.norm_sa  = nn.LayerNorm(width)
        self.norm_ca  = nn.LayerNorm(width)
        self.norm_ff  = nn.LayerNorm(width)
        self.self_attn  = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x, z_s_kv):
        # Self-attention among z_g tokens
        h, _ = self.self_attn(self.norm_sa(x), self.norm_sa(x), self.norm_sa(x))
        x = x + h
        # Cross-attention: z_g queries, z_s keys/values
        h, _ = self.cross_attn(self.norm_ca(x), z_s_kv, z_s_kv)
        x = x + h
        # FFN
        x = x + self.ffn(self.norm_ff(x))
        return x


class ZSCondTransformerDecoder(nn.Module):
    """
    N-layer decoder where z_s conditions every layer via cross-attention.
    Input:  z_g [B, 496, width],  z_s [B, 16, width]
    Output: [B, 496, width]
    """
    def __init__(self, width, heads, layers):
        super().__init__()
        self.blocks   = nn.ModuleList(
            [ZSCondTransformerBlock(width, heads) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(width)
        total = sum(p.numel() for p in self.parameters())
        print(f"[ZSCondTransformerDecoder] {layers}× ZSCondTransformerBlock "
              f"(width={width}, heads={heads}) | {total/1e6:.2f}M params")
        print(f"  Decoder input: z_g [B, 496, {width}]  (geometry tokens only)")
        print(f"  Conditioning:  z_s [B,  16, {width}]  (cross-attn K/V per layer)")

    def forward(self, x, z_s_kv):
        for block in self.blocks:
            x = block(x, z_s_kv)
        return self.norm_out(x)


# ============================================================================
# ADALN DECODER (legacy, kept for ablation comparison)
# ============================================================================

def _modulate(h, shift, scale):
    return h * (1.0 + scale) + shift

class AdaLNBlock(nn.Module):
    def __init__(self, width, heads, cond_dim):
        super().__init__()
        self.norm1     = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2     = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn      = nn.MultiheadAttention(width, heads, batch_first=True, bias=False)
        self.ffn       = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6*width, bias=True))
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x, c):
        B, T, W = x.shape
        mod = self.adaLN_mod(c.reshape(B*T, -1)).reshape(B, T, 6*W)
        sh_a, sc_a, ga, sh_f, sc_f, gf = mod.chunk(6, dim=-1)
        h_a, _ = self.attn(_modulate(self.norm1(x), sh_a, sc_a),
                            _modulate(self.norm1(x), sh_a, sc_a),
                            _modulate(self.norm1(x), sh_a, sc_a))
        x = x + ga * h_a
        x = x + gf * self.ffn(_modulate(self.norm2(x), sh_f, sc_f))
        return x

class AdaLNTransformerDecoder(nn.Module):
    def __init__(self, width, heads, layers, cond_dim):
        super().__init__()
        self.blocks   = nn.ModuleList([AdaLNBlock(width, heads, cond_dim) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(width)
        total = sum(p.numel() for p in self.parameters())
        print(f"[AdaLNTransformerDecoder] {layers}× AdaLNBlock | {total/1e6:.2f}M params")
    def forward(self, x, c):
        for block in self.blocks: x = block(x, c)
        return self.norm_out(x)


# ============================================================================
# ANCHOR PREDICTION FROM TOKENS
# ============================================================================

class AnchorPredFromTokens(nn.Module):
    def __init__(self, width=384, num_tokens=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.head = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),   nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 3))
        print(f"[AnchorPredFromTokens] [B,{num_tokens},{width}]→[B,{num_tokens},3] | "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, transformer_tokens):
        B, T, W = transformer_tokens.shape
        return self.head(transformer_tokens.reshape(B*T, W)).reshape(B, T, 3)


_N_GAUSSIANS  = 40_000
FIXED_TOKEN_IDS_512 = torch.arange(_N_GAUSSIANS) * 512 // _N_GAUSSIANS
FIXED_TOKEN_IDS_496 = torch.arange(_N_GAUSSIANS) * 496 // _N_GAUSSIANS


# ============================================================================
# PER-GAUSSIAN INFONCE HEADS (decoder-output level)
# ============================================================================

class SegPredHead(nn.Module):
    NUM_CATS = 72
    def __init__(self, in_dim=14, num_cats=72):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, num_cats))
    def forward(self, g):
        B, N, D = g.shape
        return self.head(g.reshape(B*N, D)).reshape(B, N, self.NUM_CATS)

class TokenCondMLP(nn.Module):
    def __init__(self, fourier_dim, width):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width))
    def forward(self, fe):
        B, T, D = fe.shape
        return self.mlp(fe.reshape(B*T, D)).reshape(B, T, -1)

class SemanticProjectionHead(nn.Module):
    def __init__(self, hidden_dim=1024, num_gaussians=40000, feature_dim=32):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim   = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, num_gaussians * feature_dim))
        print(f"[SemanticProjectionHead] [{hidden_dim}]→[{num_gaussians},{feature_dim}] | "
              f"{sum(p.numel() for p in self.parameters())/1e6:.3f}M params")
    def forward(self, hidden):
        B = hidden.shape[0]
        return F.normalize(
            self.projection(hidden).reshape(B, self.num_gaussians, self.feature_dim),
            p=2, dim=-1)

class SemanticDistributionHead(nn.Module):
    def __init__(self, hidden_dim=1024, num_labels=72):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, num_labels))
    def forward(self, hidden): return self.head(hidden)

class SemanticProjectionHeadGeometric(nn.Module):
    def __init__(self, gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.projection = nn.Sequential(
            nn.Linear(gaussian_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim))
    def forward(self, gaussians):
        B, N, D = gaussians.shape
        return F.normalize(
            self.projection(gaussians.reshape(B*N, D)).reshape(B, N, -1), p=2, dim=-1)


# ============================================================================
# GS DECODER MLP — configurable num_tokens
# ============================================================================

class GS_decoder(nn.Module):
    """
    Flat MLP: flatten(transformer_output) → 40000×14 Gaussian attributes.
    num_tokens×width is the input dimension.
    num_tokens=512 for old design, 496 for new design.
    """
    def __init__(self, D=8, W=256, num_tokens=512, width=384, color_residual=False):
        super().__init__()
        input_ch           = num_tokens * width
        self.color_residual = color_residual
        self.pts_linears    = nn.ModuleList([nn.Linear(input_ch, W)])
        for _ in range(D - 1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.LayerNorm(W))
            self.pts_linears.append(nn.ReLU())
        self.output_linear = nn.Linear(W, 40_000 * 14)
        print(f"  GS_DECODER ({num_tokens} tokens): {num_tokens}×{width}={input_ch} "
              f"→ 40000×14  "
              f"({'residuals' if color_residual else 'clamp(0,1)'})")

    def forward(self, x, return_hidden=False):
        for layer in self.pts_linears: x = layer(x)
        hidden = x
        raw    = self.output_linear(x).reshape(x.shape[0], 40_000, 14)
        pos    = raw[:, :, 0:3]
        color  = raw[:, :, 3:6] if self.color_residual else raw[:, :, 3:6].clamp(0., 1.)
        opac   = torch.sigmoid(raw[:, :, 6:7])
        scale  = torch.exp(raw[:, :, 7:10])
        quat   = F.normalize(raw[:, :, 10:14], p=2, dim=-1)
        out    = torch.cat([pos, color, opac, scale, quat], dim=-1).reshape(x.shape[0], -1)
        return (out, hidden) if return_hidden else out


# ============================================================================
# ENCODER
# ============================================================================

class CrossAttentionEncoder(nn.Module):
    def __init__(self, *, device, dtype, num_latents, fourier_embedder,
                 fourier_embedder_ID, point_feats, width, heads, layers,
                 init_scale=0.25, qkv_bias=True, flash=False,
                 use_ln_post=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint      = use_checkpoint
        self.fourier_embedder    = fourier_embedder
        self.fourier_embedder_ID = fourier_embedder_ID

        voxel_reso = 4
        x_y  = np.linspace(-8, 8, voxel_reso)
        xv, yv, zv = np.meshgrid(x_y, x_y, x_y, indexing='ij')
        voxel_centers = torch.tensor(
            np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T,
            device=device, dtype=dtype).reshape([-1, 3])
        dummy = torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02
        dummy[:, :192] = voxel_centers.reshape([-1]) * 0.01
        self.query = nn.Parameter(dummy)
        self.input_proj = nn.Linear(
            fourier_embedder.out_dim + point_feats + fourier_embedder_ID.out_dim,
            width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash)
        self.self_attn = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=width,
            layers=layers, heads=heads, init_scale=init_scale,
            qkv_bias=qkv_bias, flash=flash, use_checkpoint=False)
        self.ln_post = (nn.LayerNorm(width, dtype=dtype, device=device)
                        if use_ln_post else None)

    def _forward(self, pc, feats):
        bs = pc.shape[0]
        data = torch.cat([
            self.fourier_embedder(pc[:, :, 4:7]),
            self.fourier_embedder_ID(pc[:, :, 0:3]),
            feats[:, :, 7:],
        ], dim=-1).to(dtype=torch.float32)
        data    = self.input_proj(data)
        query   = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)
        if self.ln_post: latents = self.ln_post(latents)
        return latents, pc

    def forward(self, pc, feats=None):
        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


class CrossAttentionDecoder(nn.Module):
    """Occupancy/SDF decoder, unchanged."""
    def __init__(self, *, device, dtype, num_latents, out_channels, fourier_embedder,
                 width, heads, init_scale=0.25, qkv_bias=True, flash=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint   = use_checkpoint
        self.fourier_embedder = fourier_embedder
        self.query_proj = nn.Linear(fourier_embedder.out_dim, width, device=device, dtype=dtype)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, n_data=num_latents, width=width,
            heads=heads, init_scale=init_scale, qkv_bias=qkv_bias, flash=flash)
        self.ln_post     = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries, latents):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        return self.output_proj(self.ln_post(x))

    def forward(self, queries, latents):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


# ============================================================================
# BASE PERCEIVER
# ============================================================================

class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False, color_residual=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_latents    = num_latents
        self.fourier_embedder    = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi, input_dim=3)
        self.fourier_embedder_ID = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi, input_dim=3)
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder,
            fourier_embedder_ID=self.fourier_embedder_ID, num_latents=num_latents,
            point_feats=point_feats, width=width, heads=heads, layers=num_encoder_layers,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_ln_post=use_ln_post, use_checkpoint=use_checkpoint)
        self.embed_dim = embed_dim
        if embed_dim > 0:
            self.pre_kl  = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)
        self.transformer = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=width,
            layers=num_decoder_layers, heads=heads, init_scale=init_scale,
            qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint)
        # Default GS decoder (512 tokens, old design — overridden in AlignedShapeLatentPerceiver)
        self.GS_decoder = GS_decoder(3, 1024, num_tokens=512, width=width,
                                     color_residual=color_residual)
        self.kl_emb_proj_mean = nn.Linear((num_latents - 1) * embed_dim, 64 * 64 * 4)
        self.kl_emb_proj_var  = nn.Linear((num_latents - 1) * embed_dim, 64 * 64 * 4)
        self.geo_decoder = CrossAttentionDecoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder,
            out_channels=1, num_latents=num_latents, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_checkpoint=use_checkpoint)

    def encode(self, pc, feats=None, sample_posterior=True):
        latents, center_pos = self.encoder(pc, feats)
        posterior = None
        if self.embed_dim > 0:
            moments   = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            latents   = posterior.sample() if sample_posterior else posterior.mode()
        return latents, center_pos, posterior

    def decode(self, latents, volume_queries=None):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return self.GS_decoder(latents.reshape(latents.shape[0], -1))

    def query_geometry(self, queries, latents):
        return self.geo_decoder(queries, latents).squeeze(-1)

    def forward(self, pc, feats, volume_queries, sample_posterior=True):
        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior)
        return self.decode(latents), center_pos, posterior


# ============================================================================
# ALIGNED SHAPE LATENT PERCEIVER — Full Can3Tok Model
# ============================================================================

class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):

    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False,
                 semantic_mode='none',
                 color_residual=False,
                 scene_semantic_head=False,
                 position_scaffold=False,
                 latent_disentangle=False,
                 semantic_dims=512,
                 scene_layout_head=False,
                 decoder_pos_enc=False,
                 predict_seg_labels=False,
                 token_cond=False,
                 token_cond_approach='B',
                 decoder_fourier_pe=False,
                 token_cond_adaln=False,
                 semantic_token_heads=False,
                 # ── MAIN NEW IDEA (failed, kept for reference) ───────────
                 decoder_zs_cross_attn=False,
                 # ── STRATEGY B (new) ─────────────────────────────────────
                 decoder_layout_cross_attn=False,
                 decoder_layout_additive=False,
                 # ── STRUCTURED TOKEN SPLIT ───────────────────────────────
                 structured_layout_tokens=False,
                 # Legacy flags for backward compat
                 position_layout_residual=False,
                 jepa_idea1=False,
                 query_decoder=False):

        # num_latents passed to super is already the full 512 (511 geom + 1 shape_embed)
        super().__init__(
            device=device, dtype=dtype, num_latents=1 + num_latents,
            point_feats=point_feats, embed_dim=embed_dim,
            num_freqs=num_freqs, include_pi=include_pi, width=width, heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_ln_post=use_ln_post, use_checkpoint=use_checkpoint,
            color_residual=color_residual)

        self.width                   = width
        self.semantic_mode           = semantic_mode
        self.color_residual          = color_residual
        self.scene_semantic_flag     = scene_semantic_head
        self.position_scaffold       = position_scaffold
        self.latent_disentangle      = latent_disentangle
        self.semantic_dims           = semantic_dims
        self.decoder_pos_enc_flag    = decoder_pos_enc
        self.predict_seg_labels_flag = predict_seg_labels
        self.token_cond_flag         = token_cond
        self.token_cond_approach     = token_cond_approach.upper()
        self.query_decoder_flag      = query_decoder
        self.decoder_zs_cross_attn    = decoder_zs_cross_attn
        self.decoder_layout_cross_attn = decoder_layout_cross_attn
        self.decoder_layout_additive       = decoder_layout_additive
        self.structured_layout_tokens_flag = structured_layout_tokens

        # Token counts in the LATENT SPACE z.
        # z is always [B, 16384] = [B, 512, 32] when latent_disentangle=True,
        # regardless of the encoder's num_latents (which only sets encoder depth).
        _Z_TOKENS         = 16384 // embed_dim                 # always 512
        self._n_zs_tokens = semantic_dims // embed_dim         # 512 // 32 = 16
        self._n_zg_tokens = _Z_TOKENS - self._n_zs_tokens      # 512 - 16  = 496

        print(f"\n{'='*70}")
        print(f"  CAN3TOK")
        print(f"  ── STRATEGY FLAGS ───────────────────────────────────────────────")
        print(f"  decoder_zs_cross_attn    = {decoder_zs_cross_attn}  (Strategy D, failed)")
        print(f"  decoder_layout_cross_attn= {decoder_layout_cross_attn}  (Strategy B1, NEW)")
        print(f"  decoder_layout_additive  = {decoder_layout_additive}  (Strategy B2, NEW)")
        _strat = "A" if latent_disentangle and not decoder_zs_cross_attn                  else ("B1+B2" if decoder_layout_cross_attn and decoder_layout_additive                  else ("B1" if decoder_layout_cross_attn                  else ("B2" if decoder_layout_additive                  else ("D" if decoder_zs_cross_attn else "C (baseline)"))))
        print(f"  Active strategy: {_strat}")
        if decoder_zs_cross_attn:
            print(f"  z_s: {self._n_zs_tokens} tokens → cross-attn K/V (NOT in decoder sequence)")
            print(f"  z_g: {self._n_zg_tokens} tokens → decoder input sequence")
            print(f"  Each layer: self_attn(z_g) + cross_attn(Q=z_g,K=z_s,V=z_s) + FFN")
        else:
            print(f"  LEGACY: all 512 tokens → decoder (backward compatible)")
        print(f"  semantic_mode={semantic_mode} | color_residual={color_residual}")
        print(f"  latent_disentangle={latent_disentangle}  semantic_dims={semantic_dims}")
        print(f"  scene_layout_head={scene_layout_head}")
        print(f"  decoder_fourier_pe={decoder_fourier_pe}")
        print(f"  token_cond={token_cond} adaln={token_cond_adaln}")
        print(f"  semantic_token_heads={semantic_token_heads}")
        print(f"{'='*70}")

        # ── HEAD INPUT DIMENSIONS ─────────────────────────────────────────────
        if semantic_token_heads and not latent_disentangle:
            raise ValueError("semantic_token_heads requires latent_disentangle=True")
        self.semantic_token_heads_flag = semantic_token_heads

        # ── STRUCTURED TOKEN SPLIT ───────────────────────────────────────────
        # Token split constants (used when structured_layout_tokens=True)
        _N_SEM_TOKENS = 8   # tokens 1-8  → SceneSemanticHead
        _N_LAY_TOKENS = 7   # tokens 9-15 → SceneLayoutHead
        self._n_sem_tokens = _N_SEM_TOKENS
        self._n_lay_tokens = _N_LAY_TOKENS

        # WITHOUT structured_layout_tokens (default):
        #   Both heads receive tokens 1-15 flattened [B, 480] — same floats, interference
        # WITH structured_layout_tokens=True:
        #   SceneSemanticHead → tokens 1-8  only [B, 8×32=256] — exclusive
        #   SceneLayoutHead   → tokens 9-15 only [B, 7×32=224] — exclusive
        #   Each head's gradient reaches only its own token range, no cross-contamination

        if semantic_token_heads or structured_layout_tokens:
            _color_in = embed_dim
            if structured_layout_tokens:
                _sem_in = _N_SEM_TOKENS * embed_dim   # 8×32 = 256
                _lay_in = _N_LAY_TOKENS * embed_dim   # 7×32 = 224
            else:
                # semantic_token_heads without split: both on full 480
                _sem_in = semantic_dims - embed_dim   # 480
                _lay_in = semantic_dims - embed_dim   # 480 (same — interference)
        else:
            # legacy: heads receive shape_embed [B, width=384]
            _color_in = width
            _sem_in   = width
            _lay_in   = width

        # ── AUXILIARY HEADS (Strategy A / legacy) ────────────────────────────
        self.mean_color_head      = None
        self.last_mean_color_pred = None
        if color_residual:
            self.mean_color_head = MeanColorHead(in_dim=_color_in)

        self.scene_semantic_module    = None
        self.last_scene_semantic_pred = None
        if scene_semantic_head:
            self.scene_semantic_module = SceneSemanticHead(in_dim=_sem_in)

        self.scene_layout_module    = None
        self.last_scene_layout_pred = None
        if scene_layout_head:
            self.scene_layout_module = SceneLayoutHead(in_dim=_lay_in)

        # ── STRATEGY B STRUCTURED HEADS ────────────────────────────────────────
        # Separate heads for z_layout tokens [B, 16, embed_dim]:
        #   z_layout[:, 0, :]          [B, embed_dim=32]   → color
        #   z_layout[:, 1:, :].flatten [B, 15*embed_dim=480] → semantic + layout
        #
        # These are ALWAYS created when any Strategy B flag is True AND the
        # corresponding head flag is enabled. They are independent of the
        # semantic_token_heads flag (which only applies to Strategy A).
        #
        # WHY SEPARATE HEADS:
        #   Strategy B heads receive [B,32] and [B,480] from z_layout tokens.
        #   Strategy A legacy heads receive [B,384] from shape_embed.
        #   Same linear cannot handle both — separate heads needed.
        _lay_color_in = embed_dim               # 32  — token 0
        # With structured split: semantic head → tokens 1-8 only, layout → tokens 9-15
        if structured_layout_tokens:
            _lay_sem_in = _N_SEM_TOKENS * embed_dim   # 8*32=256
            _lay_lay_in = _N_LAY_TOKENS * embed_dim   # 7*32=224
        else:
            _lay_sem_in = (self._n_zs_tokens - 1) * embed_dim  # 15*32=480
            _lay_lay_in = (self._n_zs_tokens - 1) * embed_dim  # 15*32=480 (same)

        self.lay_color_head     = None   # z_layout[:,0,:]      → mean_color [B,3]
        self.lay_semantic_head  = None   # z_layout[:,1:,:].flat → label_dist [B,72]
        self.lay_layout_head    = None   # z_layout[:,1:,:].flat → centroids  [B,72,3]

        _any_B_flag = decoder_layout_cross_attn or decoder_layout_additive
        if _any_B_flag:
            if color_residual:
                self.lay_color_head = MeanColorHead(in_dim=_lay_color_in)
                print(f"  [StratB] lay_color_head:    [{_lay_color_in}]→[3]  (z_layout token 0)")
            if scene_semantic_head:
                self.lay_semantic_head = SceneSemanticHead(in_dim=_lay_sem_in)
                print(f"  [StratB] lay_semantic_head: [{_lay_sem_in}]→[72]  (z_layout tokens 1-15)")
            if scene_layout_head:
                self.lay_layout_head = SceneLayoutHead(in_dim=_lay_lay_in)
                tok_range = 'tokens 9-15' if structured_layout_tokens else 'tokens 1-15'
                print(f"  [StratB] lay_layout_head:   [{_lay_lay_in}]→[72,3] ({tok_range})")

        # ── LATENT DISENTANGLEMENT ────────────────────────────────────────────
        self._mu_s_cache = None
        self._mu_g_cache = None
        if latent_disentangle:
            assert embed_dim > 0 and semantic_dims % embed_dim == 0
            geom_dims = 64 * 64 * 4 - semantic_dims
            assert geom_dims > 0
            self.mu_s_proj_mean     = nn.Linear(width, semantic_dims)
            self.mu_s_proj_var      = nn.Linear(width, semantic_dims)
            kl_in = (1 + num_latents - 1) * embed_dim
            self.kl_emb_proj_mean_g = nn.Linear(kl_in, geom_dims)
            self.kl_emb_proj_var_g  = nn.Linear(kl_in, geom_dims)
            print(f"  DISENTANGLE: mu_s[{semantic_dims}] | mu_g[{geom_dims}]")

        # ── Z_S INFONCE HEAD ─────────────────────────────────────────────────
        self.z_s_infonce_head      = None
        self.last_z_s_infonce_proj = None
        if latent_disentangle:
            self.z_s_infonce_head = SemanticTokenInfoNCEHead(
                in_dim=semantic_dims, proj_dim=128)

        # ── ANCHOR PREDICTION ────────────────────────────────────────────────
        self.anchor_pred_from_tokens            = None
        self.last_predicted_anchors_from_tokens = None
        if position_scaffold:
            n_tok = self._n_zg_tokens if decoder_zs_cross_attn else _Z_TOKENS
            self.anchor_pred_from_tokens = AnchorPredFromTokens(width=width, num_tokens=n_tok)

        # ── MAIN NEW IDEA: ZSCond DECODER ─────────────────────────────────────
        self.zs_cond_decoder = None
        self.post_kl_g       = None
        self.post_kl_s       = None
        self.GS_decoder_new  = None
        if decoder_zs_cross_attn:
            # Separate projections: z_g and z_s expand to transformer width independently
            self.post_kl_g = nn.Linear(embed_dim, width)
            self.post_kl_s = nn.Linear(embed_dim, width)
            nn.init.trunc_normal_(self.post_kl_g.weight, std=0.02)
            nn.init.trunc_normal_(self.post_kl_s.weight, std=0.02)
            self.zs_cond_decoder = ZSCondTransformerDecoder(
                width=width, heads=heads, layers=num_decoder_layers)
            self.GS_decoder_new = GS_decoder(
                3, 1024, num_tokens=self._n_zg_tokens, width=width,
                color_residual=color_residual)

        # ── STRATEGY B COMPONENTS ─────────────────────────────────────────────
        self.layout_projector      = None   # shape_embed → z_layout [B,16,embed_dim]
        self.post_kl_layout        = None   # z_layout embed_dim → width (for cross-attn)
        self.layout_additive_cond  = None   # z_layout → broadcast bias [B,width]
        self.zs_cond_decoder_B     = None   # ZSCond decoder for 512-token input
        self.GS_decoder_B          = None   # 512-token GS decoder
        self.z_layout_infonce_head = None   # flatten(z_layout) [B,512]→[B,128] L2-norm
        self.last_z_layout         = None   # cached [B,16,embed_dim] for losses + vis
        self.last_z_layout_proj    = None   # cached [B,128] for InfoNCE

        _any_B = decoder_layout_cross_attn or decoder_layout_additive
        if _any_B:
            self.layout_projector = Layout16Projector(
                in_dim=width, n_tokens=self._n_zs_tokens, token_dim=embed_dim)
            # InfoNCE head: flatten z_layout → [B, 16*32=512] → MLP → [B,128]
            _lay_flat = self._n_zs_tokens * embed_dim   # 16 * 32 = 512
            self.z_layout_infonce_head = SemanticTokenInfoNCEHead(
                in_dim=_lay_flat, proj_dim=128)
            print(f"  [Strategy B] Layout16Projector + InfoNCE head active")
            print(f"  z_layout InfoNCE: flatten [B,{_lay_flat}] → MLP → [B,128] L2-norm")

        if decoder_layout_cross_attn:
            # Cross-attn conditioning: z_layout projected to width as K/V
            self.post_kl_layout = nn.Linear(embed_dim, width)
            nn.init.trunc_normal_(self.post_kl_layout.weight, std=0.02)
            # ZSCond decoder: 512 geometry tokens in sequence, 16 layout K/V
            self.zs_cond_decoder_B = ZSCondTransformerDecoder(
                width=width, heads=heads, layers=num_decoder_layers)
            # 512-token GS decoder (same input size as original Can3Tok)
            self.GS_decoder_B = GS_decoder(
                3, 1024, num_tokens=_Z_TOKENS, width=width,
                color_residual=color_residual)
            print(f"  [Strategy B1] cross-attn: 512 geom + z_layout K/V per layer")

        if decoder_layout_additive:
            self.layout_additive_cond = LayoutAdditiveConditioner(
                n_tokens=self._n_zs_tokens, token_dim=embed_dim, width=width)
            if not decoder_layout_cross_attn:
                # B2 only — need standard decoder since no zs_cond_decoder_B
                # (legacy GS_decoder with 512 tokens is already in ShapeAsLatentPerceiver)
                pass
            print(f"  [Strategy B2] additive: flatten(z_layout)→MLP→[B,{width}] broadcast bias")

        if decoder_layout_cross_attn and decoder_layout_additive:
            print(f"  [Strategy B3] both additive bias + cross-attn active")

        # ── POSITIONAL ENCODING ───────────────────────────────────────────────
        self.decoder_pos_emb = None
        if decoder_pos_enc and not decoder_zs_cross_attn:
            n_tok = _Z_TOKENS
            self.decoder_pos_emb = nn.Parameter(torch.zeros(n_tok, width))
            nn.init.trunc_normal_(self.decoder_pos_emb, std=0.02)

        self.decoder_fourier_pe_flag   = decoder_fourier_pe
        self.decoder_fourier_pe_module = None
        if decoder_fourier_pe:
            n_tok = self._n_zg_tokens if decoder_zs_cross_attn else _Z_TOKENS
            self.decoder_fourier_pe_module = FourierDecoderPE(
                fourier_embedder=self.fourier_embedder, width=width, num_tokens=n_tok)

        # ── LEGACY TOKEN CONDITIONING (only when NOT using new design) ────────
        self.token_cond_adaln_flag = False
        self.adaLN_transformer     = None
        self.token_cond_mlp_B      = None
        self.token_cat_assign      = None
        fourier_out_dim = self.fourier_embedder.out_dim

        if decoder_zs_cross_attn:
            if token_cond or token_cond_adaln:
                print("  [INFO] decoder_zs_cross_attn=True: "
                      "TokenCond/AdaLN disabled (z_s conditions via cross-attn)")
        else:
            _adaln_valid = (token_cond and 'B' in token_cond_approach.upper())
            if token_cond_adaln and _adaln_valid:
                self.token_cond_adaln_flag = True
                self.adaLN_transformer = AdaLNTransformerDecoder(
                    width=width, heads=heads, layers=num_decoder_layers,
                    cond_dim=fourier_out_dim)
            if token_cond and 'B' in token_cond_approach.upper():
                self.token_cat_assign = nn.Parameter(torch.zeros(_Z_TOKENS, 72))
                nn.init.trunc_normal_(self.token_cat_assign, std=0.01)
                self.token_cond_mlp_B = TokenCondMLP(fourier_out_dim, width)

        # ── SEGMENT PREDICTION HEAD ───────────────────────────────────────────
        self.seg_pred_head = None
        self.last_seg_pred = None
        if predict_seg_labels:
            self.seg_pred_head = SegPredHead(in_dim=14, num_cats=72)

        # ── PER-GAUSSIAN INFONCE HEADS ────────────────────────────────────────
        self.semantic_projection_hidden    = None
        self.semantic_projection_geometric = None
        self.semantic_distribution_head    = None
        if semantic_mode == 'hidden':
            self.semantic_projection_hidden = SemanticProjectionHead(1024, 40000, 32)
        elif semantic_mode == 'geometric':
            self.semantic_projection_geometric = SemanticProjectionHeadGeometric(14, 40000, 32, 128)
        elif semantic_mode == 'dist':
            self.semantic_distribution_head = SemanticDistributionHead(1024, 72)
        elif semantic_mode not in ('none', 'attention'):
            raise ValueError(f"Unknown semantic_mode: '{semantic_mode}'")

        print(f"{'='*70}\n")

    # ── ENCODE ────────────────────────────────────────────────────────────────

    def encode_latents(self, pc, feats=None):
        x, _ = self.encoder(pc, feats)
        return x[:, 0], x[:, 1:]

    def encode_kl_embed(self, latents, sample_posterior=True):
        posterior = None
        if self.embed_dim > 0:
            moments   = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            kl_embed  = posterior.sample() if sample_posterior else posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior

    def encode(self, pc, feats=None, sample_posterior=True):
        shape_embed, latents    = self.encode_latents(pc, feats)
        self._shape_embed_cache = shape_embed
        kl_embed, posterior     = self.encode_kl_embed(latents, sample_posterior)
        kl_flat                 = kl_embed.reshape(kl_embed.shape[0], -1)

        if self.latent_disentangle:
            mu_s      = self.mu_s_proj_mean(shape_embed)
            log_var_s = self.mu_s_proj_var(shape_embed)
            mu_g      = self.kl_emb_proj_mean_g(kl_flat)
            log_var_g = self.kl_emb_proj_var_g(kl_flat)
            self._mu_s_cache = mu_s
            self._mu_g_cache = mu_g
            mu      = torch.cat([mu_s, mu_g],           dim=-1)
            log_var = torch.cat([log_var_s, log_var_g], dim=-1)
        else:
            self._mu_s_cache = None
            self._mu_g_cache = None
            mu      = self.kl_emb_proj_mean(kl_flat)
            log_var = self.kl_emb_proj_var(kl_flat)

        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return shape_embed, mu, log_var, z, posterior

    # ── DECODE ────────────────────────────────────────────────────────────────

    def decode(self, latents, volume_queries=None, return_semantic_features=False,
               shape_embed=None, scaffold_anchors=None, scaffold_token_ids=None,
               z_layout=None):
        """
        latents:  Z [B, 512, 32] always.
        z_layout: [B, 16, 32]  layout tokens (Strategy B1/B2/B3 only).
                  Pass self.last_z_layout when calling decode() from forward().

        STRATEGY A  (latent_disentangle=True, no cross-attn):
          All 512 tokens [z_layout|z_geo] in decoder sequence. Best performance.

        STRATEGY B1 (decoder_layout_cross_attn=True):
          512 geometry tokens in sequence + z_layout as cross-attn K/V per layer.

        STRATEGY B2 (decoder_layout_additive=True):
          512 geometry tokens in sequence + z_layout projected to additive broadcast bias.

        STRATEGY B3 (both B1 and B2):
          Additive bias (once, before stack) + cross-attn per layer.

        STRATEGY C (baseline, no flags):
          512 geometry tokens, standard transformer.

        STRATEGY D (decoder_zs_cross_attn=True, FAILED):
          496 geometry tokens + 16 z_s as cross-attn K/V.
        """
        B = latents.shape[0]

        # ── STRATEGY B1 / B2 / B3: 512 geometry + layout conditioning ─────────
        _any_B = (self.decoder_layout_cross_attn or self.decoder_layout_additive)
        if _any_B and z_layout is not None:
            H = self.post_kl(latents)   # [B, 512, 384] — standard post_kl, full geometry

            # Strategy B2 (or B3 additive part): broadcast bias before transformer
            if self.decoder_layout_additive and self.layout_additive_cond is not None:
                bias = self.layout_additive_cond(z_layout)   # [B, 384]
                H    = H + bias.unsqueeze(1)                  # [B, 512, 384]

            # Fourier PE / learnable PE on the 512-token geometry sequence
            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H = H + self.decoder_fourier_pe_module(B, H.device)
            elif self.decoder_pos_emb is not None:
                H = H + self.decoder_pos_emb.unsqueeze(0)

            # Strategy B1 (or B3 cross-attn part): per-layer cross-attn to z_layout
            if self.decoder_layout_cross_attn and self.zs_cond_decoder_B is not None:
                H_lay = self.post_kl_layout(z_layout)   # [B, 16, 384]
                H     = self.zs_cond_decoder_B(H, H_lay) # [B, 512, 384]
            else:
                # B2 only: standard transformer (no cross-attn)
                H = self.transformer(H)

            H_out = H

            self.last_predicted_anchors_from_tokens = None
            pred_anchors = None
            if self.anchor_pred_from_tokens is not None:
                pred_anchors = self.anchor_pred_from_tokens(H_out)
                self.last_predicted_anchors_from_tokens = pred_anchors

            has_sem     = any([self.semantic_projection_hidden,
                               self.semantic_projection_geometric,
                               self.semantic_distribution_head])
            need_hidden = return_semantic_features and has_sem
            latents_flat = H_out.reshape(B, -1)

            if self.decoder_layout_cross_attn and self.GS_decoder_B is not None:
                gs_dec = self.GS_decoder_B
            else:
                gs_dec = self.GS_decoder   # B2 only: use standard 512-token decoder

            if need_hidden:
                reconstruction, hidden = gs_dec(latents_flat, return_hidden=True)
            else:
                hidden        = None
                reconstruction = gs_dec(latents_flat)

            _fixed_ids = FIXED_TOKEN_IDS_512

        elif self.decoder_zs_cross_attn:
            # ── NEW DESIGN ────────────────────────────────────────────────────
            n_s = self._n_zs_tokens   # 16
            z_s_raw = latents[:, :n_s, :]   # [B, 16, 32]
            z_g_raw = latents[:, n_s:, :]   # [B, 496, 32]

            # Project to transformer width separately
            H_g = self.post_kl_g(z_g_raw)   # [B, 496, 384]
            H_s = self.post_kl_s(z_s_raw)   # [B, 16,  384]

            # Fourier PE on z_g
            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H_g = H_g + self.decoder_fourier_pe_module(B, H_g.device)

            # Run ZSCond transformer: self-attn(z_g) + cross-attn(z_g, z_s)
            H_out = self.zs_cond_decoder(H_g, H_s)   # [B, 496, 384]

            # Anchor prediction
            self.last_predicted_anchors_from_tokens = None
            pred_anchors = None
            if self.anchor_pred_from_tokens is not None:
                pred_anchors = self.anchor_pred_from_tokens(H_out)
                self.last_predicted_anchors_from_tokens = pred_anchors

            has_sem     = any([self.semantic_projection_hidden,
                               self.semantic_projection_geometric,
                               self.semantic_distribution_head])
            need_hidden = return_semantic_features and has_sem
            latents_flat = H_out.reshape(B, -1)
            if need_hidden:
                reconstruction, hidden = self.GS_decoder_new(latents_flat, return_hidden=True)
            else:
                hidden        = None
                reconstruction = self.GS_decoder_new(latents_flat)

            _fixed_ids = FIXED_TOKEN_IDS_496

        else:
            # ── LEGACY DESIGN ─────────────────────────────────────────────────
            H = self.post_kl(latents)

            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H = H + self.decoder_fourier_pe_module(B, H.device)
            elif self.decoder_pos_emb is not None:
                H = H + self.decoder_pos_emb.unsqueeze(0)

            cond_for_adaln = None
            if (self.token_cond_flag and 'B' in self.token_cond_approach and
                    self.token_cat_assign is not None and
                    self.last_scene_layout_pred is not None):
                W_cat     = F.softmax(self.token_cat_assign, dim=-1)
                tok_cents = torch.einsum('tk,bkd->btd', W_cat, self.last_scene_layout_pred)
                fourier_B = self.fourier_embedder(tok_cents)
                if self.token_cond_adaln_flag:
                    cond_for_adaln = fourier_B
                elif self.token_cond_mlp_B is not None:
                    H = H + self.token_cond_mlp_B(fourier_B)

            if (self.token_cond_adaln_flag and self.adaLN_transformer is not None
                    and cond_for_adaln is not None):
                H_out = self.adaLN_transformer(H, cond_for_adaln)
            else:
                H_out = self.transformer(H)

            self.last_predicted_anchors_from_tokens = None
            pred_anchors = None
            if self.anchor_pred_from_tokens is not None:
                pred_anchors = self.anchor_pred_from_tokens(H_out)
                self.last_predicted_anchors_from_tokens = pred_anchors

            has_sem     = any([self.semantic_projection_hidden,
                               self.semantic_projection_geometric,
                               self.semantic_distribution_head])
            need_hidden = return_semantic_features and has_sem
            latents_flat = H_out.reshape(B, -1)
            if need_hidden:
                reconstruction, hidden = self.GS_decoder(latents_flat, return_hidden=True)
            else:
                hidden = None
                reconstruction = self.GS_decoder(latents_flat)

            _fixed_ids = FIXED_TOKEN_IDS_512

        # ── ADD DC TERM (position scaffold) ──────────────────────────────────
        if pred_anchors is not None:
            pred_3d = reconstruction.reshape(B, 40_000, 14)
            if scaffold_token_ids is not None:
                idx_3d = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, 3)
                dc     = torch.gather(pred_anchors, 1, idx_3d)
            else:
                dc = pred_anchors[:, _fixed_ids.to(pred_anchors.device), :]
            pred_3d[:, :, 0:3] += dc
            reconstruction = pred_3d.reshape(B, -1)

        # ── SEGMENT PREDICTION ────────────────────────────────────────────────
        self.last_seg_pred = None
        if self.seg_pred_head is not None:
            self.last_seg_pred = self.seg_pred_head(reconstruction.reshape(B, 40000, 14))

        # ── PER-GAUSSIAN SEMANTIC FEATURES ───────────────────────────────────
        semantic_features = None
        if return_semantic_features and hidden is not None:
            if self.semantic_mode == 'hidden':
                semantic_features = self.semantic_projection_hidden(hidden)
            elif self.semantic_mode == 'geometric':
                semantic_features = self.semantic_projection_geometric(
                    reconstruction.reshape(B, 40000, 14))
            elif self.semantic_mode == 'dist':
                semantic_features = self.semantic_distribution_head(hidden)

        return reconstruction, semantic_features

    # ── FORWARD ───────────────────────────────────────────────────────────────

    def forward(self, pc, feats, volume_queries, sample_posterior=True,
                scaffold_anchors=None, scaffold_token_ids=None,
                return_semantic_features=None):
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)
        _se = self._shape_embed_cache

        # ── STRATEGY A/D: z_s InfoNCE projection (when latent_disentangle) ────
        self.last_z_s_infonce_proj = None
        if self.z_s_infonce_head is not None:
            self.last_z_s_infonce_proj = self.z_s_infonce_head(
                z[:, :self.semantic_dims])

        # ── STRATEGY B: compute z_layout from shape_embed ────────────────────
        self.last_z_layout      = None
        self.last_z_layout_proj = None
        _any_B = self.decoder_layout_cross_attn or self.decoder_layout_additive
        if _any_B and self.layout_projector is not None:
            self.last_z_layout = self.layout_projector(_se)   # [B, 16, 32]
            # InfoNCE projection: flatten z_layout → [B, 512] → MLP → [B, 128]
            if self.z_layout_infonce_head is not None:
                z_lay_flat = self.last_z_layout.reshape(z.shape[0], -1)  # [B, 512]
                self.last_z_layout_proj = self.z_layout_infonce_head(z_lay_flat)

        # ── AUXILIARY HEADS ───────────────────────────────────────────────────
        # Route to z tokens (semantic_token_heads) or shape_embed (legacy)
        # For Strategy B: run heads on z_layout tokens when available
        _lay_src = self.last_z_layout   # [B, 16, 32] or None

        if self.semantic_token_heads_flag or self.structured_layout_tokens_flag:
            _ed = self.embed_dim
            _sd = self.semantic_dims
            self.last_mean_color_pred = (
                self.mean_color_head(z[:, :_ed]) if self.mean_color_head else None)

            if self.structured_layout_tokens_flag:
                # STRUCTURED: separate token ranges, no gradient interference
                # z[:, :_ed] = token 0 (color, already used above)
                # z[:, _ed : _ed + _n_sem*_ed] = tokens 1-8 → semantic
                # z[:, _ed + _n_sem*_ed : _sd] = tokens 9-15 → layout
                _n_s = self._n_sem_tokens  # 8
                z_sem = z[:, _ed : _ed + _n_s * _ed]          # [B, 256]
                z_lay = z[:, _ed + _n_s * _ed : _sd]          # [B, 224]
                self.last_scene_semantic_pred = (
                    self.scene_semantic_module(z_sem) if self.scene_semantic_module else None)
                self.last_scene_layout_pred = (
                    self.scene_layout_module(z_lay) if self.scene_layout_module else None)
            else:
                # UNSTRUCTURED: both heads on all tokens 1-15 (old behaviour)
                z_sem = z[:, _ed:_sd]   # [B, 480]
                self.last_scene_semantic_pred = (
                    self.scene_semantic_module(z_sem) if self.scene_semantic_module else None)
                self.last_scene_layout_pred = (
                    self.scene_layout_module(z_sem) if self.scene_layout_module else None)
        elif _lay_src is not None and _any_B:
            # Strategy B: dedicated heads on z_layout tokens (correct input dims).
            # Token 0 → lay_color_head [B, 32]
            # Without structured split: tokens 1-15 → both heads [B, 480] (interference)
            # With structured split:    tokens 1-8  → lay_semantic_head [B, 256]
            #                           tokens 9-15 → lay_layout_head   [B, 224]
            B_cur = z.shape[0]
            self.last_mean_color_pred = (
                self.lay_color_head(_lay_src[:, 0, :]) if self.lay_color_head else None)

            if self.structured_layout_tokens_flag:
                _n_s   = self._n_sem_tokens   # 8
                z_sem  = _lay_src[:, 1 : 1+_n_s, :].reshape(B_cur, -1)  # [B, 256]
                z_lay  = _lay_src[:, 1+_n_s : , :].reshape(B_cur, -1)   # [B, 224]
                self.last_scene_semantic_pred = (
                    self.lay_semantic_head(z_sem) if self.lay_semantic_head else None)
                self.last_scene_layout_pred = (
                    self.lay_layout_head(z_lay) if self.lay_layout_head else None)
            else:
                _lay_all = _lay_src[:, 1:, :].reshape(B_cur, -1)  # [B, 480]
                self.last_scene_semantic_pred = (
                    self.lay_semantic_head(_lay_all) if self.lay_semantic_head else None)
                self.last_scene_layout_pred = (
                    self.lay_layout_head(_lay_all) if self.lay_layout_head else None)
        else:
            # Legacy: heads on shape_embed [B, 384]
            self.last_scene_layout_pred = (
                self.scene_layout_module(_se) if self.scene_layout_module else None)

        latents = z.reshape(z.shape[0], 512, 32)
        _rsf = self.training if return_semantic_features is None else return_semantic_features

        UV_gs_recover, per_gaussian_features = self.decode(
            latents, volume_queries,
            return_semantic_features=_rsf,
            shape_embed=_se,
            scaffold_anchors=scaffold_anchors,
            scaffold_token_ids=scaffold_token_ids,
            z_layout=self.last_z_layout)   # None if not Strategy B

        if not self.semantic_token_heads_flag and not _any_B:
            self.last_mean_color_pred = (
                self.mean_color_head(_se) if self.mean_color_head else None)
            self.last_scene_semantic_pred = (
                self.scene_semantic_module(_se) if self.scene_semantic_module else None)

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features