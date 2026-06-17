# -*- coding: utf-8 -*-
"""
sal_perceiver_dist_changes.py  —  Can3Tok VAE
==============================================
DECODER STRATEGIES (controlled by flags, all backward-compatible):

  A  latent_disentangle=True, decoder_zs_cross_attn=False  [BEST PERFORMANCE]
  B1 decoder_layout_cross_attn=True   [NEW — 512 geom + cross-attn conditioning]
  B2 decoder_layout_additive=True     [NEW — 512 geom + additive bias conditioning]
  B3 Both B1+B2 simultaneously
  C  baseline (no flags)
  D  decoder_zs_cross_attn=True [FAILED — kept for reference]

CHANGE vs original: get_decoder_transformer_features() added between encode() and decode()
for use by Stage 2 Latent Perceptual Loss (LPL).
See train_stage2.py --lpl_weight flag.

TOKEN-LOCAL DECODER ADDITION (token_local_decoder=True flag):
  Replaces the flat 777M-param GS_decoder (1024-d bottleneck → memorisation
  ceiling) with a shared per-token MLP (~1.6M params, no bottleneck). Each
  of the 512 decoder tokens decodes its own slice of 79 Gaussians using
  shared weights. Per-Gaussian semantic features for InfoNCE flow through
  the existing SemanticProjectionHead via a pooled-hidden path that matches
  the original [B, 1024] interface, so semantic_mode='hidden' works unchanged.
  All other strategies (A/B/C/D) remain backward-compatible.
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
from .token_local_decoder import TokenLocalDecoder


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


class SemanticTokenInfoNCEHead(nn.Module):
    def __init__(self, in_dim=512, proj_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, proj_dim))
        print(f"[SemanticTokenInfoNCEHead] [{in_dim}]→[{proj_dim}] L2-norm | "
              f"{sum(p.numel() for p in self.parameters()):,} params")
    def forward(self, z_s_flat):
        return F.normalize(self.head(z_s_flat), p=2, dim=-1)


class ZSTokenPoolProjectHead(nn.Module):
    def __init__(self, n_tokens=16, token_dim=32, hidden_dim=1024, feature_dim=32):
        super().__init__()
        self.n_tokens    = n_tokens
        self.feature_dim = feature_dim
        self.to_hidden   = nn.Linear(token_dim, hidden_dim)
        self.projection  = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, n_tokens * feature_dim))
        total = sum(p.numel() for p in self.parameters())
        print(f"[ZSTokenPoolProjectHead] [{n_tokens},{token_dim}]->"
              f"pool->[{token_dim}]->[{hidden_dim}]->MLP->[{n_tokens},{feature_dim}] L2-norm | "
              f"{total/1e3:.1f}K params")

    def forward(self, tokens):
        pooled    = tokens.mean(dim=1)
        hidden    = self.to_hidden(pooled)
        proj      = self.projection(hidden)
        B         = tokens.shape[0]
        embeddings = F.normalize(
            proj.reshape(B, self.n_tokens, self.feature_dim), p=2, dim=-1)
        return embeddings, hidden


# ============================================================================
# STRATEGY B: LAYOUT CONDITIONING COMPONENTS
# ============================================================================

class Layout16Projector(nn.Module):
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
    def __init__(self, n_tokens=16, token_dim=32, width=384):
        super().__init__()
        in_dim = n_tokens * token_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width))
        print(f"[LayoutAdditiveConditioner] [{in_dim}]→[{width}] broadcast bias | "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, z_layout):
        B = z_layout.shape[0]
        return self.proj(z_layout.reshape(B, -1))


# ============================================================================
# FOURIER DECODER PE
# ============================================================================

class FourierDecoderPE(nn.Module):
    SCAFFOLD_DIMS = 8

    def __init__(self, fourier_embedder, width, num_tokens=512):
        super().__init__()
        S = self.SCAFFOLD_DIMS
        coords = []
        for i in range(S):
            for j in range(S):
                for k in range(S):
                    coords.append([(2*i/(S-1))-1, (2*j/(S-1))-1, (2*k/(S-1))-1])
        all_coords = torch.tensor(coords, dtype=torch.float32)
        if num_tokens != 512:
            all_coords = all_coords[512 - num_tokens:]
        self.register_buffer('voxel_coords', all_coords)
        assert all_coords.shape[0] == num_tokens
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
# ZSCond DECODER TRANSFORMER
# ============================================================================

class ZSCondTransformerBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm_sa    = nn.LayerNorm(width)
        self.norm_ca    = nn.LayerNorm(width)
        self.norm_ff    = nn.LayerNorm(width)
        self.self_attn  = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn        = nn.Sequential(
            nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x, z_s_kv):
        h, _ = self.self_attn(self.norm_sa(x), self.norm_sa(x), self.norm_sa(x))
        x = x + h
        h, _ = self.cross_attn(self.norm_ca(x), z_s_kv, z_s_kv)
        x = x + h
        x = x + self.ffn(self.norm_ff(x))
        return x


class ZSCondTransformerDecoder(nn.Module):
    def __init__(self, width, heads, layers):
        super().__init__()
        self.blocks   = nn.ModuleList(
            [ZSCondTransformerBlock(width, heads) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(width)
        total = sum(p.numel() for p in self.parameters())
        print(f"[ZSCondTransformerDecoder] {layers}× ZSCondTransformerBlock "
              f"(width={width}, heads={heads}) | {total/1e6:.2f}M params")

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


# ============================================================================
# GAUSSIANCUBE-STYLE FRAMED MICRO-PATTERN  (scene-level structured residual)
# ============================================================================
# GaussianCube (Zhang et al., NeurIPS 2024) gets crisp geometry by snapping
# Gaussians onto a DENSE regular voxel grid via optimal transport, so position is
# essentially a grid index (structurally free) plus a small offset. A dense grid
# is the wrong primitive for SCENES (surfaces are ~2D in a 3D volume, so a dense
# grid is mostly empty and the bijection drags Gaussians far off-surface). The
# portable part of the idea is narrow and real: position = structural reference +
# SMALL residual, so the latent never spends bits on coarse placement.
#
# Here the structural reference is, per latent token, a FIXED canonical point set
# (a low-discrepancy ball, free / shared / no bits) carried by a learned per-token
# FRAME — an anisotropic scale + rotation predicted from the latent — so the
# canonical ball can be flattened and oriented to the local surface patch. The
# decoder then only has to predict a small per-Gaussian residual. This is the
# sparse, scene-adaptive analogue of GaussianCube's grid, and it sits on top of
# the existing anchor-relative decode (anchor = block centre, already ~free).
#
# Alignment (the reason this is exact, not approximate): in scaffold_mode=
# 'hilbert_block' the dataset sets token_ids = arange(N) // g (a FIXED array), which
# is precisely the TokenLocalDecoder's own layout (Gaussian i -> token i//g, slot
# i%g). So the anchor a Gaussian receives, the decoder slot that produced it, and
# the canonical-pattern slot c[i%g] all index the same token consistently. The
# micro-pattern therefore REQUIRES hilbert_block (guarded in the training script).

def _fibonacci_ball(n):
    """`n` deterministic low-discrepancy points in the unit ball — the canonical
    per-token micro-pattern (the scene-level stand-in for GaussianCube's voxel
    grid). Fibonacci-sphere directions on radius cbrt(i/n) shells fill the ball
    ~uniformly. Returned as float32 [n, 3]."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    golden = math.pi * (3.0 - math.sqrt(5.0))           # golden angle
    z = 1.0 - 2.0 * i / n                               # in (-1, 1)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = golden * i
    dirs = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)
    radii = (i / n) ** (1.0 / 3.0)                      # ~uniform ball density
    return (dirs * radii.unsqueeze(-1)).to(torch.float32)   # [n, 3]


def _rot6d_to_matrix(r6):
    """Zhou et al. (CVPR 2019) continuous 6D rotation rep -> [..., 3, 3] rotation
    matrix via Gram-Schmidt. r6[..., 0:3] and r6[..., 3:6] are the first two
    (un-orthonormalised) columns; identity rep is [1,0,0, 0,1,0]."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)            # columns (b1,b2,b3)


class BlockFramePredFromTokens(nn.Module):
    """Per-token local FRAME for the canonical micro-pattern: an anisotropic scale
    (3) and, optionally, a 6D rotation (6) per latent token, predicted from the
    transformer tokens. Lets each token's fixed canonical ball be flattened (one
    small scale) and oriented to the local surface, so the decoder predicts only a
    small residual. Deterministic function of the latent (Stage-2 compatible),
    exactly like AnchorPredFromTokens. Inits to scale=scale_init, rotation=identity
    so the first forward is a benign ~isotropic ball of the right size."""
    def __init__(self, width=384, num_tokens=512, use_rotation=True, scale_init=0.5):
        super().__init__()
        self.num_tokens   = num_tokens
        self.use_rotation = use_rotation
        self.trunk = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),   nn.LayerNorm(64),  nn.ReLU())
        self.log_scale_head = nn.Linear(64, 3)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.constant_(self.log_scale_head.bias, float(math.log(scale_init)))
        if use_rotation:
            self.rot6_head = nn.Linear(64, 6)
            nn.init.zeros_(self.rot6_head.weight)
            with torch.no_grad():
                self.rot6_head.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        else:
            self.rot6_head = None
        print(f"[BlockFramePredFromTokens] [B,{num_tokens},{width}] -> scale[B,{num_tokens},3]"
              f"{' + rot6' if use_rotation else ''} | scale_init={scale_init} | "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, transformer_tokens):
        B, T, W = transformer_tokens.shape
        h = self.trunk(transformer_tokens.reshape(B * T, W))
        s = self.log_scale_head(h).exp().reshape(B, T, 3)          # [B,T,3] > 0
        R = (_rot6d_to_matrix(self.rot6_head(h)).reshape(B, T, 3, 3)
             if self.rot6_head is not None else None)
        return s, R


_N_GAUSSIANS  = 10_000
FIXED_TOKEN_IDS_512 = torch.arange(_N_GAUSSIANS) * 512 // _N_GAUSSIANS
FIXED_TOKEN_IDS_496 = torch.arange(_N_GAUSSIANS) * 496 // _N_GAUSSIANS

# Number of LATENT tokens for the structured / local-encoder path. This is the
# Hilbert-block count (the dataset's compute_hilbert_block_scaffold, the decoder's
# num_tokens, FIXED_TOKEN_IDS_512 are all 512), so token k <-> block k <-> anchor k
# end to end. It used to be derived as 16384 // embed_dim, which only equalled 512
# at embed_dim=32 and HALVED the tokens whenever embed_dim grew. Pinning it to the
# block count instead makes embed_dim a pure CAPACITY knob: the structured latent
# total = _N_LATENT_TOKENS * embed_dim (16384 at embed_dim=32, 32768 at 64, ...)
# while the block structure, scaffold, decoder grouping and micro-pattern are all
# unchanged.
_N_LATENT_TOKENS = 512


def set_num_gaussians(n):
    """Override the per-scene Gaussian count (default 10_000) and recompute the
    fixed token-id buffers. Call ONCE from the training script BEFORE building the
    model, so every decoder/encoder that reads _N_GAUSSIANS uses the new value. The
    latent token count (_Z_TOKENS = 16384 // embed_dim) is independent and does NOT
    change; only Gaussians-per-token g = ceil(_N_GAUSSIANS / num_tokens) scales."""
    global _N_GAUSSIANS, FIXED_TOKEN_IDS_512, FIXED_TOKEN_IDS_496
    _N_GAUSSIANS = int(n)
    FIXED_TOKEN_IDS_512 = torch.arange(_N_GAUSSIANS) * 512 // _N_GAUSSIANS
    FIXED_TOKEN_IDS_496 = torch.arange(_N_GAUSSIANS) * 496 // _N_GAUSSIANS
    print(f"[set_num_gaussians] _N_GAUSSIANS = {_N_GAUSSIANS}")


# Position-conditioned colour/rotation refinement heads (see position_conditioned_heads.py).
# Off by default; toggled from the training script before the model is built, so the
# YAML config does not need to change. enabled=False => exact baseline behaviour.
_POS_COND = {'enabled': False, 'n_freqs': 32, 'sigma': 6.0, 'pos_scale': 10.0,
             'hidden': 128, 'color': True, 'rotation': True}

def set_pos_cond_heads(enabled=True, n_freqs=32, sigma=6.0, pos_scale=10.0,
                       hidden=128, color=True, rotation=True):
    """Configure the position-conditioned refinement heads. Call ONCE before building
    the model. pos_scale should be ~ the scene radius so the Fourier frequencies are
    scene-appropriate."""
    _POS_COND.update(enabled=bool(enabled), n_freqs=int(n_freqs), sigma=float(sigma),
                     pos_scale=float(pos_scale), hidden=int(hidden),
                     color=bool(color), rotation=bool(rotation))
    print(f"[set_pos_cond_heads] {_POS_COND}")


# ============================================================================
# PER-GAUSSIAN INFONCE HEADS
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
    def __init__(self, hidden_dim=1024, num_gaussians=10000, feature_dim=32):
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
    def __init__(self, gaussian_dim=14, num_gaussians=10000, feature_dim=32, hidden_dim=128):
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
# GS DECODER MLP
# ============================================================================

class GS_decoder(nn.Module):
    """
    Flat MLP: flatten(transformer_output) → 10000×14 Gaussian attributes.
    Instantiated as GS_decoder(D=3, W=1024, num_tokens=512, width=384).
    Parameter count: ~777M  (Linear(196608→1024)=201M + Linear(1024→560000)=574M)
    """
    def __init__(self, D=8, W=256, num_tokens=512, width=384, color_residual=False):
        super().__init__()
        input_ch            = num_tokens * width
        self.color_residual = color_residual
        self.pts_linears    = nn.ModuleList([nn.Linear(input_ch, W)])
        for _ in range(D - 1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.LayerNorm(W))
            self.pts_linears.append(nn.ReLU())
        self.output_linear = nn.Linear(W, _N_GAUSSIANS * 14)
        print(f"  GS_DECODER ({num_tokens} tokens): {num_tokens}×{width}={input_ch} "
              f"→ 10000×14  "
              f"({'residuals' if color_residual else 'clamp(0,1)'})")

    def forward(self, x, return_hidden=False):
        for layer in self.pts_linears: x = layer(x)
        hidden = x
        raw    = self.output_linear(x).reshape(x.shape[0], _N_GAUSSIANS, 14)
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


class LocalCrossAttentionEncoder(nn.Module):
    """
    Spatially-LOCAL encoder for the structured per-token latent.

    The input points are space-filling-ordered (Hilbert), so a contiguous block of
    g = ceil(num_gaussians / K) points is a local spatial cluster -- and block k here
    is the SAME block k that the decoder's anchor + TokenLocalDecoder reconstruct.
    Each of the K geometry queries attends only to a local WINDOW of blocks around its
    own block (k-window .. k+window) instead of attending to the whole scene; one
    extra global CLS query (index 0) attends to all points (semantic summary). So
    latent token k encodes the LOCAL geometry of block k. Local patches recur across
    scenes, so the features are compositional and generalize, unlike a global
    per-scene code (which memorizes -- the failure mode observed at 3800 scenes).

    No global self-attention runs inside the encoder (that would re-mix tokens and
    destroy locality); cross-token mixing happens later in the decoder transformer.
    Returns latents [B, 1+K, width] to match CrossAttentionEncoder's interface.
    """
    def __init__(self, *, device, dtype, num_latents, fourier_embedder,
                 fourier_embedder_ID, point_feats, width, heads,
                 window=1, num_gaussians=10000, qkv_bias=True,
                 use_ln_post=False, use_checkpoint=False):
        super().__init__()
        assert width % heads == 0, "width must be divisible by heads"
        self.width            = width
        self.heads            = heads
        self.head_dim         = width // heads
        self.window           = int(window)
        self.num_latents      = num_latents          # = 1 + K
        self.K                = num_latents - 1
        self.num_gaussians    = int(num_gaussians)
        self.g                = math.ceil(self.num_gaussians / self.K)
        self.use_checkpoint   = use_checkpoint
        self.fourier_embedder    = fourier_embedder
        self.fourier_embedder_ID = fourier_embedder_ID

        self.query = nn.Parameter(
            torch.randn(num_latents, width, device=device, dtype=dtype) * 0.02)
        self.input_proj = nn.Linear(
            fourier_embedder.out_dim + point_feats + fourier_embedder_ID.out_dim,
            width, device=device, dtype=dtype)
        self.norm_q   = nn.LayerNorm(width, device=device, dtype=dtype)
        self.norm_kv  = nn.LayerNorm(width, device=device, dtype=dtype)
        self.q_proj   = nn.Linear(width, width, bias=qkv_bias, device=device, dtype=dtype)
        self.k_proj   = nn.Linear(width, width, bias=qkv_bias, device=device, dtype=dtype)
        self.v_proj   = nn.Linear(width, width, bias=qkv_bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.norm_ff  = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width, device=device, dtype=dtype), nn.GELU(),
            nn.Linear(4 * width, width, device=device, dtype=dtype))
        self.ln_post = (nn.LayerNorm(width, device=device, dtype=dtype)
                        if use_ln_post else None)
        _mode = "GLOBAL geometry attn" if self.window < 0 else f"local window=±{self.window}"
        print(f"[LocalCrossAttentionEncoder] K={self.K} g={self.g} {_mode} "
              f"heads={heads} width={width} | per-token latent, no global self-attn")

    def _window_gather(self, xb, w):
        # [B, K, g, H, hd] -> [B, K, (2w+1)*g, H, hd]  (sliding window over the block dim)
        B, K, g, H, hd = xb.shape
        F_ = g * H * hd
        x = xb.reshape(B, K, F_).transpose(1, 2)              # [B, F_, K]
        x = F.pad(x, (w, w))                                  # [B, F_, K+2w]
        x = x.unfold(2, 2 * w + 1, 1)                         # [B, F_, K, 2w+1]
        x = x.permute(0, 2, 3, 1).contiguous()                # [B, K, 2w+1, F_]
        return x.reshape(B, K, (2 * w + 1) * g, H, hd)        # [B, K, (2w+1)*g, H, hd]

    def _window_valid(self, vb, w):
        # [B, K, g] bool -> [B, K, (2w+1)*g] bool  (same ordering as _window_gather)
        B, K, g = vb.shape
        x = vb.float().transpose(1, 2)                        # [B, g, K]
        x = F.pad(x, (w, w))                                  # [B, g, K+2w]
        x = x.unfold(2, 2 * w + 1, 1)                         # [B, g, K, 2w+1]
        x = x.permute(0, 2, 3, 1).contiguous()                # [B, K, 2w+1, g]
        return (x.reshape(B, K, (2 * w + 1) * g) > 0.5)       # [B, K, (2w+1)*g]

    def _forward(self, pc, feats):
        B, N, _ = pc.shape
        K, g, w, H, hd, Wd = self.K, self.g, self.window, self.heads, self.head_dim, self.width

        data = torch.cat([
            self.fourier_embedder(pc[:, :, 4:7]),
            self.fourier_embedder_ID(pc[:, :, 0:3]),
            feats[:, :, 7:],
        ], dim=-1).to(dtype=torch.float32)
        data = self.input_proj(data)                          # [B, N, Wd]
        data = self.norm_kv(data)

        Kg = K * g
        if N < Kg:
            pad   = data.new_zeros(B, Kg - N, Wd)
            dat_p = torch.cat([data, pad], dim=1)             # [B, Kg, Wd]
            valid = torch.cat([
                torch.ones(B, N, device=data.device, dtype=torch.bool),
                torch.zeros(B, Kg - N, device=data.device, dtype=torch.bool)], dim=1)
        else:
            dat_p = data[:, :Kg]
            valid = torch.ones(B, Kg, device=data.device, dtype=torch.bool)

        kp = self.k_proj(dat_p)                               # [B, Kg, Wd]
        vp = self.v_proj(dat_p)

        q_raw = self.query.unsqueeze(0).expand(B, -1, -1)     # [B, 1+K, Wd]
        q     = self.q_proj(self.norm_q(q_raw))               # [B, 1+K, Wd]

        # ── CLS query (index 0): GLOBAL attention over all N real points ───────
        q_cls = q[:, 0:1].reshape(B, 1, H, hd).transpose(1, 2)            # [B, H, 1, hd]
        k_cls = kp[:, :N].reshape(B, N, H, hd).transpose(1, 2)           # [B, H, N, hd]
        v_cls = vp[:, :N].reshape(B, N, H, hd).transpose(1, 2)           # [B, H, N, hd]
        o_cls = F.scaled_dot_product_attention(q_cls, k_cls, v_cls)       # [B, H, 1, hd]
        o_cls = o_cls.transpose(1, 2).reshape(B, 1, Wd)                   # [B, 1, Wd]

        # ── geometry queries (1..K) ────────────────────────────────────────────
        if w < 0:
            # GLOBAL ablation (Run B): every geometry query attends to ALL points,
            # exactly like the CLS query. Everything else is identical to the
            # windowed encoder -- same 512 per-token queries, same per-token latent
            # (no global remix), still NO encoder global self-attention -- so this
            # isolates attention SCOPE (global vs local window) as the only variable.
            # No padding mask (all N input points are real), so SDPA uses the
            # flash/mem-efficient path and never materializes the [K, N] scores.
            qg    = q[:, 1:].reshape(B, K, H, hd).transpose(1, 2)   # [B, H, K, hd]
            ka    = kp[:, :N].reshape(B, N, H, hd).transpose(1, 2)  # [B, H, N, hd]
            va    = vp[:, :N].reshape(B, N, H, hd).transpose(1, 2)  # [B, H, N, hd]
            o_geo = F.scaled_dot_product_attention(qg, ka, va)      # [B, H, K, hd]
            o_geo = o_geo.transpose(1, 2).reshape(B, K, Wd)         # [B, K, Wd]
        else:
            # ── LOCAL windowed attention (window>=0) ───────────────────────────
            kb   = kp.reshape(B, K, g, H, hd)
            vb   = vp.reshape(B, K, g, H, hd)
            vblk = valid.reshape(B, K, g)
            if w > 0:
                kb    = self._window_gather(kb, w)                # [B, K, S, H, hd]
                vb    = self._window_gather(vb, w)
                vmask = self._window_valid(vblk, w)               # [B, K, S]
            else:
                vmask = vblk                                      # [B, K, g]
            S = kb.shape[2]
            # empty tail blocks (whose whole window is padding) would give all-masked
            # rows -> NaN softmax; force each such row to attend to slot 0 (the decoder
            # never reads these blocks). Use a shape-safe slice assignment: vmask[...,0]
            # and all_masked are both [B, K], so this works regardless of vmask rank and
            # avoids the version-dependent vmask[empty, 0] advanced-index semantics that
            # raised IndexError on the cluster's torch.
            all_masked = ~vmask.any(dim=-1)                       # [B, K]
            if bool(all_masked.any()):
                vmask = vmask.clone()
                vmask[..., 0] = vmask[..., 0] | all_masked        # [B,K] | [B,K]
            q_geo = q[:, 1:].reshape(B, K, H, hd).unsqueeze(3)    # [B, K, H, 1, hd]
            k_geo = kb.permute(0, 1, 3, 2, 4)                     # [B, K, H, S, hd]
            v_geo = vb.permute(0, 1, 3, 2, 4)                     # [B, K, H, S, hd]
            m_geo = vmask.reshape(B, K, 1, 1, S)                  # [B, K, 1, 1, S]
            o_geo = F.scaled_dot_product_attention(q_geo, k_geo, v_geo, attn_mask=m_geo)
            o_geo = o_geo.squeeze(3).reshape(B, K, Wd)            # [B, K, Wd]

        attn   = torch.cat([o_cls, o_geo], dim=1)             # [B, 1+K, Wd]
        tokens = q_raw + self.out_proj(attn)                  # residual cross-attn
        tokens = tokens + self.ffn(self.norm_ff(tokens))      # residual FFN
        if self.ln_post is not None:
            tokens = self.ln_post(tokens)
        return tokens, pc

    def forward(self, pc, feats=None):
        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


class CrossAttentionDecoder(nn.Module):
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
                 decoder_zs_cross_attn=False,
                 decoder_layout_cross_attn=False,
                 decoder_layout_additive=False,
                 structured_layout_tokens=False,
                 position_layout_residual=False,
                 jepa_idea1=False,
                 query_decoder=False,
                 token_local_decoder=False,
                 anchor_relative_decode=False,
                 anchor_teacher_force=False,
                 offset_scale_init=2.0,
                 structured_latent=False,
                 local_encoder=False,
                 local_window=1,
                 micro_pattern=False,
                 micro_pattern_rotation=True,
                 micro_offset_scale=0.3):

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
        self.token_local_decoder_flag      = token_local_decoder

        # ── STRUCTURED PER-TOKEN LATENT / LOCAL ENCODER ─────────────────────────
        # local_encoder implies structured_latent (a local encoder is pointless if the
        # per-token structure is then globally re-mixed). structured_latent uses the
        # per-token posterior directly as z and SKIPS the dense kl_emb_proj_mean(_g)
        # remix, so latent token k corresponds to encoder token k (= Hilbert block k).
        # Both rely on dropping the global remix + the 16/496 disentangle split, so
        # latent_disentangle is forced off here (the training script likewise zeroes
        # cross_recon / ortho / z_s-InfoNCE when disentangle is off).
        if local_encoder:
            structured_latent = True
        self.structured_latent = structured_latent
        self.local_encoder     = local_encoder
        self.local_window      = int(local_window)
        if structured_latent and latent_disentangle:
            print("  [INFO] structured_latent/local_encoder: forcing latent_disentangle=False")
            latent_disentangle      = False
            self.latent_disentangle = False

        if local_encoder:
            # The structured latent has _Z_TOKENS = 16384 // embed_dim tokens (the
            # decoder / scaffold token count), which can DIFFER from the encoder
            # bottleneck self.num_latents-1 (the original model maps between them with
            # the dense kl_emb_proj_mean). The local encoder must emit one local token
            # per LATENT token, so it gets _Z_TOKENS geometry queries (+1 CLS) and
            # g = N / _Z_TOKENS Gaussians per Hilbert block, aligning encoder block k
            # == decoder block k == scaffold anchor k end to end.
            _n_latent_tokens = _N_LATENT_TOKENS
            self.encoder = LocalCrossAttentionEncoder(
                device=device, dtype=dtype,
                num_latents=_n_latent_tokens + 1,      # _Z_TOKENS geometry queries + 1 CLS
                fourier_embedder=self.fourier_embedder,
                fourier_embedder_ID=self.fourier_embedder_ID,
                point_feats=point_feats, width=width, heads=heads,
                window=self.local_window, num_gaussians=_N_GAUSSIANS,
                qkv_bias=qkv_bias, use_ln_post=use_ln_post,
                use_checkpoint=use_checkpoint)

        _Z_TOKENS         = _N_LATENT_TOKENS
        self._n_zs_tokens = semantic_dims // embed_dim
        self._n_zg_tokens = _Z_TOKENS - self._n_zs_tokens

        print(f"\n{'='*70}")
        print(f"  CAN3TOK")
        print(f"  decoder_zs_cross_attn    = {decoder_zs_cross_attn}  (Strategy D)")
        print(f"  decoder_layout_cross_attn= {decoder_layout_cross_attn}  (Strategy B1)")
        print(f"  decoder_layout_additive  = {decoder_layout_additive}  (Strategy B2)")
        _strat = ("A" if latent_disentangle and not decoder_zs_cross_attn
                  else ("B1+B2" if decoder_layout_cross_attn and decoder_layout_additive
                  else ("B1" if decoder_layout_cross_attn
                  else ("B2" if decoder_layout_additive
                  else ("D" if decoder_zs_cross_attn else "C (baseline)")))))
        print(f"  Active strategy: {_strat}")
        print(f"  semantic_mode={semantic_mode} | color_residual={color_residual}")
        print(f"  latent_disentangle={latent_disentangle}  semantic_dims={semantic_dims}")
        print(f"  scene_layout_head={scene_layout_head}  decoder_fourier_pe={decoder_fourier_pe}")
        print(f"  token_cond={token_cond} adaln={token_cond_adaln}")
        print(f"  semantic_token_heads={semantic_token_heads}")
        print(f"  token_local_decoder={token_local_decoder}  (architectural fix; "
              f"1.6M-param shared per-token MLP replaces 777M flat GS_decoder)")
        print(f"{'='*70}")

        if semantic_token_heads and not latent_disentangle:
            raise ValueError("semantic_token_heads requires latent_disentangle=True")
        self.semantic_token_heads_flag = semantic_token_heads

        _N_SEM_TOKENS = 8
        _N_LAY_TOKENS = 7
        self._n_sem_tokens = _N_SEM_TOKENS
        self._n_lay_tokens = _N_LAY_TOKENS

        if semantic_token_heads or structured_layout_tokens:
            _color_in = embed_dim
            if structured_layout_tokens:
                _sem_in = _N_SEM_TOKENS * embed_dim
                _lay_in = _N_LAY_TOKENS * embed_dim
            else:
                _sem_in = semantic_dims - embed_dim
                _lay_in = semantic_dims - embed_dim
        else:
            _color_in = width
            _sem_in   = width
            _lay_in   = width

        self.mean_color_head      = None
        self.last_mean_color_pred = None
        if color_residual:
            self.mean_color_head = MeanColorHead(in_dim=_color_in)

        # Position-conditioned per-Gaussian colour/rotation refinement (optional).
        # Conditions a shared MLP on (per-token feature H_out, Fourier(final position))
        # and adds a high-frequency residual to colour/quaternion inside decode().
        self.pos_cond_heads = None
        if _POS_COND['enabled']:
            from .position_conditioned_heads import PositionConditionedHeads
            self.pos_cond_heads = PositionConditionedHeads(
                token_feat_dim=width, n_freqs=_POS_COND['n_freqs'],
                sigma=_POS_COND['sigma'], pos_scale=_POS_COND['pos_scale'],
                hidden=_POS_COND['hidden'], do_color=_POS_COND['color'],
                do_rotation=_POS_COND['rotation'])

        self.scene_semantic_module    = None
        self.last_scene_semantic_pred = None
        if scene_semantic_head:
            self.scene_semantic_module = SceneSemanticHead(in_dim=_sem_in)

        self.scene_layout_module    = None
        self.last_scene_layout_pred = None
        if scene_layout_head:
            self.scene_layout_module = SceneLayoutHead(in_dim=_lay_in)

        _lay_color_in = embed_dim
        if structured_layout_tokens:
            _lay_sem_in = _N_SEM_TOKENS * embed_dim
            _lay_lay_in = _N_LAY_TOKENS * embed_dim
        else:
            _lay_sem_in = (self._n_zs_tokens - 1) * embed_dim
            _lay_lay_in = (self._n_zs_tokens - 1) * embed_dim

        self.lay_color_head    = None
        self.lay_semantic_head = None
        self.lay_layout_head   = None

        _any_B_flag = decoder_layout_cross_attn or decoder_layout_additive
        if _any_B_flag:
            if color_residual:
                self.lay_color_head = MeanColorHead(in_dim=_lay_color_in)
            if scene_semantic_head:
                self.lay_semantic_head = SceneSemanticHead(in_dim=_lay_sem_in)
            if scene_layout_head:
                self.lay_layout_head = SceneLayoutHead(in_dim=_lay_lay_in)

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

        self.z_s_infonce_head      = None
        self.last_z_s_infonce_proj = None
        if latent_disentangle:
            self.z_s_infonce_head = SemanticTokenInfoNCEHead(
                in_dim=semantic_dims, proj_dim=128)

        self.zs_pool_proj_head   = None
        self.last_zs_pool_proj   = None
        self.last_zs_pool_hidden = None
        if latent_disentangle:
            self.zs_pool_proj_head = ZSTokenPoolProjectHead(
                n_tokens=self._n_zs_tokens, token_dim=embed_dim)

        self.anchor_pred_from_tokens            = None
        self.last_predicted_anchors_from_tokens = None
        if position_scaffold:
            n_tok = self._n_zg_tokens if decoder_zs_cross_attn else _Z_TOKENS
            self.anchor_pred_from_tokens = AnchorPredFromTokens(width=width, num_tokens=n_tok)

        # ── ANCHOR-RELATIVE LOCAL DECODING (spatially-anchored latent) ──────────
        # Scaffold-GS-style local decoding: each Gaussian's position is decoded as
        #     pos = anchor[token] + offset_scale * tanh(raw_offset)
        # The tanh BOUND is the locality pressure that the plain position_scaffold
        # path lacks. In the legacy path pos = raw + anchor with raw UNBOUNDED, so
        # "anchor + offset" is just a reparametrised absolute position and the anchor
        # carries no information the decoder is forced to use. Bounding the offset
        # makes it physically a LOCAL displacement, so the per-token anchor must
        # carry the coarse position. Paired with the adaptive Hilbert-block anchors
        # from the dataset (scaffold_mode='hilbert_block'), token k then owns a real
        # per-scene local cluster (Scaffold-GS: mu_i = x + O_i*l). offset_scale is a
        # single learnable, strictly-positive global parameter (exp-parameterised),
        # so the bound adapts to the typical within-block spread automatically.
        self.anchor_relative_decode = anchor_relative_decode
        self.anchor_teacher_force   = anchor_teacher_force
        self.log_offset_scale       = None
        if anchor_relative_decode:
            if self.anchor_pred_from_tokens is None:
                raise ValueError(
                    "anchor_relative_decode=True requires position_scaffold=True "
                    "(it provides the per-token anchor head). Enable position_scaffold.")
            self.log_offset_scale = nn.Parameter(
                torch.tensor(float(math.log(offset_scale_init))))
            print(f"  [ANCHOR-RELATIVE DECODE] pos = anchor + "
                  f"{offset_scale_init:.2f}*tanh(offset) | "
                  f"teacher_force={anchor_teacher_force} | learnable offset_scale")

        # ── GAUSSIANCUBE-STYLE FRAMED MICRO-PATTERN ─────────────────────────────
        # On top of the anchor-relative decode, replace the free within-block
        # offset with: framed canonical point set + SMALL residual, i.e.
        #     pos = anchor + R_block · (s_block · c[slot]) + micro_scale·tanh(resid)
        # c is a fixed unit-ball point set (no bits); (s_block, R_block) is the
        # per-token frame from BlockFramePredFromTokens; resid reuses the decoder's
        # position output, now bounded SMALL. Requires anchor_relative_decode (for
        # the per-token anchor + the raw-offset slot) and hilbert_block ids.
        self.micro_pattern          = micro_pattern
        self.micro_pattern_rotation = micro_pattern_rotation
        self.block_frame_pred       = None
        self.log_micro_offset_scale = None
        if micro_pattern:
            if not anchor_relative_decode:
                raise ValueError(
                    "micro_pattern=True requires anchor_relative_decode=True (it "
                    "provides the per-token anchor and the raw-offset slot).")
            _mp_ntok = self._n_zg_tokens if decoder_zs_cross_attn else _Z_TOKENS
            _mp_g    = -(-_N_GAUSSIANS // _mp_ntok)      # ceil(N / n_tokens) = g/token
            # register_buffer must NOT be preceded by a plain `self.canonical_pattern`
            # attribute (PyTorch raises "attribute already exists"), so it is only set
            # here (buffer) or in the else branch (None).
            self.register_buffer("canonical_pattern",
                                 _fibonacci_ball(_mp_g), persistent=False)   # [g,3]
            self.block_frame_pred = BlockFramePredFromTokens(
                width=width, num_tokens=_mp_ntok, use_rotation=micro_pattern_rotation)
            self.log_micro_offset_scale = nn.Parameter(
                torch.tensor(float(math.log(micro_offset_scale))))
            print(f"  [MICRO-PATTERN] pos = anchor + "
                  f"R_block·(s_block·c[{_mp_g}]) + {micro_offset_scale:.2f}*tanh(resid)"
                  f" | rotation={micro_pattern_rotation} | learnable residual scale")
        else:
            self.canonical_pattern = None

        self.zs_cond_decoder = None
        self.post_kl_g       = None
        self.post_kl_s       = None
        self.GS_decoder_new  = None
        if decoder_zs_cross_attn:
            self.post_kl_g = nn.Linear(embed_dim, width)
            self.post_kl_s = nn.Linear(embed_dim, width)
            nn.init.trunc_normal_(self.post_kl_g.weight, std=0.02)
            nn.init.trunc_normal_(self.post_kl_s.weight, std=0.02)
            self.zs_cond_decoder = ZSCondTransformerDecoder(
                width=width, heads=heads, layers=num_decoder_layers)
            self.GS_decoder_new = GS_decoder(
                3, 1024, num_tokens=self._n_zg_tokens, width=width,
                color_residual=color_residual)

        self.layout_projector          = None
        self.post_kl_layout            = None
        self.layout_additive_cond      = None
        self.zs_cond_decoder_B         = None
        self.GS_decoder_B              = None
        self.z_layout_infonce_head     = None
        self.z_layout_pool_head        = None
        self.last_z_layout             = None
        self.last_z_layout_proj        = None
        self.last_z_layout_pool_proj   = None
        self.last_z_layout_pool_hidden = None

        _any_B = decoder_layout_cross_attn or decoder_layout_additive
        if _any_B:
            self.layout_projector = Layout16Projector(
                in_dim=width, n_tokens=self._n_zs_tokens, token_dim=embed_dim)
            _lay_flat = self._n_zs_tokens * embed_dim
            self.z_layout_infonce_head = SemanticTokenInfoNCEHead(
                in_dim=_lay_flat, proj_dim=128)
            self.z_layout_pool_head = ZSTokenPoolProjectHead(
                n_tokens=self._n_zs_tokens, token_dim=embed_dim)
            print(f"  [Strategy B] Layout16Projector + InfoNCE heads active")

        if decoder_layout_cross_attn:
            self.post_kl_layout = nn.Linear(embed_dim, width)
            nn.init.trunc_normal_(self.post_kl_layout.weight, std=0.02)
            self.zs_cond_decoder_B = ZSCondTransformerDecoder(
                width=width, heads=heads, layers=num_decoder_layers)
            self.GS_decoder_B = GS_decoder(
                3, 1024, num_tokens=_Z_TOKENS, width=width,
                color_residual=color_residual)
            print(f"  [Strategy B1] cross-attn: 512 geom + z_layout K/V per layer")

        if decoder_layout_additive:
            self.layout_additive_cond = LayoutAdditiveConditioner(
                n_tokens=self._n_zs_tokens, token_dim=embed_dim, width=width)
            print(f"  [Strategy B2] additive: flatten(z_layout)→MLP→[B,{width}] broadcast bias")

        if decoder_layout_cross_attn and decoder_layout_additive:
            print(f"  [Strategy B3] both additive bias + cross-attn active")

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

        self.token_cond_adaln_flag = False
        self.adaLN_transformer     = None
        self.token_cond_mlp_B      = None
        self.token_cat_assign      = None
        fourier_out_dim = self.fourier_embedder.out_dim

        if decoder_zs_cross_attn:
            if token_cond or token_cond_adaln:
                print("  [INFO] decoder_zs_cross_attn=True: TokenCond/AdaLN disabled")
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

        self.seg_pred_head = None
        self.last_seg_pred = None
        if predict_seg_labels:
            self.seg_pred_head = SegPredHead(in_dim=14, num_cats=72)

        self.semantic_projection_hidden    = None
        self.semantic_projection_geometric = None
        self.semantic_distribution_head    = None
        if semantic_mode == 'hidden':
            self.semantic_projection_hidden = SemanticProjectionHead(1024, _N_GAUSSIANS, 32)
        elif semantic_mode == 'geometric':
            self.semantic_projection_geometric = SemanticProjectionHeadGeometric(14, _N_GAUSSIANS, 32, 128)
        elif semantic_mode == 'dist':
            self.semantic_distribution_head = SemanticDistributionHead(1024, 72)
        elif semantic_mode not in ('none', 'attention'):
            raise ValueError(f"Unknown semantic_mode: '{semantic_mode}'")

        # ────────────────────────────────────────────────────────────────────
        # TOKEN-LOCAL DECODER OVERRIDE
        # ────────────────────────────────────────────────────────────────────
        # When token_local_decoder=True, replace every standard GS_decoder
        # instance (already built above by super().__init__ and the strategy
        # branches below) with a TokenLocalDecoder. The TokenLocalDecoder
        # exposes the identical forward(x, return_hidden=False) interface and
        # output shape [B, 10000*14] as the flat MLP, so decode() and the
        # semantic-feature pipeline both work unchanged.
        #
        # GS_decoder instances that may exist at this point:
        #   - self.GS_decoder       (512 tokens) → Strategy A / B / C paths
        #   - self.GS_decoder_B     (512 tokens) → Strategy B1 path
        #   - self.GS_decoder_new   (_n_zg_tokens) → Strategy D path
        # Only the relevant decoder is actually used per forward, but consistent
        # replacement keeps checkpoint compatibility simple.
        #
        # Per-Gaussian InfoNCE: the SemanticProjectionHead receives the [B, 1024]
        # pooled hidden from TokenLocalDecoder.hidden_proj and produces per-
        # Gaussian features [B, 10000, 32] exactly as before. No changes needed
        # to the per_gaussian_features pipeline or the 6-tuple forward return.
        if token_local_decoder:
            print(f"\n  [TOKEN-LOCAL DECODER] Replacing flat GS_decoder(s) with "
                  f"shared per-token MLPs:")
            self.GS_decoder = TokenLocalDecoder(
                width=width, hidden_dim=512, num_tokens=512,
                num_gaussians=_N_GAUSSIANS, color_residual=color_residual)
            if self.GS_decoder_B is not None:
                self.GS_decoder_B = TokenLocalDecoder(
                    width=width, hidden_dim=512, num_tokens=_Z_TOKENS,
                    num_gaussians=_N_GAUSSIANS, color_residual=color_residual)
            if self.GS_decoder_new is not None:
                self.GS_decoder_new = TokenLocalDecoder(
                    width=width, hidden_dim=512, num_tokens=self._n_zg_tokens,
                    num_gaussians=_N_GAUSSIANS, color_residual=color_residual)

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

        if self.structured_latent:
            # Per-token latent: token k of z == encoder token k == Hilbert block k.
            # Use the per-token posterior DIRECTLY and skip the dense kl_emb_proj_mean(_g)
            # remix that would flatten + globally mix the 512 tokens (destroying the
            # spatial structure the local encoder just produced).
            moments   = self.pre_kl(latents)                     # [B, K, 2*embed_dim]
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            z_tok     = posterior.sample() if sample_posterior else posterior.mode()  # [B,K,embed_dim]
            mean, logvar = moments.chunk(2, dim=-1)              # each [B, K, embed_dim]
            self._mu_s_cache = None
            self._mu_g_cache = None
            Bsz     = z_tok.shape[0]
            mu      = mean.reshape(Bsz, -1)                      # [B, K*embed_dim]
            log_var = logvar.reshape(Bsz, -1)
            z       = z_tok.reshape(Bsz, -1)
            return shape_embed, mu, log_var, z, posterior

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

    # =========================================================================
    # LATENT PERCEPTUAL LOSS — DECODER TRANSFORMER FEATURE EXTRACTOR
    # =========================================================================
    #
    # Reference: Berrada et al., NeurIPS 2024 — arXiv:2411.04873
    #   "Boosting Latent Diffusion with Perceptual Objectives"
    #
    # Problem (Autoencoder-Diffusion Disconnect):
    #   Stage 2 flow matching generates z_g tokens whose DISTRIBUTION matches
    #   the encoder aggregate posterior (KL converges ~epoch 1400). However each
    #   generated z_g_gen is ~211 units from the nearest encoder z_g_clean in
    #   R^{15872}. The 777M GS_decoder was trained exclusively on encoder outputs
    #   and extrapolates poorly for off-manifold Stage 2 inputs → blurry Gaussians.
    #
    # Solution (LPL):
    #   Include the frozen decoder transformer in Stage 2 training objective.
    #   Each geometry DiT step, after predicting v_pred:
    #
    #     z_g_est   = x_t + (1-t) * v_pred                   [B, 496, 32]
    #     Z_est     = cat([z_s_clean, z_g_est],   dim=1)      [B, 512, 32]
    #     Z_clean   = cat([z_s_clean, z_g_clean], dim=1)      [B, 512, 32]
    #     feat_gen  = get_decoder_transformer_features(Z_est)   [B, 512, 384]
    #     feat_clean= get_decoder_transformer_features(Z_clean) [B, 512, 384]  # no_grad
    #     L_LPL     = MSE(feat_gen, feat_clean)
    #     L_total   = L_flow + lpl_weight * L_LPL
    #
    # Feature level: transformer output H_out [B, 512, 384]
    #   — BEFORE the 777M GS_decoder MLP
    #   — only 21M transformer in backward path (~3.3GB extra activation memory)
    #   — captures decoder's spatial per-token understanding of the scene
    #
    # Gradient: L_LPL → H_out_gen → Z_est → z_g_est → v_pred → DiT parameters
    #   Stage 1 decoder weights: requires_grad=False → NEVER updated.
    #
    # Enable in Stage 2: --lpl_weight 0.01  (default 0.0 = disabled)

    def get_decoder_transformer_features(self, latents, z_layout=None):
        """
        Return decoder transformer output H_out for Latent Perceptual Loss.

        Runs the Stage 1 decoder forward pass up to and including the 12-layer
        self-attention transformer, stopping BEFORE the 777M GS_decoder MLP.

        Exactly mirrors the corresponding branch of decode() for all strategies.
        Verified line-by-line against the actual decode() source code.

        Parameters
        ----------
        latents  : Z [B, 512, 32]  full latent (z_s and z_g concatenated)
        z_layout : [B, 16, 32]     Strategy B1/B2/B3 only; pass None for A and D

        Returns
        -------
        H_out : [B, 512, 384]  (Strategy A / B)   decoder transformer output
                [B, 496, 384]  (Strategy D)        cross-attn decoder output

        Gradient flow
        -------------
        Do NOT use torch.no_grad() for the generated latent Z_est — gradients
        must flow: L_LPL → feat_gen → Z_est → z_g_est → v_pred → DiT params.

        DO use torch.no_grad() for the clean reference:
            with torch.no_grad():
                feat_clean = shape_model.get_decoder_transformer_features(Z_clean)

        Stage 1 decoder parameters (requires_grad=False) are NEVER updated.
        """
        B = latents.shape[0]
        _any_B = (self.decoder_layout_cross_attn or self.decoder_layout_additive)

        if _any_B and z_layout is not None:
            # ── Strategy B1 / B2 / B3 ────────────────────────────────────────
            # Exact mirror of decode() _any_B branch:
            H = self.post_kl(latents)                               # [B, 512, 384]
            if self.decoder_layout_additive and self.layout_additive_cond is not None:
                bias = self.layout_additive_cond(z_layout)          # [B, 384]
                H    = H + bias.unsqueeze(1)                        # broadcast: [B, 512, 384]
            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H = H + self.decoder_fourier_pe_module(B, H.device)
            elif self.decoder_pos_emb is not None:
                H = H + self.decoder_pos_emb.unsqueeze(0)
            if self.decoder_layout_cross_attn and self.zs_cond_decoder_B is not None:
                H_lay = self.post_kl_layout(z_layout)               # [B, 16, 384]
                return self.zs_cond_decoder_B(H, H_lay)             # [B, 512, 384]
            else:
                return self.transformer(H)                          # [B, 512, 384]

        elif self.decoder_zs_cross_attn:
            # ── Strategy D ───────────────────────────────────────────────────
            # Exact mirror of decode() decoder_zs_cross_attn branch:
            n_s = self._n_zs_tokens                                 # 16
            H_g = self.post_kl_g(latents[:, n_s:, :])              # [B, 496, 384]
            H_s = self.post_kl_s(latents[:, :n_s, :])              # [B,  16, 384]
            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H_g = H_g + self.decoder_fourier_pe_module(B, H_g.device)
            return self.zs_cond_decoder(H_g, H_s)                  # [B, 496, 384]

        else:
            # ── Strategy A  (primary LPL use case) ───────────────────────────
            # Exact mirror of decode() legacy/else branch.
            # token_cond_adaln is always False for Stage 2 because load_stage1()
            # sets p.token_cond=False → self.token_cond_adaln_flag=False,
            # so the branch always falls through to self.transformer(H).
            H = self.post_kl(latents)                               # [B, 512, 384]
            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H = H + self.decoder_fourier_pe_module(B, H.device)
            elif self.decoder_pos_emb is not None:
                H = H + self.decoder_pos_emb.unsqueeze(0)
            return self.transformer(H)                              # [B, 512, 384]

    # ── DECODE ────────────────────────────────────────────────────────────────

    def decode(self, latents, volume_queries=None, return_semantic_features=False,
               shape_embed=None, scaffold_anchors=None, scaffold_token_ids=None,
               z_layout=None):
        B = latents.shape[0]

        _any_B = (self.decoder_layout_cross_attn or self.decoder_layout_additive)
        if _any_B and z_layout is not None:
            H = self.post_kl(latents)
            if self.decoder_layout_additive and self.layout_additive_cond is not None:
                bias = self.layout_additive_cond(z_layout)
                H    = H + bias.unsqueeze(1)
            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H = H + self.decoder_fourier_pe_module(B, H.device)
            elif self.decoder_pos_emb is not None:
                H = H + self.decoder_pos_emb.unsqueeze(0)
            if self.decoder_layout_cross_attn and self.zs_cond_decoder_B is not None:
                H_lay = self.post_kl_layout(z_layout)
                H     = self.zs_cond_decoder_B(H, H_lay)
            else:
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
            gs_dec = self.GS_decoder_B if (self.decoder_layout_cross_attn and self.GS_decoder_B is not None) else self.GS_decoder
            if need_hidden:
                reconstruction, hidden = gs_dec(latents_flat, return_hidden=True)
            else:
                hidden = None
                reconstruction = gs_dec(latents_flat)
            _fixed_ids = FIXED_TOKEN_IDS_512

        elif self.decoder_zs_cross_attn:
            n_s     = self._n_zs_tokens
            z_s_raw = latents[:, :n_s, :]
            z_g_raw = latents[:, n_s:, :]
            H_g = self.post_kl_g(z_g_raw)
            H_s = self.post_kl_s(z_s_raw)
            if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
                H_g = H_g + self.decoder_fourier_pe_module(B, H_g.device)
            H_out = self.zs_cond_decoder(H_g, H_s)

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
                hidden = None
                reconstruction = self.GS_decoder_new(latents_flat)
            _fixed_ids = FIXED_TOKEN_IDS_496

        else:
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

        if self.anchor_relative_decode and pred_anchors is not None:
            # ── Anchor-relative LOCAL decoding (Scaffold-GS style) ──────────────
            #     pos = anchor[token] + offset_scale * tanh(raw_offset)
            # raw_offset is the decoder's raw position output (pred_3d[...,0:3]).
            # The offset is BOUNDED, so the anchor carries the coarse position.
            # Anchor source:
            #   - teacher-forced GT block centroids (diagnostic upper bound) when
            #     anchor_teacher_force and scaffold_anchors is provided; isolates
            #     the decoder from the encoder's ability to predict anchors.
            #   - else the per-token PREDICTED anchors (Stage-2-compatible: they are
            #     a deterministic function of the latent, so the Stage-2 DiT that
            #     generates the latent also controls them).
            # The OUTPUT is absolute positions, so the training recon loss uses the
            # absolute target (see --anchor_relative_decode branch in gs_can3tok_2).
            pred_3d = reconstruction.reshape(B, _N_GAUSSIANS, 14)
            raw_off = pred_3d[:, :, 0:3]
            if self.anchor_teacher_force:
                assert scaffold_anchors is not None, (
                    "anchor_teacher_force=True needs scaffold_anchors forwarded to "
                    "decode(); if the wrapper does not pass it, run with predicted "
                    "anchors (anchor_teacher_force=False).")
                anchor_src = scaffold_anchors
            else:
                anchor_src = pred_anchors
            if scaffold_token_ids is not None:
                idx_3d    = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, 3)
                anchor_pg = torch.gather(anchor_src, 1, idx_3d)
            else:
                anchor_pg = anchor_src[:, _fixed_ids.to(anchor_src.device), :]
            pred_3d = pred_3d.clone()
            if self.micro_pattern and self.block_frame_pred is not None:
                # ── Framed canonical micro-pattern + small residual ─────────────
                #     pos = anchor + R_block·(s_block·c[slot]) + micro·tanh(resid)
                # Frame is per latent token; gather it by the SAME token map as the
                # anchor. Canonical slot = i % g aligns with the decoder's layout
                # (hilbert_block: token = i // g). raw_off is the decoder's position
                # output, here used as the SMALL residual.
                s_tok, R_tok = self.block_frame_pred(H_out)         # [B,T,3], [B,T,3,3]|None
                if scaffold_token_ids is not None:
                    sid = scaffold_token_ids.long()                 # [B,N]
                else:
                    sid = _fixed_ids.to(raw_off.device).unsqueeze(0).expand(B, -1)
                s_pg = torch.gather(s_tok, 1, sid.unsqueeze(-1).expand(-1, -1, 3))
                g_pt = self.canonical_pattern.shape[0]
                slot = torch.arange(_N_GAUSSIANS, device=raw_off.device) % g_pt
                c_pg = self.canonical_pattern.to(raw_off.dtype)[slot].unsqueeze(0).expand(B, -1, -1)
                framed = s_pg * c_pg                                # anisotropic scale
                if R_tok is not None:
                    R_pg = torch.gather(
                        R_tok.reshape(B, R_tok.shape[1], 9), 1,
                        sid.unsqueeze(-1).expand(-1, -1, 9)
                    ).reshape(B, _N_GAUSSIANS, 3, 3)
                    framed = torch.einsum('bnij,bnj->bni', R_pg, framed)   # orient to surface
                micro = self.log_micro_offset_scale.exp()
                pred_3d[:, :, 0:3] = anchor_pg + framed + micro * torch.tanh(raw_off)
            else:
                scale = self.log_offset_scale.exp()
                pred_3d[:, :, 0:3] = anchor_pg + scale * torch.tanh(raw_off)
            reconstruction = pred_3d.reshape(B, -1)

        elif pred_anchors is not None:
            # ── Legacy position_scaffold path (UNBOUNDED additive anchor) ───────
            pred_3d = reconstruction.reshape(B, _N_GAUSSIANS, 14)
            if scaffold_token_ids is not None:
                idx_3d = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, 3)
                dc     = torch.gather(pred_anchors, 1, idx_3d)
            else:
                dc = pred_anchors[:, _fixed_ids.to(pred_anchors.device), :]
            pred_3d[:, :, 0:3] += dc
            reconstruction = pred_3d.reshape(B, -1)

        # Position-conditioned refinement of per-Gaussian colour / rotation, using the
        # FINAL positions just assembled and the per-token feature H_out broadcast to each
        # of its g Gaussians (matching the token-local decoder's token->Gaussian mapping).
        if self.pos_cond_heads is not None:
            _T  = H_out.shape[1]
            _g  = (_N_GAUSSIANS + _T - 1) // _T
            _Wd = H_out.shape[2]
            tok_pg = (H_out.unsqueeze(2).expand(B, _T, _g, _Wd)
                          .reshape(B, _T * _g, _Wd)[:, :_N_GAUSSIANS, :])
            reconstruction = self.pos_cond_heads(
                reconstruction.reshape(B, _N_GAUSSIANS, 14), tok_pg).reshape(B, -1)

        self.last_seg_pred = None
        if self.seg_pred_head is not None:
            self.last_seg_pred = self.seg_pred_head(reconstruction.reshape(B, _N_GAUSSIANS, 14))

        semantic_features = None
        if return_semantic_features and hidden is not None:
            if self.semantic_mode == 'hidden':
                semantic_features = self.semantic_projection_hidden(hidden)
            elif self.semantic_mode == 'geometric':
                semantic_features = self.semantic_projection_geometric(
                    reconstruction.reshape(B, _N_GAUSSIANS, 14))
            elif self.semantic_mode == 'dist':
                semantic_features = self.semantic_distribution_head(hidden)

        return reconstruction, semantic_features

    # ── FORWARD ───────────────────────────────────────────────────────────────

    def forward(self, pc, feats, volume_queries, sample_posterior=True,
                scaffold_anchors=None, scaffold_token_ids=None,
                return_semantic_features=None):
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)
        _se = self._shape_embed_cache

        self.last_z_s_infonce_proj = None
        if self.z_s_infonce_head is not None:
            self.last_z_s_infonce_proj = self.z_s_infonce_head(
                z[:, :self.semantic_dims])

        self.last_zs_pool_proj   = None
        self.last_zs_pool_hidden = None
        if self.zs_pool_proj_head is not None:
            _z_s_toks = z.reshape(z.shape[0], _N_LATENT_TOKENS, self.embed_dim)[:, :self._n_zs_tokens, :]
            self.last_zs_pool_proj, self.last_zs_pool_hidden = \
                self.zs_pool_proj_head(_z_s_toks)

        self.last_z_layout      = None
        self.last_z_layout_proj = None
        _any_B = self.decoder_layout_cross_attn or self.decoder_layout_additive
        if _any_B and self.layout_projector is not None:
            self.last_z_layout = self.layout_projector(_se)
            if self.z_layout_infonce_head is not None:
                z_lay_flat = self.last_z_layout.reshape(z.shape[0], -1)
                self.last_z_layout_proj = self.z_layout_infonce_head(z_lay_flat)
            self.last_z_layout_pool_proj   = None
            self.last_z_layout_pool_hidden = None
            if self.z_layout_pool_head is not None:
                self.last_z_layout_pool_proj, self.last_z_layout_pool_hidden = \
                    self.z_layout_pool_head(self.last_z_layout)

        if (not _any_B and self.latent_disentangle
                and self.z_s_infonce_head is not None):
            self.last_z_layout_proj = self.last_z_s_infonce_proj

        _lay_src = self.last_z_layout

        if self.semantic_token_heads_flag or self.structured_layout_tokens_flag:
            _ed = self.embed_dim
            _sd = self.semantic_dims
            self.last_mean_color_pred = (
                self.mean_color_head(z[:, :_ed]) if self.mean_color_head else None)
            if self.structured_layout_tokens_flag:
                _n_s  = self._n_sem_tokens
                z_sem = z[:, _ed : _ed + _n_s * _ed]
                z_lay = z[:, _ed + _n_s * _ed : _sd]
                self.last_scene_semantic_pred = (
                    self.scene_semantic_module(z_sem) if self.scene_semantic_module else None)
                self.last_scene_layout_pred = (
                    self.scene_layout_module(z_lay) if self.scene_layout_module else None)
            else:
                z_sem = z[:, _ed:_sd]
                self.last_scene_semantic_pred = (
                    self.scene_semantic_module(z_sem) if self.scene_semantic_module else None)
                self.last_scene_layout_pred = (
                    self.scene_layout_module(z_sem) if self.scene_layout_module else None)
        elif _lay_src is not None and _any_B:
            B_cur = z.shape[0]
            self.last_mean_color_pred = (
                self.lay_color_head(_lay_src[:, 0, :]) if self.lay_color_head else None)
            if self.structured_layout_tokens_flag:
                _n_s  = self._n_sem_tokens
                z_sem = _lay_src[:, 1 : 1+_n_s, :].reshape(B_cur, -1)
                z_lay = _lay_src[:, 1+_n_s : , :].reshape(B_cur, -1)
                self.last_scene_semantic_pred = (
                    self.lay_semantic_head(z_sem) if self.lay_semantic_head else None)
                self.last_scene_layout_pred = (
                    self.lay_layout_head(z_lay) if self.lay_layout_head else None)
            else:
                _lay_all = _lay_src[:, 1:, :].reshape(B_cur, -1)
                self.last_scene_semantic_pred = (
                    self.lay_semantic_head(_lay_all) if self.lay_semantic_head else None)
                self.last_scene_layout_pred = (
                    self.lay_layout_head(_lay_all) if self.lay_layout_head else None)
        else:
            self.last_scene_layout_pred = (
                self.scene_layout_module(_se) if self.scene_layout_module else None)

        latents = z.reshape(z.shape[0], _N_LATENT_TOKENS, self.embed_dim)
        _rsf = self.training if return_semantic_features is None else return_semantic_features

        UV_gs_recover, per_gaussian_features = self.decode(
            latents, volume_queries,
            return_semantic_features=_rsf,
            shape_embed=_se,
            scaffold_anchors=scaffold_anchors,
            scaffold_token_ids=scaffold_token_ids,
            z_layout=self.last_z_layout)

        if not self.semantic_token_heads_flag and not _any_B:
            self.last_mean_color_pred = (
                self.mean_color_head(_se) if self.mean_color_head else None)
            self.last_scene_semantic_pred = (
                self.scene_semantic_module(_se) if self.scene_semantic_module else None)

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features