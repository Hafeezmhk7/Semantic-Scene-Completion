# -*- coding: utf-8 -*-
"""
sal_perceiver.py  —  Can3Tok VAE
==================================
INFERENCE FIX — AnchorPredFromTokens
======================================
Previously:
  position_offsets  = coord - smooth_anchor   (GT, computed in dataset)
  abs_pos at save   = decoder_offsets + GT smooth_anchor   ← GT data leaked

Now:
  AnchorPredFromTokens: transformer_tokens [B,512,width] → predicted_anchors [B,512,3]
  abs_pos = decoder_offsets + predicted_anchors[:, token_ids, :]
  Entirely from z — no encoder, no GT batch data needed at second-stage inference.

  Training:  scaffold_token_ids (GT) used to index predicted_anchors → accurate DC
  Inference: fixed assignment  j → j*512//40000  → no GT needed

  AnchorPositionHead (shape_embed → [B,512,3]) removed.
  All other ideas (0,1,2,3) unchanged.

GRADIENT PATHS (updated):
  PATH 1 — Reconstruction:          L_recon → GS_decoder → post_kl → transformer → mu
  PATH 2 — KL:                      L_KL → mu, log_var → encoder
  PATH 3 — Mean Color:              L_color → MeanColorHead → shape_embed
  PATH 4 — Scene Semantic:          L_scene_kl → SceneSemanticHead → shape_embed
  PATH 5 — Layout Centroids:        L_layout → SceneLayoutHead → shape_embed
  PATH 6 — Anchor from Tokens:      L_anchor → AnchorPredFromTokens → transformer → z
  PATH 7 — Per-Gaussian InfoNCE:    L_infonce → SemanticProjectionHead → decoder hidden
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
# SHAPE_EMBED AUXILIARY HEADS
# ============================================================================

class MeanColorHead(nn.Module):
    def __init__(self, in_dim=384):
        super().__init__()
        hidden = max(32, min(64, in_dim))
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 3), nn.Sigmoid())
        total = sum(p.numel() for p in self.parameters())
        print(f"[MeanColorHead] [B,{in_dim}] -> [B,3] sigmoid | {total:,} params")

    def forward(self, x):
        return self.head(x)


class SceneSemanticHead(nn.Module):
    NUM_LABELS = 72
    def __init__(self, in_dim=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, self.NUM_LABELS))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SceneSemanticHead] [B,{in_dim}] -> [B,72] softmax | {total:,} params")

    def forward(self, x):
        return F.softmax(self.head(x), dim=-1)


class SceneLayoutHead(nn.Module):
    NUM_CATS = 72
    def __init__(self, in_dim=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, self.NUM_CATS * 3))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SceneLayoutHead] [B,{in_dim}] -> [B,72,3] per-cat centroids | {total:,} params")

    def forward(self, x):
        B = x.shape[0]
        return self.head(x).reshape(B, self.NUM_CATS, 3)


class SpatialSemanticHead(nn.Module):
    NUM_CATS = 72
    def __init__(self, width=384, num_tokens=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width + 3, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),        nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, self.NUM_CATS))
        self.num_tokens = num_tokens
        total = sum(p.numel() for p in self.parameters())
        print(f"[SpatialSemanticHead] [B,{width}+3] -> [B,{num_tokens},72] | {total:,} params")

    def forward(self, shape_embed, voxel_centers):
        B, K, _ = voxel_centers.shape
        se_exp   = shape_embed.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([se_exp, voxel_centers], dim=-1)
        flat     = combined.reshape(B * K, -1)
        out      = self.head(flat).reshape(B, K, self.NUM_CATS)
        return F.softmax(out, dim=-1)


# ============================================================================
# NEW: 3D FOURIER POSITIONAL ENCODING FOR DECODER
# ============================================================================

class FourierDecoderPE(nn.Module):
    """
    3D Fourier positional encoding over the 8×8×8 scaffold voxel grid.

    WHY THIS OVER LEARNABLE PE:
      Learnable PE treats each of the 512 token indices as independent.
      The model must discover from data that token 0 (voxel [0,0,0]) is
      spatially adjacent to token 1 (voxel [0,0,1]) — a geometric prior
      that is completely absent from a lookup table.

      Fourier PE over the actual 3D voxel grid encodes spatial proximity
      by construction: nearby voxels get similar PE vectors, so self-attention
      is initialised with a local-spatial bias that matches the inductive prior
      that nearby voxels carry correlated geometric content.

      This is consistent with the encoder's Fourier embedder (spectral continuity):
      the encoder uses Fourier features over 3D Gaussian positions; the decoder
      should use the same spectral basis for the tokens that represent those regions.

    Coordinates: normalised to [-1, 1]³ over the 8³ grid.
    """
    SCAFFOLD_DIMS = 8

    def __init__(self, fourier_embedder, width):
        super().__init__()
        S = self.SCAFFOLD_DIMS
        coords = []
        for i in range(S):
            for j in range(S):
                for k in range(S):
                    ci = (2.0 * i / (S - 1)) - 1.0
                    cj = (2.0 * j / (S - 1)) - 1.0
                    ck = (2.0 * k / (S - 1)) - 1.0
                    coords.append([ci, cj, ck])
        self.register_buffer('voxel_coords',
                             torch.tensor(coords, dtype=torch.float32))  # [512, 3]
        # Re-use the model's existing FourierEmbedder (no extra learnable params)
        self.fourier_embedder = fourier_embedder
        fourier_dim = fourier_embedder.out_dim
        # Learnable projection → model width (only learned component)
        self.proj = nn.Linear(fourier_dim, width)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        total = sum(p.numel() for p in self.proj.parameters())
        print(f"[FourierDecoderPE] 8³ voxel grid → Fourier({fourier_dim})"
              f" → Linear({width}) | {total:,} proj params")

    def forward(self, B, device):
        """Returns [B, 512, width] broadcast from [1, 512, width]."""
        coords  = self.voxel_coords.to(device)              # [512, 3]
        fourier = self.fourier_embedder(coords.unsqueeze(0)) # [1, 512, fourier_dim]
        pe      = self.proj(fourier)                         # [1, 512, width]
        return pe.expand(B, -1, -1)                         # [B, 512, width]


# ============================================================================
# NEW: PER-LAYER ADALN-ZERO CONDITIONED TRANSFORMER DECODER
# ============================================================================

def _modulate(h, shift, scale):
    """Element-wise AdaLN modulation. h, shift, scale: [B, T, D] or [B, D]."""
    return h * (1.0 + scale) + shift


class AdaLNBlock(nn.Module):
    """
    Single transformer block with per-token AdaLN-Zero conditioning.

    WHY ADALN OVER ONCE-BEFORE-STACK ADDITIVE BIAS:
      The current TokenCond B adds a spatial bias once before all 12 layers.
      After that, 12 rounds of self-attention dilute and redistribute the signal
      with no mechanism to re-inject it. By layer 12, the conditioning effect
      is heavily attenuated.

      AdaLN-Zero applies scale + shift + gate at EVERY layer, allowing the
      conditioning to modulate which features are amplified or suppressed
      throughout the full depth of the transformer. The gate is initialised to
      zero (identity) so the block starts as a plain transformer and learns to
      use the conditioning signal progressively.

      Reference: DiT (Peebles & Xie, ICCV 2023) — AdaLN-Zero reduces FID
      10× compared to additive conditioning at identical parameter count.

    Conditioning: per-token semantic centroid (from TokenCond B pipeline),
    Fourier-encoded. Each token is modulated by its own spatial-semantic signal.
    """

    def __init__(self, width, heads, cond_dim):
        super().__init__()
        # elementwise_affine=False: AdaLN supplies scale/shift externally
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(width, heads, batch_first=True, bias=False)
        self.ffn   = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width))
        # Produces: [shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn]
        # Zero-init → identity at start (safe to add to pre-trained checkpoint)
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * width, bias=True))
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x, c):
        """
        x: [B, T, width]   — token sequence
        c: [B, T, cond_dim] — per-token conditioning (semantic centroid, Fourier-encoded)
        """
        B, T, W = x.shape
        mod = self.adaLN_mod(c.reshape(B * T, -1)).reshape(B, T, 6 * W)
        sh_a, sc_a, ga, sh_f, sc_f, gf = mod.chunk(6, dim=-1)  # each [B, T, W]

        # Attention branch
        h_a, _ = self.attn(_modulate(self.norm1(x), sh_a, sc_a),
                            _modulate(self.norm1(x), sh_a, sc_a),
                            _modulate(self.norm1(x), sh_a, sc_a))
        x = x + ga * h_a

        # FFN branch
        x = x + gf * self.ffn(_modulate(self.norm2(x), sh_f, sc_f))
        return x


class AdaLNTransformerDecoder(nn.Module):
    """
    Full 12-layer decoder transformer with per-token AdaLN-Zero conditioning.

    Replaces self.transformer in decode() when token_cond_adaln=True AND
    a valid conditioning signal (TokenCond B semantic centroids) is available.
    Falls back to self.transformer automatically if conditioning is absent.

    cond_dim: Fourier embedding dimension of the per-token semantic centroid.
              = fourier_embedder.out_dim (typically 48 for num_freqs=8, input_dim=3).
    """

    def __init__(self, width, heads, layers, cond_dim):
        super().__init__()
        self.blocks   = nn.ModuleList(
            [AdaLNBlock(width, heads, cond_dim) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(width)
        total = sum(p.numel() for p in self.parameters())
        print(f"[AdaLNTransformerDecoder] {layers}× AdaLNBlock "
              f"(width={width}, heads={heads}, cond={cond_dim}) | {total/1e6:.2f}M params")
        print(f"  AdaLN-Zero init: blocks start as identity → safe checkpoint resume")
        print(f"  Conditioning: per-token semantic centroid (Fourier-encoded) at every layer")

    def forward(self, x, c):
        """x: [B, T, width], c: [B, T, cond_dim]"""
        for block in self.blocks:
            x = block(x, c)
        return self.norm_out(x)


# ============================================================================
# NEW: ANCHOR PREDICTION FROM DECODER TOKENS
# ============================================================================

class AnchorPredFromTokens(nn.Module):
    """
    Predicts scaffold voxel anchors from post-transformer decoder tokens.

    WHY THIS EXISTS:
      Previously, scaffold_anchors were GT quantities from the dataset, added
      back to decoder position outputs at PLY save time (smooth_anchor path).
      At second-stage diffusion inference, no GT data is available — only z.

      This head runs inside decode() on the post-transformer tokens [B,512,width]
      which are derived entirely from z. It produces predicted_anchors [B,512,3]
      which are used as the DC term for position recovery.

    TRAINING:   scaffold_token_ids from batch (spatially accurate assignment)
                predicted_anchors[b, scaffold_token_ids[b, j], :] added to output j
                Supervised: MSE(predicted_anchors, GT_scaffold_anchors)

    INFERENCE:  fixed assignment: j → j*512//40000  (no GT needed)
                predicted_anchors[:, fixed_ids, :] added to output

    GRADIENT PATH:
      L_anchor (MSE) → AnchorPredFromTokens → transformer tokens → post_kl → z → encoder
      This forces the latent to encode spatial voxel layout.
    """
    NUM_TOKENS = 512

    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),   nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 3))
        total = sum(p.numel() for p in self.parameters())
        print(f"[AnchorPredFromTokens] [B,512,{width}] -> [B,512,3] | {total:,} params")
        print(f"  Self-contained at inference: scaffold anchors from z alone, no GT needed")
        print(f"  Training:  token_ids from GT scaffold_token_ids (accurate spatial DC)")
        print(f"  Inference: fixed assignment j -> j*512//40000 (no GT dependency)")

    def forward(self, transformer_tokens):
        """
        transformer_tokens: [B, 512, width]
        returns:            [B, 512, 3]  — one predicted anchor per token/voxel
        """
        B, T, W = transformer_tokens.shape
        return self.head(transformer_tokens.reshape(B * T, W)).reshape(B, T, 3)


# Fixed Gaussian-to-token assignment used at inference
# Gaussian j → token j*N_TOKENS//N_GAUSSIANS
# Distributes 40000 Gaussians evenly across 512 tokens, no GT data needed
_N_GAUSSIANS = 40_000
_N_TOKENS    = 512
FIXED_TOKEN_IDS = torch.arange(_N_GAUSSIANS) * _N_TOKENS // _N_GAUSSIANS  # [40000], values 0-511


# ============================================================================
# NEW: THREE IDEAS — SPATIAL INDUCTIVE BIAS MODULES (unchanged)
# ============================================================================

class SegPredHead(nn.Module):
    NUM_CATS = 72
    def __init__(self, in_dim=14, num_cats=72):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, num_cats))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SegPredHead] [B,40000,{in_dim}] -> [B,40000,{num_cats}] | {total:,} params")

    def forward(self, gaussian_params):
        B, N, D = gaussian_params.shape
        return self.head(gaussian_params.reshape(B * N, D)).reshape(B, N, self.NUM_CATS)


class TokenCondMLP(nn.Module):
    def __init__(self, fourier_dim, width):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width))
        total = sum(p.numel() for p in self.parameters())
        print(f"  [TokenCondMLP] fourier({fourier_dim}) -> token_bias({width}) | {total:,} params")

    def forward(self, fourier_encoded):
        B, T, D = fourier_encoded.shape
        return self.mlp(fourier_encoded.reshape(B * T, D)).reshape(B, T, -1)


class SpatialAwareDecoder(nn.Module):
    def __init__(self, token_dim, fourier_embedder, hidden_dim=256, color_residual=False):
        super().__init__()
        self.fourier_embedder = fourier_embedder
        fourier_dim = fourier_embedder.out_dim
        self.color_residual = color_residual
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + fourier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 14))
        total = sum(p.numel() for p in self.mlp.parameters())
        print(f"[SpatialAwareDecoder] token({token_dim})+fourier({fourier_dim})"
              f" -> hidden({hidden_dim}) -> 14 | {total:,} MLP params")

    def forward(self, transformer_tokens, scaffold_anchors, scaffold_token_ids):
        B, T, D = transformer_tokens.shape
        N = scaffold_token_ids.shape[1]
        idx_for_anchors = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, 3)
        per_gaussian_anchor = torch.gather(scaffold_anchors, 1, idx_for_anchors)
        spatial_enc = self.fourier_embedder(per_gaussian_anchor)
        idx_for_tokens = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, D)
        token_feats = torch.gather(transformer_tokens, 1, idx_for_tokens)
        combined = torch.cat([token_feats, spatial_enc], dim=-1)
        raw = self.mlp(combined.reshape(B * N, -1)).reshape(B, N, 14)
        pos   = raw[:, :, 0:3]
        color = (raw[:, :, 3:6] if self.color_residual
                 else torch.clamp(raw[:, :, 3:6], 0.0, 1.0))
        opac  = torch.sigmoid(raw[:, :, 6:7])
        scale = torch.exp(raw[:, :, 7:10])
        quat  = F.normalize(raw[:, :, 10:14], p=2, dim=-1)
        out = torch.cat([pos, color, opac, scale, quat], dim=-1)
        return out.reshape(B, -1)


# ============================================================================
# PER-GAUSSIAN INFONCE HEADS (unchanged)
# ============================================================================

class SemanticProjectionHead(nn.Module):
    def __init__(self, hidden_dim=1024, num_gaussians=40000, feature_dim=32):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim   = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, num_gaussians * feature_dim))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHead] [B,{hidden_dim}] -> [B,{num_gaussians},{feature_dim}] | {total/1e6:.3f}M params")

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
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticDistributionHead] [B,{hidden_dim}] -> [B,{num_labels}] | {total/1e6:.3f}M params")

    def forward(self, hidden):
        return self.head(hidden)


class SemanticProjectionHeadGeometric(nn.Module):
    def __init__(self, gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.projection = nn.Sequential(
            nn.Linear(gaussian_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHeadGeometric] [B,{num_gaussians},{gaussian_dim}] -> [B,{num_gaussians},{feature_dim}] | {total/1e6:.3f}M params")

    def forward(self, gaussians):
        B, N, D = gaussians.shape
        return F.normalize(
            self.projection(gaussians.reshape(B * N, D)).reshape(B, N, -1),
            p=2, dim=-1)


# ============================================================================
# ENCODER (unchanged)
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
        x_y = np.linspace(-8, 8, voxel_reso)
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
        bs              = pc.shape[0]
        voxel_centers   = pc[:, :, 0:3]
        xyz_actual      = pc[:, :, 4:7]
        gaussian_params = feats[:, :, 7:]
        data = torch.cat([
            self.fourier_embedder(xyz_actual),
            self.fourier_embedder_ID(voxel_centers),
            gaussian_params,
        ], dim=-1).to(dtype=torch.float32)
        data    = self.input_proj(data)
        query   = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)
        if self.ln_post is not None:
            latents = self.ln_post(latents)
        return latents, pc

    def forward(self, pc, feats=None):
        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


# ============================================================================
# GEOMETRY DECODER (unchanged)
# ============================================================================

class CrossAttentionDecoder(nn.Module):
    def __init__(self, *, device, dtype, num_latents, out_channels,
                 fourier_embedder, width, heads, init_scale=0.25,
                 qkv_bias=True, flash=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint   = use_checkpoint
        self.fourier_embedder = fourier_embedder
        self.query_proj       = nn.Linear(
            fourier_embedder.out_dim, width, device=device, dtype=dtype)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, n_data=num_latents, width=width,
            heads=heads, init_scale=init_scale, qkv_bias=qkv_bias, flash=flash)
        self.ln_post     = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries, latents):
        queries = self.query_proj(self.fourier_embedder(queries))
        x       = self.cross_attn_decoder(queries, latents)
        return self.output_proj(self.ln_post(x))

    def forward(self, queries, latents):
        return checkpoint(self._forward, (queries, latents),
                          self.parameters(), self.use_checkpoint)


class GaussianSemanticAttentionHead(CrossAttentionDecoder):
    def forward(self, gaussian_xyz, scene_tokens):
        return F.normalize(super().forward(gaussian_xyz, scene_tokens), p=2, dim=-1)


# ============================================================================
# GS DECODER MLP (unchanged)
# ============================================================================

class GS_decoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56,
                 color_residual=False):
        super().__init__()
        self.D, self.W      = D, W
        self.color_residual = color_residual
        self.pts_linears    = nn.ModuleList([nn.Linear(input_ch, W)])
        for i in range(D - 1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.LayerNorm(W))
            self.pts_linears.append(nn.ReLU())
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, return_hidden=False):
        for layer in self.pts_linears:
            x = layer(x)
        hidden = x
        raw    = self.output_linear(x).reshape(x.shape[0], 40_000, 14)
        pos    = raw[:, :, 0:3]
        color  = raw[:, :, 3:6] if self.color_residual \
                 else torch.clamp(raw[:, :, 3:6], 0.0, 1.0)
        opac   = torch.sigmoid(raw[:, :, 6:7])
        scale  = torch.exp(raw[:, :, 7:10])
        quat   = F.normalize(raw[:, :, 10:14], p=2, dim=-1)
        out    = torch.cat([pos, color, opac, scale, quat], dim=-1).reshape(x.shape[0], -1)
        return (out, hidden) if return_hidden else out


# ============================================================================
# BASE PERCEIVER (unchanged)
# ============================================================================

class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False, color_residual=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_latents    = num_latents
        self.fourier_embedder    = FourierEmbedder(num_freqs=num_freqs,
                                                    include_pi=include_pi, input_dim=3)
        self.fourier_embedder_ID = FourierEmbedder(num_freqs=num_freqs,
                                                    include_pi=include_pi, input_dim=3)
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device, dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            fourier_embedder_ID=self.fourier_embedder_ID,
            num_latents=num_latents, point_feats=point_feats,
            width=width, heads=heads, layers=num_encoder_layers,
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
        print(f"\n  GS_DECODER: 40000 x 14 = 560,000 output dims")
        print(f"  Color activation: {'NONE (residuals)' if color_residual else 'clamp(0,1)'}")
        self.GS_decoder = GS_decoder(3, 1024, width * 512, [4], 40000 * 14,
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
    """
    Full Can3Tok VAE.

    KEY CHANGE vs previous version:
      AnchorPositionHead (shape_embed → anchors) REMOVED.
      AnchorPredFromTokens (transformer_tokens → anchors) ADDED inside decode().
      The predicted anchors are added to decoder position outputs directly,
      making the decoder output absolute positions.
      At second-stage diffusion inference, only z is needed — no GT scaffold data.
    """

    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False,
                 semantic_mode='none',
                 color_residual=False,
                 scene_semantic_head=False,
                 position_scaffold=False,
                 decoder_shape_prepend=False,
                 decoder_shape_cross_attn=False,
                 decoder_cross_attn_layers=4,
                 latent_disentangle=False,
                 semantic_dims=512,
                 scene_layout_head=False,
                 jepa_idea1=False,
                 decoder_pos_enc=False,
                 predict_seg_labels=False,
                 token_cond=False,
                 token_cond_approach='A',
                 query_decoder=False,
                 # ── NEW ablation flags ────────────────────────────────────────
                 decoder_fourier_pe=False,
                 token_cond_adaln=False,
                 semantic_token_heads=False):

        super().__init__(
            device=device, dtype=dtype, num_latents=1 + num_latents,
            point_feats=point_feats, embed_dim=embed_dim,
            num_freqs=num_freqs, include_pi=include_pi, width=width, heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_ln_post=use_ln_post, use_checkpoint=use_checkpoint,
            color_residual=color_residual)

        self.width                    = width
        self.semantic_mode            = semantic_mode
        self.color_residual           = color_residual
        self.scene_semantic_head_flag = scene_semantic_head
        self.position_scaffold        = position_scaffold
        self.decoder_shape_prepend    = decoder_shape_prepend
        self.decoder_shape_cross_attn = decoder_shape_cross_attn
        self.latent_disentangle       = latent_disentangle
        self.semantic_dims            = semantic_dims
        self._jepa_idea1_enabled      = jepa_idea1
        self.decoder_pos_enc_flag     = decoder_pos_enc
        self.predict_seg_labels_flag  = predict_seg_labels
        self.token_cond_flag          = token_cond
        self.token_cond_approach      = token_cond_approach.upper()
        self.query_decoder_flag       = query_decoder

        print(f"\n{'='*70}")
        print(f"  CAN3TOK (INFERENCE-FIXED)")
        print(f"  semantic='{semantic_mode}' | color_residual={color_residual}")
        print(f"  scene_semantic_head={scene_semantic_head} | position_scaffold={position_scaffold}")
        print(f"  latent_disentangle={latent_disentangle}  semantic_dims={semantic_dims}")
        print(f"  scene_layout_head={scene_layout_head}  jepa_idea1={jepa_idea1}")
        print(f"  ── INFERENCE FIX ──────────────────────────────────────────────")
        print(f"  AnchorPredFromTokens: transformer_tokens -> predicted_anchors")
        print(f"  DC added INSIDE decode() -> output is absolute positions")
        print(f"  No GT scaffold data needed at second-stage inference")
        print(f"  ── IDEAS ───────────────────────────────────────────────────────")
        print(f"  decoder_pos_enc:       {decoder_pos_enc}  (learnable PE)")
        print(f"  decoder_fourier_pe:    {decoder_fourier_pe}  (3D Fourier PE — takes priority)")
        print(f"  predict_seg_labels:    {predict_seg_labels}")
        print(f"  token_cond:            {token_cond}  approach={token_cond_approach}")
        print(f"  token_cond_adaln:      {token_cond_adaln}  (per-layer AdaLN-Zero)")
        print(f"  semantic_token_heads:  {semantic_token_heads}  (heads on z tokens, not shape_embed)")
        print(f"  query_decoder:         {query_decoder}")
        print(f"{'='*70}")

        # ── SHAPE_EMBED HEADS ────────────────────────────────────────────────
        # Input dimension depends on semantic_token_heads:
        #   False → shape_embed [B, width=384]   (encoder output, current behaviour)
        #   True  → z tokens:
        #             color:    z[:, :embed_dim]          [B, 32]  — token 0
        #             semantic: z[:, embed_dim:sem_dims].flatten [B, 480] — tokens 1-15
        # Gradient path is equivalent either way (reparameterisation trick preserves
        # unbiased gradients through the stochastic z sampling step).
        if semantic_token_heads and not latent_disentangle:
            raise ValueError("semantic_token_heads=True requires latent_disentangle=True")
        self.semantic_token_heads_flag = semantic_token_heads
        if semantic_token_heads:
            _color_in = embed_dim                    # 32 — single z token
            _sem_in   = semantic_dims - embed_dim    # 480 — 15 z tokens flattened
        else:
            _color_in = width                        # 384 — shape_embed (original)
            _sem_in   = width                        # 384 — shape_embed (original)

        self.mean_color_head      = None
        self.last_mean_color_pred = None
        if color_residual:
            self.mean_color_head = MeanColorHead(in_dim=_color_in)

        self.scene_semantic_module    = None
        self.last_scene_semantic_pred = None
        if scene_semantic_head:
            self.scene_semantic_module = SceneSemanticHead(in_dim=_sem_in)

        # NOTE: AnchorPositionHead (shape_embed -> scaffold_anchors) REMOVED.
        # Replaced by AnchorPredFromTokens (decoder tokens -> scaffold_anchors).
        # This ensures the anchor prediction is self-contained in the decode path.

        self.scene_layout_module    = None
        self.last_scene_layout_pred = None
        if scene_layout_head:
            self.scene_layout_module = SceneLayoutHead(in_dim=_sem_in)

        self.spatial_semantic_module = None
        if jepa_idea1:
            if not position_scaffold:
                print(f"  [WARNING] jepa_idea1=True requires position_scaffold=True. Disabled.")
            else:
                self.spatial_semantic_module = SpatialSemanticHead(width=width, num_tokens=512)

        # ── NEW: ANCHOR PREDICTION FROM DECODER TOKENS ───────────────────────
        # Replaces AnchorPositionHead. Runs inside decode() after transformer.
        # Predicted anchors are used as DC term for position recovery.
        # Self-contained: no GT data needed at inference.
        self.anchor_pred_from_tokens              = None
        self.last_predicted_anchors_from_tokens   = None
        if position_scaffold:
            self.anchor_pred_from_tokens = AnchorPredFromTokens(width=width)

        # ── LATENT DISENTANGLEMENT ───────────────────────────────────────────
        self._mu_s_cache = None
        self._mu_g_cache = None
        if latent_disentangle:
            assert embed_dim > 0
            assert semantic_dims % embed_dim == 0
            geom_dims = 64 * 64 * 4 - semantic_dims
            assert geom_dims > 0
            self.mu_s_proj_mean = nn.Linear(width, semantic_dims)
            self.mu_s_proj_var  = nn.Linear(width, semantic_dims)
            kl_in = (1 + num_latents - 1) * embed_dim
            self.kl_emb_proj_mean_g = nn.Linear(kl_in, geom_dims)
            self.kl_emb_proj_var_g  = nn.Linear(kl_in, geom_dims)
            print(f"  DISENTANGLE: mu_s[{semantic_dims}] from shape_embed | "
                  f"mu_g[{geom_dims}] from tokens")

        # ── DECODER SHAPE CONDITIONING (existing) ────────────────────────────
        self.project_shape_for_prepend = None
        if decoder_shape_prepend:
            self.project_shape_for_prepend = nn.Sequential(
                nn.Linear(width, width), nn.LayerNorm(width))

        self.project_shape_for_cross_attn = None
        self.shape_cross_attn_layers      = None
        if decoder_shape_cross_attn:
            self.project_shape_for_cross_attn = nn.Sequential(
                nn.Linear(width, width), nn.LayerNorm(width))
            self.shape_cross_attn_layers = nn.ModuleList([
                ResidualCrossAttentionBlock(
                    device=device, dtype=dtype, width=width, heads=heads,
                    init_scale=init_scale * math.sqrt(1.0 / width),
                    qkv_bias=qkv_bias, flash=flash)
                for _ in range(decoder_cross_attn_layers)
            ])

        # ── IDEA 0: POSITIONAL ENCODING (learnable) ──────────────────────────
        self.decoder_pos_emb = None
        if decoder_pos_enc:
            self.decoder_pos_emb = nn.Parameter(torch.zeros(512, width))
            nn.init.trunc_normal_(self.decoder_pos_emb, std=0.02)
            print(f"[DecoderPosEnc] learnable PE [512, {width}] — {512*width:,} params")

        # ── NEW: FOURIER DECODER PE (replaces learnable PE when True) ─────────
        # 3D Fourier features over the 8×8×8 voxel grid coordinate system.
        # Takes priority over learnable PE if both flags are True.
        self.decoder_fourier_pe_flag   = decoder_fourier_pe
        self.decoder_fourier_pe_module = None
        if decoder_fourier_pe:
            self.decoder_fourier_pe_module = FourierDecoderPE(
                fourier_embedder=self.fourier_embedder, width=width)
            if decoder_pos_enc:
                print(f"[FourierDecoderPE] Note: decoder_pos_enc=True is overridden by "
                      f"decoder_fourier_pe=True — only Fourier PE will be applied")

        # ── NEW: ADALN-ZERO CONDITIONED TRANSFORMER ───────────────────────────
        # Replaces the standard self.transformer inside decode() when
        # token_cond_adaln=True AND a valid TokenCond B signal is available.
        # Requires token_cond=True and token_cond_approach to include 'B'.
        self.token_cond_adaln_flag = False
        self.adaLN_transformer     = None
        _adaln_valid = (token_cond and 'B' in token_cond_approach.upper())
        if token_cond_adaln and not _adaln_valid:
            print(f"[WARNING] token_cond_adaln=True requires token_cond=True and "
                  f"approach includes 'B'. AdaLN disabled.")
        elif token_cond_adaln and _adaln_valid:
            self.token_cond_adaln_flag = True
            cond_dim = self.fourier_embedder.out_dim
            self.adaLN_transformer = AdaLNTransformerDecoder(
                width=width, heads=heads, layers=num_decoder_layers, cond_dim=cond_dim)

        # ── IDEA 1: SEGMENT PREDICTION HEAD ──────────────────────────────────
        self.seg_pred_head = None
        self.last_seg_pred = None
        if predict_seg_labels:
            self.seg_pred_head = SegPredHead(in_dim=14, num_cats=72)

        # ── IDEA 2: TOKEN CENTROID CONDITIONING ───────────────────────────────
        self.token_cond_mlp_A = None
        self.token_cond_mlp_B = None
        self.token_cat_assign = None
        fourier_out_dim = self.fourier_embedder.out_dim

        if token_cond:
            print(f"[TokenCond] approach='{token_cond_approach}' | fourier_dim={fourier_out_dim}")
            if 'A' in self.token_cond_approach:
                self.token_cond_mlp_A = TokenCondMLP(fourier_out_dim, width)
                print(f"  Approach A: Fourier(scaffold_anchor[B,512,3]) -> bias[B,512,{width}]")
            if 'B' in self.token_cond_approach:
                self.token_cat_assign = nn.Parameter(torch.zeros(512, 72))
                nn.init.trunc_normal_(self.token_cat_assign, std=0.01)
                self.token_cond_mlp_B = TokenCondMLP(fourier_out_dim, width)
                print(f"  Approach B: W[512,72] x pred_centroids -> Fourier -> bias")

        # ── IDEA 3: QUERY-BASED SPATIAL DECODER ───────────────────────────────
        self.spatial_aware_decoder = None
        if query_decoder:
            self.spatial_aware_decoder = SpatialAwareDecoder(
                token_dim=width, fourier_embedder=self.fourier_embedder,
                hidden_dim=256, color_residual=color_residual)

        # ── PER-GAUSSIAN INFONCE HEADS ────────────────────────────────────────
        self.semantic_projection_hidden    = None
        self.semantic_projection_geometric = None
        self.semantic_attention_head       = None
        self.semantic_distribution_head    = None
        if semantic_mode == 'hidden':
            self.semantic_projection_hidden = SemanticProjectionHead(1024, 40000, 32)
        elif semantic_mode == 'geometric':
            self.semantic_projection_geometric = SemanticProjectionHeadGeometric(14, 40000, 32, 128)
        elif semantic_mode == 'attention':
            self.semantic_attention_head = GaussianSemanticAttentionHead(
                device=device, dtype=dtype, num_latents=num_latents, out_channels=32,
                fourier_embedder=self.fourier_embedder, width=width, heads=heads,
                init_scale=init_scale * math.sqrt(1.0 / width),
                qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint)
        elif semantic_mode == 'dist':
            self.semantic_distribution_head = SemanticDistributionHead(1024, 72)
        elif semantic_mode != 'none':
            raise ValueError(f"Unknown semantic_mode: '{semantic_mode}'")

        print(f"{'='*70}\n")

    # ── Encode helpers ────────────────────────────────────────────────────────

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

    def decode(self, latents, volume_queries=None, return_semantic_features=False,
               shape_embed=None,
               scaffold_anchors=None,
               scaffold_token_ids=None):
        """
        Decode [B, 512, 32] -> Gaussian attributes [B, 560000].

        INFERENCE FIX:
          AnchorPredFromTokens runs after the transformer on [B,512,width] tokens.
          Predicted anchors [B,512,3] are added to decoder position outputs as DC term.
          Result: output positions are ABSOLUTE (offset + DC), no GT data needed.

          scaffold_token_ids:
            Not None (training)  → GT spatial assignment used for accurate DC
            None     (inference) → fixed assignment j → j*512//40000 used

        Args:
            latents:            [B, 512, 32]
            scaffold_anchors:   [B, 512, 3]  — for Idea 2A token conditioning (training)
            scaffold_token_ids: [B, 40000]   — for accurate DC at training time
        """
        latents = self.post_kl(latents)   # [B, 512, width]

        # ── Shape cross-attention (Idea existing) ────────────────────────────
        if (self.decoder_shape_cross_attn and
                self.shape_cross_attn_layers is not None and
                shape_embed is not None):
            shape_context = self.project_shape_for_cross_attn(shape_embed).unsqueeze(1)
            for cross_attn_layer in self.shape_cross_attn_layers:
                latents = cross_attn_layer(latents, shape_context)

        # ── Idea 0: Positional Encoding ──────────────────────────────────────
        # Fourier PE takes priority over learnable PE when both flags are set.
        if self.decoder_fourier_pe_flag and self.decoder_fourier_pe_module is not None:
            latents = latents + self.decoder_fourier_pe_module(latents.shape[0], latents.device)
        elif self.decoder_pos_emb is not None:
            latents = latents + self.decoder_pos_emb.unsqueeze(0)

        # ── Idea 2: Token Centroid Conditioning ──────────────────────────────
        # Compute per-token semantic centroid signal (used for additive bias OR AdaLN).
        cond_for_adaln = None   # [B, 512, fourier_dim] — set when token_cond_adaln=True
        if self.token_cond_flag:
            if ('A' in self.token_cond_approach and
                    self.token_cond_mlp_A is not None and
                    scaffold_anchors is not None):
                # Approach A: additive scaffold anchor bias (unchanged)
                fourier_A = self.fourier_embedder(scaffold_anchors)
                spatial_A = self.token_cond_mlp_A(fourier_A)
                latents   = latents + spatial_A

            if ('B' in self.token_cond_approach and
                    self.token_cat_assign is not None and
                    self.last_scene_layout_pred is not None):
                # Compute per-token semantic centroid (shared by both additive and AdaLN paths)
                W = F.softmax(self.token_cat_assign, dim=-1)            # [512, 72]
                pred_c = self.last_scene_layout_pred                    # [B, 72, 3]
                token_centroids = torch.einsum('tk,bkd->btd', W, pred_c)  # [B, 512, 3]
                fourier_B = self.fourier_embedder(token_centroids)         # [B, 512, fourier_dim]

                if self.token_cond_adaln_flag:
                    # AdaLN path: pass Fourier features as conditioning to each layer
                    # (do NOT add as additive bias — AdaLN handles conditioning inside transformer)
                    cond_for_adaln = fourier_B
                elif self.token_cond_mlp_B is not None:
                    # Original additive path: project to width and add as bias
                    spatial_B = self.token_cond_mlp_B(fourier_B)
                    latents   = latents + spatial_B

        # ── Shape prepend (Idea existing) ────────────────────────────────────
        shape_token_prepended = False
        if (self.decoder_shape_prepend and
                self.project_shape_for_prepend is not None and
                shape_embed is not None):
            shape_token = self.project_shape_for_prepend(shape_embed).unsqueeze(1)
            latents = torch.cat([shape_token, latents], dim=1)
            shape_token_prepended = True

        # ── Transformer ───────────────────────────────────────────────────────
        # Use AdaLN-conditioned transformer when: flag is True AND conditioning is ready.
        # Falls back to standard transformer if conditioning signal is absent
        # (e.g. scene_layout_head=False or early training before layout converges).
        if (self.token_cond_adaln_flag and
                self.adaLN_transformer is not None and
                cond_for_adaln is not None):
            latents_out = self.adaLN_transformer(latents, cond_for_adaln)
        else:
            latents_out = self.transformer(latents)
        if shape_token_prepended:
            latents_out = latents_out[:, 1:, :]

        # ── AnchorPredFromTokens: predict scaffold anchors from decoder tokens
        # RUNS HERE: after transformer, before GS_decoder.
        # Gradient path: L_anchor → this head → transformer → post_kl → z → encoder
        self.last_predicted_anchors_from_tokens = None
        pred_anchors = None
        if self.anchor_pred_from_tokens is not None:
            pred_anchors = self.anchor_pred_from_tokens(latents_out)  # [B, 512, 3]
            self.last_predicted_anchors_from_tokens = pred_anchors

        # ── Idea 3: Spatial-Aware Decoder ────────────────────────────────────
        use_query_decoder = (
            self.query_decoder_flag and
            self.spatial_aware_decoder is not None and
            scaffold_anchors is not None and
            scaffold_token_ids is not None)

        has_sem = any([self.semantic_projection_hidden,
                       self.semantic_projection_geometric,
                       self.semantic_attention_head,
                       self.semantic_distribution_head])
        need_hidden = (return_semantic_features and self.training and
                       has_sem and not use_query_decoder)

        hidden = None
        if use_query_decoder:
            reconstruction = self.spatial_aware_decoder(
                latents_out, scaffold_anchors, scaffold_token_ids)
        else:
            latents_flat = latents_out.reshape(latents_out.shape[0], -1)
            if need_hidden:
                reconstruction, hidden = self.GS_decoder(latents_flat, return_hidden=True)
            else:
                reconstruction = self.GS_decoder(latents_flat, return_hidden=False)

        # ── ADD DC TERM TO POSITION OUTPUT ────────────────────────────────────
        # This is the INFERENCE FIX.
        # GS_decoder output positions are small residuals (learned offsets).
        # We add the predicted voxel anchor (DC) to produce absolute positions.
        # No GT data involved — pred_anchors comes from AnchorPredFromTokens.
        #
        # Training  (scaffold_token_ids available): use accurate GT spatial assignment
        # Inference (scaffold_token_ids is None):   use fixed j → j*512//40000
        if pred_anchors is not None:
            B_r = reconstruction.shape[0]
            pred_3d = reconstruction.reshape(B_r, 40_000, 14)

            if scaffold_token_ids is not None:
                # Training path: GT spatial assignment → accurate DC
                # Each Gaussian j gets the anchor of its actual spatial voxel
                tids = scaffold_token_ids.long()   # [B, 40000]
                # Gather: for each scene b and Gaussian j, pick pred_anchors[b, tids[b,j], :]
                idx_3d = tids.unsqueeze(-1).expand(-1, -1, 3)  # [B, 40000, 3]
                dc = torch.gather(pred_anchors, 1, idx_3d)     # [B, 40000, 3]
            else:
                # Inference path: fixed assignment, no GT needed
                # All runs of this branch produce consistent results from z alone
                fixed_ids = FIXED_TOKEN_IDS.to(latents_out.device)   # [40000]
                dc = pred_anchors[:, fixed_ids, :]                     # [B, 40000, 3]

            pred_3d[:, :, 0:3] = pred_3d[:, :, 0:3] + dc
            reconstruction = pred_3d.reshape(B_r, -1)

        # ── Idea 1: Segment Prediction ────────────────────────────────────────
        self.last_seg_pred = None
        if self.seg_pred_head is not None:
            B_r = reconstruction.shape[0]
            g_params = reconstruction.reshape(B_r, 40000, 14)
            self.last_seg_pred = self.seg_pred_head(g_params)

        # ── Semantic features for InfoNCE ─────────────────────────────────────
        semantic_features = None
        if return_semantic_features and self.training and has_sem and hidden is not None:
            B       = reconstruction.shape[0]
            recon_g = reconstruction.reshape(B, 40000, 14)
            if self.semantic_mode == 'hidden':
                semantic_features = self.semantic_projection_hidden(hidden)
            elif self.semantic_mode == 'geometric':
                semantic_features = self.semantic_projection_geometric(recon_g)
            elif self.semantic_mode == 'attention':
                semantic_features = self.semantic_attention_head(
                    recon_g[:, :, 0:3], latents_out)
            elif self.semantic_mode == 'dist':
                semantic_features = self.semantic_distribution_head(hidden)

        return reconstruction, semantic_features

    def forward(self, pc, feats, volume_queries, sample_posterior=True,
                scaffold_anchors=None, scaffold_token_ids=None):
        """
        6-value return (unchanged API):
          shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features

        UV_gs_recover positions are now ABSOLUTE (DC added inside decode()).
        No post-processing needed to recover absolute positions.

        scaffold_token_ids: pass from batch during training for accurate DC.
                            pass None at second-stage inference (fixed assignment used).
        """
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)
        _se = self._shape_embed_cache

        # ── Prediction heads: shape_embed OR z semantic tokens ────────────────
        # semantic_token_heads=False (default): heads run on shape_embed [B, width]
        #   → original behaviour; shape_embed is an encoder quantity.
        #
        # semantic_token_heads=True (new): heads run on z token subsets BEFORE decode.
        #   Token 0  (z[:, :embed_dim])               → MeanColorHead  → color
        #   Tokens 1-15 (z[:, embed_dim:sem_dims])     → SemanticHead, LayoutHead
        #
        #   WHY THIS SOLVES THE SECOND-STAGE INFERENCE PROBLEM:
        #   shape_embed only exists during VAE encoding; the second-stage DiT
        #   generates z directly and has no access to shape_embed.
        #   With heads on z tokens (Option A — pre-decoder-transformer), the full
        #   pipeline at inference is: DiT → z → extract tokens 0-15 → run heads →
        #   get conditioning → decode. No encoder needed. Fully self-contained.
        #
        #   GRADIENT PATH: L_color/sem/layout → head → z_s → reparameterisation →
        #   mu_s → encoder. Reparameterisation trick guarantees unbiased gradients.
        if self.semantic_token_heads_flag:
            _ed = self.embed_dim          # 32: single token dimension
            _sd = self.semantic_dims      # 512: semantic subspace
            z_color = z[:, :_ed]                          # [B, 32]  — token 0
            z_sem   = z[:, _ed:_sd]                       # [B, 480] — tokens 1-15
            self.last_mean_color_pred = (
                self.mean_color_head(z_color)
                if self.mean_color_head is not None else None)
            self.last_scene_semantic_pred = (
                self.scene_semantic_module(z_sem)
                if self.scene_semantic_module is not None else None)
            self.last_scene_layout_pred = (
                self.scene_layout_module(z_sem)
                if self.scene_layout_module is not None else None)
        else:
            # Original path: layout must be set BEFORE decode so TokenCond B works
            self.last_scene_layout_pred = (
                self.scene_layout_module(_se)
                if self.scene_layout_module is not None else None)
            # Color and semantic set after decode below

        latents = z.reshape(z.shape[0], 512, 32)
        UV_gs_recover, per_gaussian_features = self.decode(
            latents, volume_queries,
            return_semantic_features=self.training,
            shape_embed=_se,
            scaffold_anchors=scaffold_anchors,
            scaffold_token_ids=scaffold_token_ids)

        # Shape_embed auxiliary heads (only when NOT using semantic_token_heads)
        if not self.semantic_token_heads_flag:
            self.last_mean_color_pred = (
                self.mean_color_head(_se) if self.mean_color_head is not None else None)
            self.last_scene_semantic_pred = (
                self.scene_semantic_module(_se)
                if self.scene_semantic_module is not None else None)
        # NOTE: last_predicted_anchors_from_tokens is set inside decode() above.
        self.last_spatial_semantic_pred = None

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features