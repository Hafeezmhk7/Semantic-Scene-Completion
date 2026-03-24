# -*- coding: utf-8 -*-
"""
sal_perceiver.py  —  Can3Tok VAE
==================================
Step 1:    Color Residual         (color_residual)
Move 1:    Scene Semantic Head    (scene_semantic_head)
Scaffold:  Position Scaffold      (position_scaffold)
Option 1:  Decoder Shape Prepend  (decoder_shape_prepend)   ← NEW
Option 2:  Decoder Shape Cross-Attn (decoder_shape_cross_attn) ← NEW

═══════════════════════════════════════════════════════════════
GRADIENT PATHS INTO shape_embed (encoder token 0)
═══════════════════════════════════════════════════════════════
  Head 1  MeanColorHead:       MSE(pred_mean_color, gt)       [3 numbers]
  Head 2  SceneSemanticHead:   KL(p_s ‖ p̂)                  [72 numbers]
  Head 3  AnchorPositionHead:  MSE(pred_anchors, gt)          [512×3 numbers]
  Option1/2 decoder conditioning: reconstruction loss        [indirect]

All paths share the same shape_embed token but are fully independent.

═══════════════════════════════════════════════════════════════
OPTION 1 — DECODER SHAPE PREPEND  (decoder_shape_prepend=True)
═══════════════════════════════════════════════════════════════
Mirrors the encoder's own structure: the encoder produces shape_embed as
token 0 alongside 512 geometry tokens. The decoder now receives shape_embed
as token 0 alongside the 512 latent tokens, making the information flow
symmetric.

Architecture:
  shape_embed [B, 384]
       ↓  project_shape_for_prepend (Linear + LayerNorm)
  shape_token [B, 1, 384]
       ↓  concatenate with geometry tokens
  [shape_token | latent_tokens]  [B, 513, 384]
       ↓  Transformer (n_ctx=513 — already correct!)
  [shape_out | latent_out]  [B, 513, 384]
       ↓  drop token 0
  latent_out  [B, 512, 384]
       ↓  GS_decoder MLP
  Gaussians [B, 40000, 14]

WHY n_ctx=513 is already correct:
  ShapeAsLatentPerceiver is initialized with num_latents=1+512=513.
  The transformer was always built for 513 tokens.
  Currently only 512 are fed (shape_embed was dropped after encoding).
  Prepending restores the intended full sequence length.

STAGE 2 IMPLICATION:
  At generation time shape_embed must be provided to the decoder.
  A lightweight ShapeEmbedPredictor MLP (z → shape_embed) trained on the
  VAE's training data can approximate it from the generated z tokens.
  The diffusion model can also generate it jointly as an extra token.

═══════════════════════════════════════════════════════════════
OPTION 2 — DECODER SHAPE CROSS-ATTENTION  (decoder_shape_cross_attn=True)
═══════════════════════════════════════════════════════════════
Applies K cross-attention blocks BEFORE the main transformer so that all
12 self-attention layers then refine shape-conditioned representations.
Each geometry token independently decides how much global context to incorporate
before the transformer propagates that context across the full sequence.

Architecture:
  post_kl(z) → latent_tokens [B, 512, 384]
       ↓  project_shape_for_cross_attn (Linear + LayerNorm)
  shape_context [B, 1, 384]
       ↓  for each of decoder_cross_attn_layers:
            latent_tokens = ResidualCrossAttentionBlock(
                queries=latent_tokens,   [B, 512, 384]
                data=shape_context       [B, 1,   384]
            )
       ↓  Transformer (12 self-attn layers on shape-conditioned tokens)
  latent_out [B, 512, 384]
       ↓  GS_decoder MLP
  Gaussians [B, 40000, 14]

WHY BEFORE (not after) the transformer:
  Each geometry token attends to shape_embed before self-attention.
  The transformer then propagates this global context across all 512 tokens
  through 12 layers. Information flows from shape_embed → individual tokens
  → whole sequence, rather than only reaching tokens at the final layer.

  Mathematically: for each geometry token k,
    attn_k = softmax(q_k · k_shape / √d) · v_shape
  Since shape_context is a single token, this simplifies to a learned gate:
  each token independently controls how much global scene context to incorporate.

OPTIONS 1 AND 2 CAN BE COMBINED:
  Both prepend and cross-attention can be enabled simultaneously.
  In that case: cross-attn conditions tokens first, then prepend adds the
  shape token for additional full-sequence interaction in the transformer.
  Use this combination cautiously as it may risk over-relying on shape_embed.

STAGE 2 IMPLICATION:
  Same as Option 1 — shape_embed must be provided at decoder time.
  A ShapeEmbedPredictor approximates it from generated z tokens.
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
# SHAPE_EMBED AUXILIARY HEADS (gradient paths into encoder token 0)
# ============================================================================

class MeanColorHead(nn.Module):
    """Step 1: shape_embed → mean scene RGB. First gradient path to encoder token 0."""
    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid())
        total = sum(p.numel() for p in self.parameters())
        print(f"[MeanColorHead] shape_embed [B,{width}] -> Linear(64) -> [B,3] sigmoid "
              f"| {total:,} params")

    def forward(self, shape_embed):
        return self.head(shape_embed)   # [B, 3]


class SceneSemanticHead(nn.Module):
    """Move 1: shape_embed → ScanNet72 label distribution. 2nd gradient path."""
    NUM_LABELS = 72

    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),  nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, self.NUM_LABELS))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SceneSemanticHead] shape_embed [B,{width}] -> MLP(128->128) "
              f"-> [{self.NUM_LABELS}] softmax | {total:,} params")

    def forward(self, shape_embed):
        return F.softmax(self.head(shape_embed), dim=-1)   # [B, 72]


class AnchorPositionHead(nn.Module):
    """
    Scaffold-GS inspired: shape_embed → 512 spatial anchor positions.
    3rd gradient path into encoder token 0.
    Enables decoder to predict position offsets δp_i instead of absolute xyz.
    """
    NUM_TOKENS = 512

    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),   nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, self.NUM_TOKENS * 3))
        total = sum(p.numel() for p in self.parameters())
        print(f"[AnchorPositionHead] shape_embed [B,{width}] -> MLP(512->512) "
              f"-> [B,{self.NUM_TOKENS},3] | {total:,} params")

    def forward(self, shape_embed):
        B = shape_embed.shape[0]
        return self.head(shape_embed).reshape(B, self.NUM_TOKENS, 3)   # [B, 512, 3]


# ============================================================================
# PER-GAUSSIAN SEMANTIC HEADS (InfoNCE, decoder hidden path)
# ============================================================================

class SemanticProjectionHead(nn.Module):
    """InfoNCE on decoder hidden state. [B,1024] → [B, 40k, 32] L2-norm."""
    def __init__(self, hidden_dim=1024, num_gaussians=40000, feature_dim=32):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim   = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, num_gaussians * feature_dim))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHead] [B,{hidden_dim}] -> [B,{num_gaussians},"
              f"{feature_dim}] L2-norm | {total/1e6:.3f}M params")

    def forward(self, hidden):
        B = hidden.shape[0]
        return F.normalize(
            self.projection(hidden).reshape(B, self.num_gaussians, self.feature_dim),
            p=2, dim=-1)


class SemanticDistributionHead(nn.Module):
    """Label Distribution Learning. [B,1024] → [B,72] logits."""
    def __init__(self, hidden_dim=1024, num_labels=72):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, num_labels))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticDistributionHead] [B,{hidden_dim}] -> [B,{num_labels}] "
              f"logits | {total/1e6:.3f}M params")

    def forward(self, hidden):
        return self.head(hidden)


class SemanticProjectionHeadGeometric(nn.Module):
    """InfoNCE on reconstructed Gaussian params. [B, N, 14] → [B, N, 32]."""
    def __init__(self, gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.projection = nn.Sequential(
            nn.Linear(gaussian_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHeadGeometric] [B,{num_gaussians},{gaussian_dim}] "
              f"-> [B,{num_gaussians},{feature_dim}] | {total/1e6:.3f}M params")

    def forward(self, gaussians):
        B, N, D  = gaussians.shape
        return F.normalize(
            self.projection(gaussians.reshape(B * N, D)).reshape(B, N, -1),
            p=2, dim=-1)


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
        xyz_actual      = pc[:, :, 4:7]    # ALWAYS absolute — Fourier embedder
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
# GEOMETRY DECODER (cross-attention, used for geometry queries)
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
# GS DECODER MLP — 14-param Gaussian output
# ============================================================================

class GS_decoder(nn.Module):
    """
    MLP: latent [B, 512*384] → Gaussian attributes [B, 40000, 14].

    Position [0:3]: absolute xyz (scaffold=False) or offsets δp_i (scaffold=True).
    Color    [3:6]: clamped [0,1] (color_residual=False) or unbounded (True).
    Opacity  [6]:   sigmoid.
    Scale    [7:10]: exp.
    Quat     [10:14]: L2 normalized.
    """
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
        # NOTE: n_ctx=num_latents=513 (1+512). When Option 1 prepend is on,
        # exactly 513 tokens are fed; otherwise 512 tokens are fed (which also
        # works since the transformer doesn't enforce strict sequence length).
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
    Full Can3Tok VAE with all objectives and decoder shape conditioning.

    DECODER CONDITIONING OPTIONS
    ─────────────────────────────
    Option 1  decoder_shape_prepend=True:
      shape_embed projected to width=384 and prepended as token 0 to the
      512 geometry tokens before the transformer. Transformer processes all
      513 tokens (matches n_ctx=513 exactly). Shape token dropped afterwards.

    Option 2  decoder_shape_cross_attn=True:
      decoder_cross_attn_layers cross-attention blocks condition the 512
      geometry tokens on shape_embed BEFORE the main transformer runs.
      Each geometry token attends to shape_embed independently.
      The transformer then propagates this global context through all 12 layers.

    Options can be combined: cross-attn first, then prepend-in-transformer.

    STAGE 2 COMPATIBILITY
    ─────────────────────
    shape_embed [B, 384] is the pre-KL encoder output (not through the KL
    bottleneck). At inference, it must be approximated from the generated z
    tokens via a small ShapeEmbedPredictor MLP trained separately, or
    generated jointly by the diffusion model as an extra 384-dim token.

    SHAPE_EMBED HEADS (all use pre-KL shape_embed [B, 384])
    ─────────────────────────────────────────────────────────
      self.last_mean_color_pred     [B, 3]
      self.last_scene_semantic_pred [B, 72]
      self.last_anchor_pred         [B, 512, 3]
    Read by training loop after forward().
    """

    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False,
                 semantic_mode='none',
                 color_residual=False,
                 scene_semantic_head=False,
                 position_scaffold=False,
                 decoder_shape_prepend=False,       # Option 1
                 decoder_shape_cross_attn=False,    # Option 2
                 decoder_cross_attn_layers=4):      # Option 2 depth

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
        self.scene_semantic_head     = scene_semantic_head
        self.position_scaffold       = position_scaffold
        self.decoder_shape_prepend   = decoder_shape_prepend
        self.decoder_shape_cross_attn = decoder_shape_cross_attn

        print(f"\n{'='*70}")
        print(f"  CAN3TOK")
        print(f"  semantic='{semantic_mode}' | color_residual={color_residual}")
        print(f"  scene_semantic_head={scene_semantic_head} | position_scaffold={position_scaffold}")
        print(f"  decoder_shape_prepend={decoder_shape_prepend} | "
              f"decoder_shape_cross_attn={decoder_shape_cross_attn}")
        print(f"{'='*70}")

        # ── Auxiliary heads on shape_embed ────────────────────────────────────
        self.mean_color_head      = None
        self.last_mean_color_pred = None
        if color_residual:
            self.mean_color_head = MeanColorHead(width=width)
            print(f"  Step 1 ENABLED: MeanColorHead -> [B,3]")

        self.scene_semantic_module    = None
        self.last_scene_semantic_pred = None
        if scene_semantic_head:
            self.scene_semantic_module = SceneSemanticHead(width=width)
            print(f"  Move 1 ENABLED: SceneSemanticHead -> [B,72]")

        self.anchor_position_head = None
        self.last_anchor_pred     = None
        if position_scaffold:
            self.anchor_position_head = AnchorPositionHead(width=width)
            print(f"  Scaffold ENABLED: AnchorPositionHead -> [B,512,3]")

        # ── Option 1: Decoder shape prepend ──────────────────────────────────
        # Projects shape_embed [B, width] to a decoder-compatible token [B, 1, width].
        # Uses Linear + LayerNorm for stable initialisation.
        # The projected token is prepended to the 512 geometry tokens before
        # the transformer runs. n_ctx=513 in the transformer exactly accommodates this.
        # After the transformer, token 0 is dropped — it served as a global
        # context carrier for the geometry tokens during self-attention.
        self.project_shape_for_prepend = None
        if decoder_shape_prepend:
            self.project_shape_for_prepend = nn.Sequential(
                nn.Linear(width, width),
                nn.LayerNorm(width),
            )
            total = sum(p.numel() for p in self.project_shape_for_prepend.parameters())
            print(f"  Option 1 ENABLED: shape_embed → project_shape_for_prepend "
                  f"→ prepend token 0 in decoder transformer | {total:,} params")
            print(f"    Transformer sees [shape_token | 512 geometry tokens] = 513 tokens "
                  f"(n_ctx=513 ✓)")

        # ── Option 2: Decoder shape cross-attention ───────────────────────────
        # Applies K cross-attention blocks BEFORE the main transformer so that
        # all 12 self-attention layers process shape-conditioned tokens.
        # Each geometry token attends to shape_embed as a single key/value.
        # A separate projection adapts shape_embed to the decoder's representation space.
        self.project_shape_for_cross_attn = None
        self.shape_cross_attn_layers      = None
        if decoder_shape_cross_attn:
            # Project shape_embed to decoder space (same width, but separate weights
            # allow the model to learn a different representation for decoder use)
            self.project_shape_for_cross_attn = nn.Sequential(
                nn.Linear(width, width),
                nn.LayerNorm(width),
            )
            # K cross-attention blocks: geometry tokens (queries) attend to
            # shape_embed context (key/value). ResidualCrossAttentionBlock adds
            # the attended output back to the input (residual connection).
            self.shape_cross_attn_layers = nn.ModuleList([
                ResidualCrossAttentionBlock(
                    device=device, dtype=dtype, width=width, heads=heads,
                    init_scale=init_scale * math.sqrt(1.0 / width),
                    qkv_bias=qkv_bias, flash=flash)
                for _ in range(decoder_cross_attn_layers)
            ])
            proj_params  = sum(p.numel() for p in self.project_shape_for_cross_attn.parameters())
            attn_params  = sum(p.numel() for p in self.shape_cross_attn_layers.parameters())
            print(f"  Option 2 ENABLED: {decoder_cross_attn_layers} cross-attn layers "
                  f"BEFORE transformer | {(proj_params+attn_params)/1e6:.3f}M params")
            print(f"    geometry_tokens attend to shape_embed before self-attention")
            print(f"    shape context propagates through all {num_decoder_layers} "
                  f"transformer layers")

        # ── Per-Gaussian InfoNCE heads ─────────────────────────────────────────
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

        if semantic_mode != 'none':
            print(f"  InfoNCE ENABLED: mode='{semantic_mode}'")
        print(f"{'='*70}\n")

    # ── Encode helpers ────────────────────────────────────────────────────────

    def encode_latents(self, pc, feats=None):
        """Encoder output: shape_embed [B,width] (token 0) + latents [B,512,width]."""
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
        """
        Full encode pass. Caches shape_embed in self._shape_embed_cache so
        forward() can pass it to decode() without a second encoder call.
        """
        shape_embed, latents    = self.encode_latents(pc, feats)
        self._shape_embed_cache = shape_embed          # [B, width], pre-KL
        kl_embed, posterior     = self.encode_kl_embed(latents, sample_posterior)
        kl_flat = kl_embed.reshape([kl_embed.shape[0], -1])
        mu      = self.kl_emb_proj_mean(kl_flat)
        log_var = self.kl_emb_proj_var(kl_flat)
        z       = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return shape_embed, mu, log_var, z, posterior

    def decode(self, latents, volume_queries=None, return_semantic_features=False,
               shape_embed=None):
        """
        Decode [B, 512, 32] latent tokens → Gaussian attributes.

        shape_embed [B, width] is optionally used to condition the decoder
        via Option 1 (prepend) and/or Option 2 (cross-attention).

        ─────────────────────────────────────────────────────
        STEP 1 — post_kl projection
          z [B, 512, 32] → post_kl → [B, 512, 384]

        STEP 2 — Option 2: Cross-attention conditioning (BEFORE transformer)
          If decoder_shape_cross_attn and shape_embed provided:
            shape_context = project_shape(shape_embed)  [B, 1, 384]
            for each cross_attn_layer (K layers):
              latents = cross_attn_layer(queries=latents, data=shape_context)
          Result: geometry tokens conditioned on global scene context

        STEP 3 — Option 1: Prepend shape token (IN transformer)
          If decoder_shape_prepend and shape_embed provided:
            shape_token = project_shape_for_prepend(shape_embed)  [B, 1, 384]
            latents = cat([shape_token, latents], dim=1)          [B, 513, 384]
          Run transformer (n_ctx=513 — exact match)
          Drop token 0: latents = latents[:, 1:, :]               [B, 512, 384]
          else (no prepend):
            Run transformer on 512 tokens directly

        STEP 4 — GS_decoder MLP → [B, 40000, 14]
        ─────────────────────────────────────────────────────
        """
        # STEP 1: KL projection
        latents             = self.post_kl(latents)    # [B, 512, 384]

        # STEP 2: Option 2 — Cross-attention conditioning before transformer
        # Applied first so all transformer layers process shape-conditioned tokens.
        # Each geometry token independently attends to the single shape context token.
        if (self.decoder_shape_cross_attn and
                self.shape_cross_attn_layers is not None and
                shape_embed is not None):
            # Project shape_embed to decoder space [B, 1, width]
            shape_context = self.project_shape_for_cross_attn(shape_embed).unsqueeze(1)
            for cross_attn_layer in self.shape_cross_attn_layers:
                # geometry tokens (queries) attend to shape_context (key/value)
                # ResidualCrossAttentionBlock: output = input + attn(LN(input), data)
                latents = cross_attn_layer(latents, shape_context)

        # STEP 3: Option 1 — Prepend shape token for full transformer self-attention
        # After cross-attn conditioning (if any), add shape token to sequence.
        # The transformer sees all tokens simultaneously — geometry tokens can
        # directly attend to the shape token through self-attention.
        shape_token_prepended = False
        if (self.decoder_shape_prepend and
                self.project_shape_for_prepend is not None and
                shape_embed is not None):
            shape_token = self.project_shape_for_prepend(shape_embed).unsqueeze(1)  # [B,1,384]
            latents = torch.cat([shape_token, latents], dim=1)   # [B, 513, 384]
            shape_token_prepended = True

        # Run main transformer (12 self-attention layers)
        # With prepend: 513 tokens (matches n_ctx=513 exactly)
        # Without prepend: 512 tokens (also works, uses first 512 of n_ctx=513)
        latents_transformed = self.transformer(latents)

        # Drop the prepended shape token — it served its purpose in the transformer
        if shape_token_prepended:
            latents_transformed = latents_transformed[:, 1:, :]   # [B, 512, 384]

        # STEP 4: GS_decoder MLP
        latents_flat = latents_transformed.reshape(latents_transformed.shape[0], -1)

        has_sem = any([self.semantic_projection_hidden,
                       self.semantic_projection_geometric,
                       self.semantic_attention_head,
                       self.semantic_distribution_head])
        need_hidden = return_semantic_features and self.training and has_sem

        if need_hidden:
            reconstruction, hidden = self.GS_decoder(latents_flat, return_hidden=True)
        else:
            reconstruction = self.GS_decoder(latents_flat, return_hidden=False)

        semantic_features = None
        if return_semantic_features and self.training and has_sem:
            B       = reconstruction.shape[0]
            recon_g = reconstruction.reshape(B, 40000, 14)
            if self.semantic_mode == 'hidden':
                semantic_features = self.semantic_projection_hidden(hidden)
            elif self.semantic_mode == 'geometric':
                semantic_features = self.semantic_projection_geometric(recon_g)
            elif self.semantic_mode == 'attention':
                semantic_features = self.semantic_attention_head(
                    recon_g[:, :, 0:3], latents_transformed)
            elif self.semantic_mode == 'dist':
                semantic_features = self.semantic_distribution_head(hidden)

        return reconstruction, semantic_features

    def forward(self, pc, feats, volume_queries, sample_posterior=True):
        """
        6-value return (unchanged for asl_pl_module.py compatibility):
          shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features

        shape_embed [B, 384] is passed to decode() to condition the decoder
        via Option 1 (prepend) and/or Option 2 (cross-attention).

        Auxiliary head predictions (read by training loop after forward()):
          self.last_mean_color_pred     [B, 3]      or None
          self.last_scene_semantic_pred [B, 72]     or None
          self.last_anchor_pred         [B, 512, 3] or None
        """
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)

        # Decode with shape conditioning.
        # _shape_embed_cache = pre-KL shape_embed [B, 384] set by encode().
        # Passed to decode() so Option 1 and Option 2 can use it to condition
        # the decoder transformer without an extra encoder call.
        latents = z.reshape(z.shape[0], 512, 32)
        UV_gs_recover, per_gaussian_features = self.decode(
            latents,
            volume_queries,
            return_semantic_features=self.training,
            shape_embed=self._shape_embed_cache,    # ← conditioning signal
        )

        # Apply all auxiliary heads to shape_embed
        _se = self._shape_embed_cache   # [B, width]

        self.last_mean_color_pred = (
            self.mean_color_head(_se) if self.mean_color_head is not None else None)

        self.last_scene_semantic_pred = (
            self.scene_semantic_module(_se)
            if self.scene_semantic_module is not None else None)

        self.last_anchor_pred = (
            self.anchor_position_head(_se)
            if self.anchor_position_head is not None else None)

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features