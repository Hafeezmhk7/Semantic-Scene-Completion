# -*- coding: utf-8 -*-
"""
sal_perceiver.py  —  Can3Tok VAE
==================================
Step 1:    Color Residual             (color_residual)
Move 1:    Scene Semantic Head        (scene_semantic_head)
Scaffold:  Position Scaffold          (position_scaffold)
Option 1:  Decoder Shape Prepend      (decoder_shape_prepend)
Option 2:  Decoder Shape Cross-Attn   (decoder_shape_cross_attn)

DISENTANGLEMENT SUITE (NEW):
  latent_disentangle:  mu_s from shape_embed | mu_g from tokens
  scene_layout_head:   shape_embed -> [B,72,3] per-category centroids
  jepa_idea1:          (shape_embed + voxel_xyz) -> [B,512,72] per-voxel dists

GRADIENT PATHS INTO shape_embed (COMPLETE):
  Head 1  MeanColorHead:       MSE(pred_mean_color, gt)       [3]
  Head 2  SceneSemanticHead:   KL(p_s || p_hat)               [72]
  Head 3  AnchorPositionHead:  MSE(pred_anchors, gt)          [512x3]
  Head 4  SceneLayoutHead:     MSE(pred_centroids, gt)        [72x3]  NEW
  Head 5  SpatialSemanticHead: KL(per-voxel dist || pred)     [512x72] NEW
  Opt1/2  decoder conditioning: reconstruction loss           (indirect)
  Disentangle: mu_s_proj adds direct KL path                  NEW
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
    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid())
        total = sum(p.numel() for p in self.parameters())
        print(f"[MeanColorHead] [B,{width}] -> [B,3] sigmoid | {total:,} params")

    def forward(self, shape_embed):
        return self.head(shape_embed)


class SceneSemanticHead(nn.Module):
    NUM_LABELS = 72
    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),  nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, self.NUM_LABELS))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SceneSemanticHead] [B,{width}] -> [B,72] softmax | {total:,} params")

    def forward(self, shape_embed):
        return F.softmax(self.head(shape_embed), dim=-1)


class AnchorPositionHead(nn.Module):
    NUM_TOKENS = 512
    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),   nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, self.NUM_TOKENS * 3))
        total = sum(p.numel() for p in self.parameters())
        print(f"[AnchorPositionHead] [B,{width}] -> [B,512,3] | {total:,} params")

    def forward(self, shape_embed):
        B = shape_embed.shape[0]
        return self.head(shape_embed).reshape(B, self.NUM_TOKENS, 3)


class SceneLayoutHead(nn.Module):
    """
    NEW — DC/AC Position Decomposition.
    shape_embed -> [B, 72, 3] per-category spatial centroids.
    Mirrors MeanColorHead but for position (72 category centres vs 1 scene mean).
    Ground truth: mean xyz per ScanNet72 category (from dataset).
    Loss: weighted MSE over occupied categories (category_valid mask).
    Reduces position dynamic range mu_g must encode: +-10m -> +-0.5m offsets.
    """
    NUM_CATS = 72
    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, self.NUM_CATS * 3))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SceneLayoutHead] [B,{width}] -> [B,72,3] per-cat centroids | {total:,} params")
        print(f"  DC/AC for position — analogue of MeanColorHead for xyz")

    def forward(self, shape_embed):
        B = shape_embed.shape[0]
        return self.head(shape_embed).reshape(B, self.NUM_CATS, 3)


class SpatialSemanticHead(nn.Module):
    """
    NEW — JEPA Idea 1: Spatially-resolved semantic prediction.
    (shape_embed [B, W] concat voxel_center [B, K, 3]) -> [B, K, 72] softmax.
    Unlike SceneSemanticHead (global WHAT), teaches WHERE each category is.
    Called EXTERNALLY in training loop — needs scaffold_anchors from batch_data.
    Requires position_scaffold=True in dataset.
    """
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
        print(f"  JEPA Idea 1 — spatially-resolved semantic prediction")

    def forward(self, shape_embed, voxel_centers):
        """
        shape_embed:   [B, W]
        voxel_centers: [B, K, 3]
        returns:       [B, K, 72]
        """
        B, K, _ = voxel_centers.shape
        se_exp   = shape_embed.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([se_exp, voxel_centers], dim=-1)
        flat     = combined.reshape(B * K, -1)
        out      = self.head(flat).reshape(B, K, self.NUM_CATS)
        return F.softmax(out, dim=-1)


# ============================================================================
# PER-GAUSSIAN SEMANTIC HEADS (InfoNCE path — via decoder hidden state)
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
# GEOMETRY DECODER
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
# GS DECODER MLP
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
# BASE PERCEIVER (unchanged from original)
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
    Full Can3Tok VAE with all objectives and decoder shape conditioning.

    NEW FLAGS (in addition to existing):
      latent_disentangle (bool):
        Splits mu = concat(mu_s [B,semantic_dims], mu_g [B,16384-semantic_dims]).
        mu_s from shape_embed via mu_s_proj_mean/var.
        mu_g from kl_flat via kl_emb_proj_mean_g/var_g.
        Total mu = 16384 unchanged. Decoder reshape unchanged.
        Training loop reads _mu_s_cache/_mu_g_cache for cross-recon and ortho.

      semantic_dims (int, default 512):
        Must satisfy: semantic_dims % embed_dim == 0 and semantic_dims < 16384.

      scene_layout_head (bool):
        SceneLayoutHead: shape_embed -> [B,72,3] category centroids.
        Prediction in self.last_scene_layout_pred.

      jepa_idea1 (bool):
        SpatialSemanticHead: (shape_embed + voxel_center) -> [B,512,72].
        Called externally (training loop calls self.spatial_semantic_module).
        self.last_spatial_semantic_pred always None after forward().
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
                 # NEW disentanglement flags
                 latent_disentangle=False,
                 semantic_dims=512,
                 scene_layout_head=False,
                 jepa_idea1=False):

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
        self.scene_semantic_head      = scene_semantic_head
        self.position_scaffold        = position_scaffold
        self.decoder_shape_prepend    = decoder_shape_prepend
        self.decoder_shape_cross_attn = decoder_shape_cross_attn
        self.latent_disentangle       = latent_disentangle
        self.semantic_dims            = semantic_dims
        self._jepa_idea1_enabled      = jepa_idea1

        print(f"\n{'='*70}")
        print(f"  CAN3TOK")
        print(f"  semantic='{semantic_mode}' | color_residual={color_residual}")
        print(f"  scene_semantic_head={scene_semantic_head} | position_scaffold={position_scaffold}")
        print(f"  decoder_shape_prepend={decoder_shape_prepend} | decoder_shape_cross_attn={decoder_shape_cross_attn}")
        print(f"  latent_disentangle={latent_disentangle}  semantic_dims={semantic_dims}")
        print(f"  scene_layout_head={scene_layout_head}  jepa_idea1={jepa_idea1}")
        print(f"{'='*70}")

        # Existing auxiliary heads
        self.mean_color_head      = None
        self.last_mean_color_pred = None
        if color_residual:
            self.mean_color_head = MeanColorHead(width=width)

        self.scene_semantic_module    = None
        self.last_scene_semantic_pred = None
        if scene_semantic_head:
            self.scene_semantic_module = SceneSemanticHead(width=width)

        self.anchor_position_head = None
        self.last_anchor_pred     = None
        if position_scaffold:
            self.anchor_position_head = AnchorPositionHead(width=width)

        # NEW: SceneLayoutHead
        self.scene_layout_module    = None
        self.last_scene_layout_pred = None
        if scene_layout_head:
            self.scene_layout_module = SceneLayoutHead(width=width)

        # NEW: SpatialSemanticHead (JEPA Idea 1)
        # Called externally — forward() always sets last_spatial_semantic_pred=None.
        self.spatial_semantic_module    = None
        self.last_spatial_semantic_pred = None
        if jepa_idea1:
            if not position_scaffold:
                print(f"  [WARNING] jepa_idea1=True requires position_scaffold=True. Disabled.")
            else:
                self.spatial_semantic_module = SpatialSemanticHead(width=width, num_tokens=512)

        # NEW: Latent disentanglement projections
        # kl_emb_proj_mean/var (from parent) KEPT for backward compat.
        # _mu_s_cache / _mu_g_cache used by training loop.
        self._mu_s_cache = None
        self._mu_g_cache = None

        if latent_disentangle:
            assert embed_dim > 0, "latent_disentangle requires embed_dim > 0"
            assert semantic_dims % embed_dim == 0, \
                f"semantic_dims ({semantic_dims}) must be divisible by embed_dim ({embed_dim})"
            geom_dims = 64 * 64 * 4 - semantic_dims
            assert geom_dims > 0, f"semantic_dims ({semantic_dims}) must be < 16384"

            self.mu_s_proj_mean = nn.Linear(width, semantic_dims)
            self.mu_s_proj_var  = nn.Linear(width, semantic_dims)

            kl_in = (1 + num_latents - 1) * embed_dim  # 512 * 32 = 16384
            self.kl_emb_proj_mean_g = nn.Linear(kl_in, geom_dims)
            self.kl_emb_proj_var_g  = nn.Linear(kl_in, geom_dims)

            print(f"  DISENTANGLE: mu_s[{semantic_dims}] from shape_embed | mu_g[{geom_dims}] from tokens")

        # Decoder shape conditioning (existing)
        self.project_shape_for_prepend = None
        if decoder_shape_prepend:
            self.project_shape_for_prepend = nn.Sequential(
                nn.Linear(width, width), nn.LayerNorm(width))
            total = sum(p.numel() for p in self.project_shape_for_prepend.parameters())
            print(f"  Option 1: shape prepend | {total:,} params")

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
            proj_p = sum(p.numel() for p in self.project_shape_for_cross_attn.parameters())
            attn_p = sum(p.numel() for p in self.shape_cross_attn_layers.parameters())
            print(f"  Option 2: {decoder_cross_attn_layers} cross-attn layers | {(proj_p+attn_p)/1e6:.3f}M params")

        # Per-Gaussian InfoNCE heads
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
            print(f"  InfoNCE: mode='{semantic_mode}'")
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
        """
        Full encode. When latent_disentangle=True:
          mu_s = mu_s_proj_mean(shape_embed)            [B, D_s]
          mu_g = kl_emb_proj_mean_g(kl_flat)            [B, D_g]
          mu   = concat(mu_s, mu_g)                     [B, 16384]
          _mu_s_cache, _mu_g_cache set for training loop.
        """
        shape_embed, latents    = self.encode_latents(pc, feats)
        self._shape_embed_cache = shape_embed

        kl_embed, posterior = self.encode_kl_embed(latents, sample_posterior)
        kl_flat = kl_embed.reshape(kl_embed.shape[0], -1)

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
               shape_embed=None):
        """
        Decode [B, 512, 32] -> Gaussian attributes.
        When latent_disentangle=True, z = concat(z_s, z_g), reshape works identically.
        Cross-reconstruction enforcement is in the training loop, not here.
        """
        latents = self.post_kl(latents)

        if (self.decoder_shape_cross_attn and
                self.shape_cross_attn_layers is not None and
                shape_embed is not None):
            shape_context = self.project_shape_for_cross_attn(shape_embed).unsqueeze(1)
            for cross_attn_layer in self.shape_cross_attn_layers:
                latents = cross_attn_layer(latents, shape_context)

        shape_token_prepended = False
        if (self.decoder_shape_prepend and
                self.project_shape_for_prepend is not None and
                shape_embed is not None):
            shape_token = self.project_shape_for_prepend(shape_embed).unsqueeze(1)
            latents = torch.cat([shape_token, latents], dim=1)
            shape_token_prepended = True

        latents_out = self.transformer(latents)
        if shape_token_prepended:
            latents_out = latents_out[:, 1:, :]

        latents_flat = latents_out.reshape(latents_out.shape[0], -1)

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
                    recon_g[:, :, 0:3], latents_out)
            elif self.semantic_mode == 'dist':
                semantic_features = self.semantic_distribution_head(hidden)

        return reconstruction, semantic_features

    def forward(self, pc, feats, volume_queries, sample_posterior=True):
        """
        6-value return (unchanged):
          shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features

        After forward(), training loop reads:
          model.shape_model.last_mean_color_pred       [B,3] or None
          model.shape_model.last_scene_semantic_pred   [B,72] or None
          model.shape_model.last_anchor_pred           [B,512,3] or None
          model.shape_model.last_scene_layout_pred     [B,72,3] or None  NEW
          model.shape_model.last_spatial_semantic_pred None (set externally) NEW
          model.shape_model._mu_s_cache                [B,D_s] or None   NEW
          model.shape_model._mu_g_cache                [B,D_g] or None   NEW
          model.shape_model._shape_embed_cache         [B,width]
        """
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)

        latents = z.reshape(z.shape[0], 512, 32)
        UV_gs_recover, per_gaussian_features = self.decode(
            latents, volume_queries,
            return_semantic_features=self.training,
            shape_embed=self._shape_embed_cache)

        _se = self._shape_embed_cache

        self.last_mean_color_pred = (
            self.mean_color_head(_se) if self.mean_color_head is not None else None)
        self.last_scene_semantic_pred = (
            self.scene_semantic_module(_se)
            if self.scene_semantic_module is not None else None)
        self.last_anchor_pred = (
            self.anchor_position_head(_se)
            if self.anchor_position_head is not None else None)
        self.last_scene_layout_pred = (
            self.scene_layout_module(_se)
            if self.scene_layout_module is not None else None)
        # SpatialSemanticHead always called externally
        self.last_spatial_semantic_pred = None

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features