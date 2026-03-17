# -*- coding: utf-8 -*-
"""
sal_perceiver.py  —  Can3Tok VAE
==================================
Step 1: Color Residual  |  Move 1: Scene Semantic Head  |  Position Scaffold

GRADIENT PATHS INTO shape_embed (token 0 of encoder output)
-------------------------------------------------------------
  HEAD 1 — MeanColorHead (color_residual=True):
    shape_embed [B,384] → Linear(64) → [B,3] sigmoid
    Loss: MSE(pred, gt_mean_color)
    Signal: scene DC color (3 numbers)

  HEAD 2 — SceneSemanticHead (scene_semantic_head=True):
    shape_embed [B,384] → MLP(128→128) → [B,72] softmax
    Loss: KL(p_s ‖ p̂)
    Signal: scene semantic composition (72 numbers)

  HEAD 3 — AnchorPositionHead (position_scaffold=True):
    shape_embed [B,384] → MLP(512→512) → [B,512,3]
    Loss: MSE(pred_anchors, gt_anchors)
    Signal: spatial layout of the 512 token regions
    Inspiration: Scaffold-GS (Lu et al. CVPR 2024), LION (Zeng et al. NeurIPS 2022)

All three heads share the same shape_embed input but have INDEPENDENT
losses, outputs, and gradient paths.

POSITION SCAFFOLD DESIGN
--------------------------
  Dataset:
    Divides scene into 8³=512 super-voxels, computes anchor positions
    (mean position of Gaussians in each cell), and per-Gaussian offsets:
      δp_i = p_i − â_{k(i)}   (range ~[−1,+1]m vs absolute ~[−10,+10]m)

  Training:
    Decoder target for position = offsets δp_i (from batch dict)
    anchor_pred = AnchorPositionHead(shape_embed)  [B, 512, 3]
    L_anchor = MSE(anchor_pred, scaffold_anchors)

  Inference / reconstruction:
    p̂_i = decoder_output_i[0:3] + anchor_pred[scaffold_token_ids[i]]
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
# HEAD 1 — MEAN COLOR HEAD
# ============================================================================

class MeanColorHead(nn.Module):
    """Step 1: shape_embed → mean scene RGB. Gives shape_embed its first gradient."""
    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid())
        total = sum(p.numel() for p in self.parameters())
        print(f"[MeanColorHead] shape_embed [B,{width}] -> Linear(64) -> [B,3] sigmoid")
        print(f"  Parameters: {total:,}  (gradient path to encoder token 0)")

    def forward(self, shape_embed):
        return self.head(shape_embed)   # [B, 3]


# ============================================================================
# HEAD 2 — SCENE SEMANTIC HEAD
# ============================================================================

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
              f"-> [{self.NUM_LABELS}] softmax  |  {total:,} params")
        print(f"  Gradient path: KL_loss -> SceneSemanticHead -> shape_embed")

    def forward(self, shape_embed):
        return F.softmax(self.head(shape_embed), dim=-1)   # [B, 72]


# ============================================================================
# HEAD 3 — ANCHOR POSITION HEAD (Position Scaffold, Scaffold-GS inspired)
# ============================================================================

class AnchorPositionHead(nn.Module):
    """
    Scaffold-GS inspired: predicts 512 spatial anchors from shape_embed.

    Each of the 512 latent tokens is responsible for Gaussians in one of
    the 8×8×8 = 512 super-voxels covering the scene. This head predicts
    where each token's region is in 3D space, giving the decoder a spatial
    reference frame from which to predict small position OFFSETS rather than
    large absolute coordinates.

    Mapping to Scaffold-GS (Lu et al. CVPR 2024):
      Scaffold-GS anchor â_k  ↔  predicted anchor_pred[k]   [3]
      Scaffold-GS child Gaussians  ↔  Gaussians with scaffold_token_ids == k
      Scaffold-GS offset δp_j  ↔  decoder_output[i, 0:3]   (position channels)

    Input:  shape_embed [B, width]
    Output: anchors     [B, 512, 3]   predicted anchor positions
    Loss:   MSE(anchors, scaffold_anchors)  gt from dataset 'scaffold_anchors' key

    Variance reduction: decoder target shrinks from ~[-10,+10]m to ~[-1,+1]m.
    This is the same principle as color_residual (Step 1) applied to position.

    Parameters: ~670K
    """
    NUM_TOKENS = 512   # 8^3 = num effective latent tokens after KL projection

    def __init__(self, width=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),   nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, self.NUM_TOKENS * 3))   # 512 anchors × xyz
        total = sum(p.numel() for p in self.parameters())
        print(f"[AnchorPositionHead] shape_embed [B,{width}] -> MLP(512->512) "
              f"-> [B,{self.NUM_TOKENS},3]  |  {total:,} params")
        print(f"  Gradient path: MSE_anchor -> AnchorPositionHead -> shape_embed "
              f"(3rd path, Scaffold-GS inspired)")
        print(f"  Decoder predicts position OFFSETS δp_i (range ~[-1,+1]m)")
        print(f"  Reconstruction: abs_pos_i = δp̂_i + anchor_pred[token_id_i]")

    def forward(self, shape_embed):
        B = shape_embed.shape[0]
        return self.head(shape_embed).reshape(B, self.NUM_TOKENS, 3)   # [B, 512, 3]


# ============================================================================
# PER-GAUSSIAN SEMANTIC HEADS (InfoNCE)
# ============================================================================

class SemanticProjectionHead(nn.Module):
    """InfoNCE on GS_decoder hidden state. hidden [B,1024] → [B, 40k, 32] L2-norm."""
    def __init__(self, hidden_dim=1024, num_gaussians=40000, feature_dim=32):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim   = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, num_gaussians * feature_dim))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHead] hidden [B,{hidden_dim}]"
              f" -> MLP -> [B,{num_gaussians},{feature_dim}] L2-norm  |  {total/1e6:.3f}M params")

    def forward(self, hidden):
        B = hidden.shape[0]
        return F.normalize(
            self.projection(hidden).reshape(B, self.num_gaussians, self.feature_dim),
            p=2, dim=-1)


class SemanticDistributionHead(nn.Module):
    """Label Distribution Learning. hidden [B,1024] → [B,72] logits."""
    def __init__(self, hidden_dim=1024, num_labels=72):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, num_labels))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticDistributionHead] hidden [B,{hidden_dim}]"
              f" -> [B,{num_labels}] logits  |  {total/1e6:.3f}M params")

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
        print(f"[SemanticProjectionHeadGeometric] [B,{num_gaussians},{gaussian_dim}]"
              f" -> [B,{num_gaussians},{feature_dim}]  |  {total/1e6:.3f}M params")

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
        xyz_actual      = pc[:, :, 4:7]    # ALWAYS absolute positions for Fourier embedding
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
# DECODER
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
# GS DECODER — 14-param Gaussian output
# ============================================================================

class GS_decoder(nn.Module):
    """
    MLP: latent [B, 512*384] → Gaussian attributes [B, 40000, 14].

    Position channels [0:3]:
      position_scaffold=False → absolute xyz (unbounded, large targets)
      position_scaffold=True  → position OFFSET δp_i (small targets ~[-1,+1]m)
    The decoder MLP is IDENTICAL in both cases. Only the training TARGET differs.
    When scaffold is on, the training loop passes offset targets instead of
    absolute positions, so the decoder naturally learns small offset values.

    Color channels [3:6]:
      color_residual=False → clamp(0,1) for absolute RGB
      color_residual=True  → raw/unbounded for color residuals
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
        hidden     = x
        raw        = self.output_linear(x).reshape(x.shape[0], 40_000, 14)
        position   = raw[:, :, 0:3]
        color      = raw[:, :, 3:6] if self.color_residual \
                     else torch.clamp(raw[:, :, 3:6], 0.0, 1.0)
        opacity    = torch.sigmoid(raw[:, :, 6:7])
        scale      = torch.exp(raw[:, :, 7:10])
        quat       = F.normalize(raw[:, :, 10:14], p=2, dim=-1)
        out        = torch.cat([position, color, opacity, scale, quat],
                               dim=-1).reshape(x.shape[0], -1)
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
        self.transformer = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=width,
            layers=num_decoder_layers, heads=heads, init_scale=init_scale,
            qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint)
        print(f"\n  GS_DECODER: 40000 x 14 = 560,000 output dims")
        print(f"  Color activation: {'NONE (unbounded residuals)' if color_residual else 'clamp(0,1)'}")
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
    Full Can3Tok autoencoder with all semantic heads and optional position scaffold.

    shape_embed gradient paths (all independent):
      Head 1 MeanColorHead:     color_residual=True
      Head 2 SceneSemanticHead: scene_semantic_head=True
      Head 3 AnchorPositionHead: position_scaffold=True  ← NEW

    After forward(), training loop reads from instance attributes:
      self.last_mean_color_pred     [B, 3]     → MSE vs 'mean_color'
      self.last_scene_semantic_pred [B, 72]    → KL vs 'label_dist'
      self.last_anchor_pred         [B, 512,3] → MSE vs 'scaffold_anchors'
    """

    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False,
                 semantic_mode='none',
                 color_residual=False,
                 scene_semantic_head=False,
                 position_scaffold=False):     # NEW flag

        super().__init__(
            device=device, dtype=dtype, num_latents=1 + num_latents,
            point_feats=point_feats, embed_dim=embed_dim,
            num_freqs=num_freqs, include_pi=include_pi, width=width, heads=heads,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_ln_post=use_ln_post, use_checkpoint=use_checkpoint,
            color_residual=color_residual)

        self.width               = width
        self.semantic_mode       = semantic_mode
        self.color_residual      = color_residual
        self.scene_semantic_head = scene_semantic_head
        self.position_scaffold   = position_scaffold

        print(f"\n{'='*70}")
        print(f"  CAN3TOK | semantic='{semantic_mode}' | color_residual={color_residual}"
              f" | scene_semantic_head={scene_semantic_head} | position_scaffold={position_scaffold}")
        print(f"{'='*70}")

        # ── Head 1: Mean Color ────────────────────────────────────────────────
        self.mean_color_head      = None
        self.last_mean_color_pred = None
        if color_residual:
            self.mean_color_head = MeanColorHead(width=width)
            print(f"  Step 1 ENABLED: shape_embed -> MeanColorHead -> [B,3] mean RGB")
        else:
            print(f"  Step 1 DISABLED: MeanColorHead not active")

        # ── Head 2: Scene Semantic ────────────────────────────────────────────
        self.scene_semantic_module    = None
        self.last_scene_semantic_pred = None
        if scene_semantic_head:
            self.scene_semantic_module = SceneSemanticHead(width=width)
            print(f"  Move 1 ENABLED: shape_embed -> SceneSemanticHead -> [B,72] label dist")
        else:
            print(f"  Move 1 DISABLED: SceneSemanticHead not active")

        # ── Head 3: Anchor Position (Scaffold-GS inspired) ────────────────────
        # Predicts where each of the 512 token regions is located in 3D space.
        # This enables the decoder to predict position OFFSETS (small values)
        # rather than absolute positions (large values), reducing the prediction
        # variance by ~10x and following the same design principle as Step 1
        # (color residual) applied to the position attribute.
        self.anchor_position_head = None
        self.last_anchor_pred     = None    # [B, 512, 3], populated by forward()
        if position_scaffold:
            self.anchor_position_head = AnchorPositionHead(width=width)
            print(f"  Scaffold ENABLED: shape_embed -> AnchorPositionHead -> [B,512,3]")
            print(f"    Decoder target = offsets δp_i ~[-1,+1]m (vs absolute ~[-10,+10]m)")
        else:
            print(f"  Scaffold DISABLED: decoder predicts absolute positions")

        # ── Per-Gaussian InfoNCE heads (decoder hidden path) ──────────────────
        self.semantic_projection_hidden    = None
        self.semantic_projection_geometric = None
        self.semantic_attention_head       = None
        self.semantic_distribution_head    = None

        if semantic_mode == 'hidden':
            self.semantic_projection_hidden = SemanticProjectionHead(
                hidden_dim=1024, num_gaussians=40000, feature_dim=32)
        elif semantic_mode == 'geometric':
            self.semantic_projection_geometric = SemanticProjectionHeadGeometric(
                gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128)
        elif semantic_mode == 'attention':
            self.semantic_attention_head = GaussianSemanticAttentionHead(
                device=device, dtype=dtype, num_latents=num_latents, out_channels=32,
                fourier_embedder=self.fourier_embedder, width=width, heads=heads,
                init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
                use_checkpoint=use_checkpoint)
        elif semantic_mode == 'dist':
            self.semantic_distribution_head = SemanticDistributionHead(1024, 72)
        elif semantic_mode != 'none':
            raise ValueError(f"Unknown semantic_mode: '{semantic_mode}'")

        if semantic_mode != 'none':
            print(f"  InfoNCE ENABLED: mode='{semantic_mode}' (decoder hidden → mu)")
        print(f"  Decoder scale activation: exp()  (no bias init)")
        print(f"{'='*70}\n")

    def encode_latents(self, pc, feats=None):
        x, _ = self.encoder(pc, feats)
        return x[:, 0], x[:, 1:]    # shape_embed [B,width], latents [B,512,width]

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
        Returns (shape_embed, mu, log_var, z, posterior).
        Caches shape_embed in self._shape_embed_cache for the head predictions
        in forward() — avoids a second encoder pass.
        """
        shape_embed, latents       = self.encode_latents(pc, feats)
        self._shape_embed_cache    = shape_embed
        kl_embed, posterior        = self.encode_kl_embed(latents, sample_posterior)
        kl_flat = kl_embed.reshape([kl_embed.shape[0], -1])
        mu      = self.kl_emb_proj_mean(kl_flat)
        log_var = self.kl_emb_proj_var(kl_flat)
        z       = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return shape_embed, mu, log_var, z, posterior

    def decode(self, latents, volume_queries=None, return_semantic_features=False):
        """
        Decode [B, 512, 32] latent tokens → Gaussian attributes.

        Position channels [0:3] in the output:
          position_scaffold=False → absolute positions (unbounded)
          position_scaffold=True  → position OFFSETS δp_i (small)
        Both cases use the identical decoder MLP. The training loop
        selects the appropriate target externally.
        """
        latents             = self.post_kl(latents)
        latents_transformed = self.transformer(latents)
        latents_flat        = latents_transformed.reshape(latents_transformed.shape[0], -1)

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

        Head predictions stored as instance attributes for training loop:
          self.last_mean_color_pred     [B, 3]
          self.last_scene_semantic_pred [B, 72]
          self.last_anchor_pred         [B, 512, 3]  ← NEW (None if scaffold off)
        """
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)
        latents = z.reshape(z.shape[0], 512, 32)
        UV_gs_recover, per_gaussian_features = self.decode(
            latents, volume_queries, return_semantic_features=self.training)

        # Apply all three shape_embed heads using the cached token
        _se = self._shape_embed_cache   # [B, width], no extra encoder call

        self.last_mean_color_pred = (
            self.mean_color_head(_se) if self.mean_color_head is not None else None)

        self.last_scene_semantic_pred = (
            self.scene_semantic_module(_se)
            if self.scene_semantic_module is not None else None)

        # Head 3: predict 512 anchor positions from shape_embed.
        # Training loop reads this and computes MSE vs scaffold_anchors from batch.
        # At reconstruction: abs_pos_i = decoder_pos_output_i + anchor_pred[token_id_i]
        self.last_anchor_pred = (
            self.anchor_position_head(_se)
            if self.anchor_position_head is not None else None)

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features