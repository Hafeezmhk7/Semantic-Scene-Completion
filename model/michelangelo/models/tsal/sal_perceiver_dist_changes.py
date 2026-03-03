# -*- coding: utf-8 -*-
"""
sal_perceiver.py  —  Can3Tok VAE with Step 1 Color Residual
=============================================================

ARCHITECTURE OVERVIEW
---------------------
AlignedShapeLatentPerceiver.encode_latents():
  Encoder produces 1 + 512 = 513 tokens via Perceiver cross-attention:
    shape_embed = x[:, 0]   [B, width=384]   first token
    latents     = x[:, 1:]  [B, 512, 384] -> pre_kl -> mu [B, 16384]

  STEP 1 — shape_embed now receives a gradient for the first time:
    MeanColorHead: shape_embed -> Linear(384,64) -> ReLU -> Linear(64,3) -> Sigmoid
    Predicts scene mean RGB in [0,1].
    Loss: MSE(pred_mean_color, gt_mean_color) added to total_loss.
    gradient path: color_loss -> MeanColorHead -> shape_embed -> encoder token 0
                   -> cross-attn weights (shared with mu path, does NOT interfere)

  shape_embed and the GS_decoder hidden state [B,1024] are fully independent.
  InfoNCE on hidden mode therefore combines cleanly with Step 1.

SEMANTIC MODES
--------------
  'none'       Pure VAE baseline
  'hidden'     InfoNCE contrastive on GS_decoder hidden state [B, 1024]
  'geometric'  InfoNCE on reconstructed Gaussians [B, N, 14]
  'attention'  InfoNCE via cross-attention head
  'dist'       Label Distribution Learning: KL(p_s || softmax(logits))

SCALE INITIALIZATION
--------------------
_initialize_scale_bias has been removed. InfoNCE hidden mode converges
correctly without it. exp(0) = 1.0m initial scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import repeat
import math
import numpy as np

from model.michelangelo.models.modules import checkpoint
from model.michelangelo.models.modules.embedder import FourierEmbedder
from model.michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from model.michelangelo.models.modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer,
)
from .tsal_base import ShapeAsLatentModule


# ============================================================================
# STEP 1 — MEAN COLOR HEAD
# ============================================================================

class MeanColorHead(nn.Module):
    """
    Predicts scene mean RGB from shape_embed [B, width].

    Gives shape_embed its first gradient signal since training began.
    Tiny (8,451 params) — negligible compute overhead.

    Output: [B, 3] in [0, 1]  (Sigmoid ensures valid RGB range)
    Loss:   MSE vs gt mean_color from dataset batch dict
    """
    def __init__(self, width: int = 384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[MeanColorHead] shape_embed [B,{width}] -> Linear(64) -> [B,3] sigmoid")
        print(f"  Parameters: {total:,}  (gradient path to encoder token 0)")

    def forward(self, shape_embed: torch.Tensor) -> torch.Tensor:
        return self.head(shape_embed)


# ============================================================================
# SEMANTIC PROJECTION HEADS
# ============================================================================

class SemanticProjectionHead(nn.Module):
    """
    InfoNCE hidden state head.
    hidden [B,1024] -> MLP -> [B, 40000, 32] L2-normalized features.
    Positive pair = same ScanNet72 segment label.
    """
    def __init__(self, hidden_dim=1024, num_gaussians=40000, feature_dim=32):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_gaussians = num_gaussians
        self.feature_dim   = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_gaussians * feature_dim),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHead] hidden [B,{hidden_dim}]"
              f" -> MLP -> [B,{num_gaussians},{feature_dim}] L2-norm  |  {total/1e6:.3f}M params")

    def forward(self, hidden):
        B = hidden.shape[0]
        features = self.projection(hidden).reshape(B, self.num_gaussians, self.feature_dim)
        return F.normalize(features, p=2, dim=-1)


class SemanticDistributionHead(nn.Module):
    """
    Label Distribution Learning head.
    hidden [B,1024] -> MLP -> [B,72] logits.
    Loss: KL(p_s || softmax(logits)).
    """
    def __init__(self, hidden_dim: int = 1024, num_labels: int = 72):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_labels),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticDistributionHead] hidden [B,{hidden_dim}]"
              f" -> MLP -> [B,{num_labels}] logits  |  {total/1e6:.3f}M params")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)


class SemanticProjectionHeadGeometric(nn.Module):
    """InfoNCE on reconstructed 14-param Gaussians."""
    def __init__(self, gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.projection = nn.Sequential(
            nn.Linear(gaussian_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHeadGeometric] gaussians [B,{num_gaussians},{gaussian_dim}]"
              f" -> MLP -> [B,{num_gaussians},{feature_dim}]  |  {total/1e6:.3f}M params")

    def forward(self, gaussians):
        B, N, D = gaussians.shape
        features = self.projection(gaussians.reshape(B * N, D)).reshape(B, N, -1)
        return F.normalize(features, p=2, dim=-1)


# ============================================================================
# ENCODER
# ============================================================================

class CrossAttentionEncoder(nn.Module):
    def __init__(self, *, device, dtype, num_latents, fourier_embedder,
                 fourier_embedder_ID, point_feats, width, heads, layers,
                 init_scale=0.25, qkv_bias=True, flash=False,
                 use_ln_post=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint  = use_checkpoint
        self.num_latents     = num_latents
        self.fourier_embedder    = fourier_embedder
        self.fourier_embedder_ID = fourier_embedder_ID
        self.point_feats     = point_feats

        # Learnable queries initialised with coarse voxel structure
        voxel_reso = 4
        x_y = np.linspace(-8, 8, voxel_reso)
        xv, yv, zv = np.meshgrid(x_y, x_y, x_y, indexing='ij')
        voxel_centers = torch.tensor(
            np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T,
            device=device, dtype=dtype
        ).reshape([-1, 3])
        dummy = torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02
        dummy[:, :192] = voxel_centers.reshape([-1]) * 0.01
        self.query = nn.Parameter(dummy)

        self.input_proj = nn.Linear(
            fourier_embedder.out_dim + point_feats + fourier_embedder_ID.out_dim,
            width, device=device, dtype=dtype
        )
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
        )
        self.self_attn = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=width,
            layers=layers, heads=heads, init_scale=init_scale,
            qkv_bias=qkv_bias, flash=flash, use_checkpoint=False
        )
        self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device) if use_ln_post else None

    def _forward(self, pc, feats):
        bs = pc.shape[0]
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
# DECODER
# ============================================================================

class CrossAttentionDecoder(nn.Module):
    def __init__(self, *, device, dtype, num_latents, out_channels,
                 fourier_embedder, width, heads, init_scale=0.25,
                 qkv_bias=True, flash=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint   = use_checkpoint
        self.fourier_embedder = fourier_embedder
        self.query_proj = nn.Linear(
            fourier_embedder.out_dim, width, device=device, dtype=dtype)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, n_data=num_latents, width=width,
            heads=heads, init_scale=init_scale, qkv_bias=qkv_bias, flash=flash)
        self.ln_post     = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries, latents):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        return self.output_proj(x)

    def forward(self, queries, latents):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class GaussianSemanticAttentionHead(CrossAttentionDecoder):
    def forward(self, gaussian_xyz, scene_tokens):
        features = super().forward(gaussian_xyz, scene_tokens)
        return F.normalize(features, p=2, dim=-1)


# ============================================================================
# GS_DECODER  (14-param output, no scale bias init)
# ============================================================================

class GS_decoder(nn.Module):
    """
    MLP decoder: latent -> 14-param Gaussians per point.

    Output per Gaussian (14 values):
      [0:3]   xyz      — unbounded
      [3:6]   rgb      — clamp(0,1) when color_residual=False
                         RAW (unbounded) when color_residual=True
                           residuals are ~[-0.7, +0.4]; L2 loss bounds them.
                           clamp(0,1) was zeroing all negative residuals — WRONG.
      [6]     opacity  — sigmoid
      [7:10]  scale    — exp()   (no bias init; exp(0)=1.0m at epoch 0)
      [10:14] quat     — L2 normalised
    """
    def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56,
                 color_residual: bool = False):
        super().__init__()
        self.D, self.W = D, W
        self.color_residual = color_residual
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)])
        for i in range(D - 1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.LayerNorm(W))
            self.pts_linears.append(nn.ReLU())
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, return_hidden=False):
        for layer in self.pts_linears:
            x = layer(x)
        hidden     = x
        raw_output = self.output_linear(x)

        B = raw_output.shape[0]
        N = 40_000
        raw_output = raw_output.reshape(B, N, 14)

        position = raw_output[:, :, 0:3]

        if self.color_residual:
            # Residuals are unbounded (L2 loss drives them to the right range).
            # DO NOT clamp — negative residuals represent darker-than-mean pixels
            # and clamp(0,1) was silently zeroing all of them.
            color = raw_output[:, :, 3:6]
        else:
            color = torch.clamp(raw_output[:, :, 3:6], 0.0, 1.0)

        opacity  = torch.sigmoid(raw_output[:, :, 6:7])
        scale    = torch.exp(raw_output[:, :, 7:10])
        quat     = F.normalize(raw_output[:, :, 10:14], p=2, dim=-1)

        output = torch.cat([position, color, opacity, scale, quat], dim=-1).reshape(B, -1)

        if return_hidden:
            return output, hidden
        return output


# ============================================================================
# SHAPE AS LATENT PERCEIVER BASE
# ============================================================================

class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False,
                 color_residual: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_latents    = num_latents

        self.fourier_embedder    = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi, input_dim=3)
        self.fourier_embedder_ID = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi, input_dim=3)

        init_scale = init_scale * math.sqrt(1.0 / width)

        self.encoder = CrossAttentionEncoder(
            device=device, dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            fourier_embedder_ID=self.fourier_embedder_ID,
            num_latents=num_latents, point_feats=point_feats,
            width=width, heads=heads, layers=num_encoder_layers,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_ln_post=use_ln_post, use_checkpoint=use_checkpoint,
        )

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
            qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint,
        )

        print(f"\n  GS_DECODER: 40000 x 14 = 560,000 output dims (xyz+rgb+opacity+scale+quat)")
        print(f"  Color activation: {'NONE (unbounded residuals)' if color_residual else 'clamp(0,1)'}")
        self.GS_decoder = GS_decoder(3, 1024, width * 512, [4], 40000 * 14,
                                     color_residual=color_residual)

        self.kl_emb_proj_mean = nn.Linear((num_latents - 1) * embed_dim, 64 * 64 * 4, dtype=dtype)
        self.kl_emb_proj_var  = nn.Linear((num_latents - 1) * embed_dim, 64 * 64 * 4, dtype=dtype)

        self.geo_decoder = CrossAttentionDecoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder,
            out_channels=1, num_latents=num_latents, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_checkpoint=use_checkpoint,
        )

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
        latents_decoded = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)
        return logits, center_pos, posterior


# ============================================================================
# ALIGNED SHAPE LATENT PERCEIVER  (full Can3Tok model)
# ============================================================================

class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):
    """
    Full Can3Tok autoencoder with optional semantic heads and Step 1 color residual.

    Key gradient paths:
      Reconstruction: L2_recon -> GS_decoder -> post_kl -> transformer -> mu
      InfoNCE:        InfoNCE  -> SemanticHead -> GS_decoder hidden -> ... -> mu
      Step 1 color:   MSE_color -> MeanColorHead -> shape_embed -> encoder token 0

    shape_embed and GS_decoder hidden are independent — no interference.
    MeanColorHead is only added when color_residual=True.
    """

    def __init__(self, *, device, dtype, num_latents, point_feats=0, embed_dim=0,
                 num_freqs=8, include_pi=True, width, heads, num_encoder_layers,
                 num_decoder_layers, init_scale=0.25, qkv_bias=True, flash=True,
                 use_ln_post=False, use_checkpoint=False,
                 semantic_mode: str = 'none',
                 color_residual: bool = False):

        super().__init__(
            device=device, dtype=dtype,
            num_latents=1 + num_latents,   # +1 for shape_embed token
            point_feats=point_feats, embed_dim=embed_dim,
            num_freqs=num_freqs, include_pi=include_pi,
            width=width, heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale, qkv_bias=qkv_bias,
            flash=flash, use_ln_post=use_ln_post, use_checkpoint=use_checkpoint,
            color_residual=color_residual,   # passed to GS_decoder
        )

        self.width          = width
        self.semantic_mode  = semantic_mode
        self.color_residual = color_residual

        print(f"\n{'='*70}")
        print(f"  CAN3TOK | semantic='{semantic_mode}' | color_residual={color_residual}")
        print(f"{'='*70}")

        # ── Step 1: Mean Color Head ───────────────────────────────────────────
        self.mean_color_head = None
        if color_residual:
            self.mean_color_head = MeanColorHead(width=width)
            print(f"  Step 1 ENABLED: shape_embed -> MeanColorHead -> [B,3] mean RGB")
        else:
            print(f"  Step 1 DISABLED: shape_embed [B,{width}] idle (no gradient)")

        # ── Semantic heads ────────────────────────────────────────────────────
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
            self.semantic_distribution_head = SemanticDistributionHead(
                hidden_dim=1024, num_labels=72)
        elif semantic_mode == 'none':
            pass
        else:
            raise ValueError(f"Unknown semantic_mode: '{semantic_mode}'")

        print(f"  Decoder scale activation: exp()  (no bias init)")
        print(f"{'='*70}\n")

    # ── encode / decode helpers ───────────────────────────────────────────────

    def encode_latents(self, pc, feats=None):
        """
        Returns:
          shape_embed [B, width]      — encoder token 0
                                        gets gradient via MeanColorHead if color_residual=True
          latents     [B, 512, width] — flows to mu [B, 16384]
        """
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
        shape_embed, latents        = self.encode_latents(pc, feats)
        kl_embed, posterior         = self.encode_kl_embed(latents, sample_posterior)
        kl_embed_flat               = kl_embed.reshape([kl_embed.shape[0], -1])
        mu      = self.kl_emb_proj_mean(kl_embed_flat)
        log_var = self.kl_emb_proj_var(kl_embed_flat)
        std     = torch.exp(0.5 * log_var)
        z       = mu + std * torch.randn_like(std)
        return shape_embed, mu, log_var, z, posterior

    def decode(self, latents, volume_queries=None, return_semantic_features=False):
        """
        Returns:
          reconstruction    [B, N*14]   always
          semantic_features [B, N, 32] or [B, 72] or None
          mean_color_pred   [B, 3]      only when color_residual=True, else None
        """
        latents             = self.post_kl(latents)
        latents_transformed = self.transformer(latents)
        latents_flat        = latents_transformed.reshape(latents_transformed.shape[0], -1)

        has_sem_head = any([
            self.semantic_projection_hidden    is not None,
            self.semantic_projection_geometric is not None,
            self.semantic_attention_head       is not None,
            self.semantic_distribution_head    is not None,
        ])
        need_hidden = return_semantic_features and self.training and has_sem_head

        if need_hidden:
            reconstruction, hidden = self.GS_decoder(latents_flat, return_hidden=True)
        else:
            reconstruction = self.GS_decoder(latents_flat, return_hidden=False)

        # Semantic features
        semantic_features = None
        if return_semantic_features and self.training and has_sem_head:
            B = reconstruction.shape[0]
            recon_g = reconstruction.reshape(B, 40000, 14)
            if self.semantic_mode == 'hidden':
                semantic_features = self.semantic_projection_hidden(hidden)
            elif self.semantic_mode == 'geometric':
                semantic_features = self.semantic_projection_geometric(recon_g)
            elif self.semantic_mode == 'attention':
                semantic_features = self.semantic_attention_head(recon_g[:, :, 0:3], latents_transformed)
            elif self.semantic_mode == 'dist':
                semantic_features = self.semantic_distribution_head(hidden)

        return reconstruction, semantic_features

    def forward(self, pc, feats, volume_queries, sample_posterior=True):
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)

        latents_reshaped = z.reshape(z.shape[0], 512, 32)

        UV_gs_recover, per_gaussian_features = self.decode(
            latents_reshaped, volume_queries,
            return_semantic_features=self.training,
        )

        # ── Step 1: predict mean color from shape_embed ───────────────────────
        # Stored as self.last_mean_color_pred (instance attribute) instead of
        # a 7th return value — keeps the 6-value return signature that
        # asl_pl_module.py expects, requiring zero changes to that wrapper.
        # Training loop reads it via: gs_autoencoder.shape_model.last_mean_color_pred
        #
        # Gradient: MSE_color_loss -> MeanColorHead -> shape_embed -> encoder token 0.
        # Independent of InfoNCE path (GS_decoder hidden).
        if self.mean_color_head is not None:
            self.last_mean_color_pred = self.mean_color_head(shape_embed)  # [B, 3]
        else:
            self.last_mean_color_pred = None

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features