# -*- coding: utf-8 -*-
"""
sal_perceiver.py  —  Can3Tok VAE
==================================
Step 1:    Color Residual             (color_residual)
Move 1:    Scene Semantic Head        (scene_semantic_head)
Scaffold:  Position Scaffold          (position_scaffold)
Option 1:  Decoder Shape Prepend      (decoder_shape_prepend)
Option 2:  Decoder Shape Cross-Attn   (decoder_shape_cross_attn)

DISENTANGLEMENT SUITE:
  latent_disentangle  — mu_s from shape_embed | mu_g from tokens
  scene_layout_head   — shape_embed -> [B,72,3] per-category centroids

NEW: THREE IDEAS FOR SPATIAL INDUCTIVE BIAS IN DECODER
======================================================

IDEA 0 — Decoder Positional Encoding  (decoder_pos_enc)
  Adds learnable positional embeddings PE[512, width] to the 512 decoder
  transformer tokens BEFORE the self-attention stack.
  Gives each token a unique identity so self-attention can learn
  structure-aware communication (semantic vs geometric tokens, nearby tokens).
  Zero risk: initialised near zero, no architectural change.

IDEA 1 — Segment Label Prediction  (predict_seg_labels)
  SegPredHead: takes decoder Gaussian outputs [B,40000,14] -> [B,40000,72].
  Per-Gaussian lightweight MLP (14 → 128 → 128 → 72).
  Cross-entropy loss vs gt segment labels.
  At inference: soft centroid lookup replaces need for gt segment labels:
    dc_i = sum_k  softmax(seg_logits_i)[k]  *  pred_centroids[k]
  Backprop forces Gaussian params (especially position) to be semantically
  discriminative. Leverages disentanglement: mu_s carries semantic structure
  that mu_g can query via self-attention to produce category-informative outputs.

IDEA 2 — Token Centroid Conditioning  (token_cond, token_cond_approach)
  Injects spatial identity into each of the 512 decoder transformer tokens
  BEFORE the self-attention stack, via Fourier-encoded spatial references.

  Approach A (--token_cond_approach A):
    Uses scaffold anchors [B, 512, 3] (voxel centroids from dataset).
    Each token gets the Fourier encoding of its voxel's mean position.
    Deterministic DC term, available from epoch 0.
    Requires position_scaffold=True.

  Approach B (--token_cond_approach B):
    Learnable soft assignment W [512, 72] (token -> category).
    token_centroid[t] = sum_k  W[t,k] * pred_centroids[k]
    Injects predicted category centroids from SceneLayoutHead.
    Semantically-grounded; adapts as SceneLayoutHead learns.
    Requires scene_layout_head=True.

  Approach AB (--token_cond_approach AB):
    Both A and B applied additively.

  Mathematical justification:
    After conditioning, each transformer token carries a spatial positional
    bias. The subsequent self-attention can exploit this to organise tokens
    spatially, and the flat GS_decoder MLP can exploit the spatial structure
    in the flattened token sequence (analogous to how CNNs exploit spatial
    structure in feature maps via shared convolutional weights).

IDEA 3 — Spatial-Aware Per-Gaussian Decoder  (query_decoder)
  Replaces the flat GS_decoder MLP with a per-Gaussian decoder that has
  explicit spatial inductive bias.

  Architecture:
    For each Gaussian i:
      token_feat[i]   = transformer_tokens[scaffold_token_id[i]]  [B,40000,width]
      spatial_enc[i]  = FourierEmbed(scaffold_anchor[token_id[i]])  [B,40000,fourier_dim]
      combined[i]     = concat(token_feat[i], spatial_enc[i])
      output[i]       = MLP_per_gaussian(combined[i])  -> 14 params

  Key properties:
    - Each Gaussian knows its spatial region (from scaffold anchor Fourier enc)
    - Each Gaussian has access to its token's learned scene representation
    - Position regression target is implicitly local to each token's voxel
    - Only ~350K params vs ~800M for GS_decoder
  Requires position_scaffold=True.

GRADIENT PATHS INTO shape_embed (COMPLETE):
  Head 1  MeanColorHead:       MSE(pred_mean_color, gt)       [3]
  Head 2  SceneSemanticHead:   KL(p_s || p_hat)               [72]
  Head 3  AnchorPositionHead:  MSE(pred_anchors, gt)          [512x3]
  Head 4  SceneLayoutHead:     MSE(pred_centroids, gt)        [72x3]
  Head 5  SpatialSemanticHead: KL(per-voxel dist || pred)     [512x72]
  Idea 2B token_cond: reconstruction path through token centroids
  Idea 1  seg_pred: reconstruction loss backprop through seg labels
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
# SHAPE_EMBED AUXILIARY HEADS (unchanged)
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
# NEW: THREE IDEAS — SPATIAL INDUCTIVE BIAS MODULES
# ============================================================================

class SegPredHead(nn.Module):
    """
    IDEA 1 — Per-Gaussian segment label prediction.

    Takes the 14-dimensional Gaussian parameters output by the decoder
    (position 3, color 3, opacity 1, scale 3, rotation 4) and predicts
    which of the 72 ScanNet categories each Gaussian belongs to.

    Why use Gaussian params as input (not the transformer hidden state):
      The hidden state is a single 1024-dim vector for ALL 40k Gaussians —
      no per-Gaussian identity. The 14 Gaussian params ARE per-Gaussian
      and carry position, which correlates strongly with category
      (floor at y=-4m, ceiling at y=+4m, etc.).

    Gradient flow:
      CE_seg → seg_pred_head → gaussian_params [B,40000,14] → GS_decoder
      → transformer → z → encoder. This forces the decoder to produce
      position values that are categorically discriminative, which is
      exactly the spatial inductive bias we want.

    At inference (no gt labels needed):
      dc_i = Σ_k  softmax(seg_logits_i)[k] × pred_centroids[k]
      final_pos_i = decoder_pos_i + dc_i
    """
    NUM_CATS = 72

    def __init__(self, in_dim=14, num_cats=72):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, num_cats))
        total = sum(p.numel() for p in self.parameters())
        print(f"[SegPredHead] [B,40000,{in_dim}] -> [B,40000,{num_cats}] | {total:,} params")
        print(f"  Idea 1: seg labels from Gaussian params (no gt needed at inference)")

    def forward(self, gaussian_params):
        """
        gaussian_params: [B, 40000, 14]
        returns: [B, 40000, 72] — raw logits (apply softmax for probabilities)
        """
        B, N, D = gaussian_params.shape
        return self.head(gaussian_params.reshape(B * N, D)).reshape(B, N, self.NUM_CATS)


class TokenCondMLP(nn.Module):
    """
    IDEA 2 — Spatial-to-token-space projection.
    Maps Fourier-encoded 3D spatial references to token-width vectors.
    Used for both Approach A (scaffold anchors) and Approach B (category centroids).
    """
    def __init__(self, fourier_dim, width):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width))
        total = sum(p.numel() for p in self.parameters())
        print(f"  [TokenCondMLP] fourier({fourier_dim}) -> token_bias({width}) | {total:,} params")

    def forward(self, fourier_encoded):
        """fourier_encoded: [B, 512, fourier_dim] -> [B, 512, width]"""
        B, T, D = fourier_encoded.shape
        return self.mlp(fourier_encoded.reshape(B * T, D)).reshape(B, T, -1)


class SpatialAwareDecoder(nn.Module):
    """
    IDEA 3 — Per-Gaussian spatial decoder, replacing the flat GS_decoder MLP.

    Architecture per Gaussian i:
      1. Gather: token_feat[i] = transformer_tokens[b, scaffold_token_id[i], :]
                                  [B, 40000, width]
      2. Spatial:  spatial_enc[i] = Fourier(scaffold_anchor[token_id[i]])
                                    [B, 40000, fourier_dim]
      3. Combined: concat(token_feat, spatial_enc) -> [B, 40000, width+fourier_dim]
      4. MLP:      per-Gaussian lightweight MLP -> [B, 40000, 14]

    Properties:
      - Each Gaussian knows WHICH spatial region (via Fourier spatial enc)
      - Each Gaussian has access to its token's learned scene context
      - MLP is tiny (~350K params) vs GS_decoder (~800M params)
      - Position regression is implicitly local within each voxel
      - Requires position_scaffold=True
    """

    def __init__(self, token_dim, fourier_embedder, hidden_dim=256, color_residual=False):
        super().__init__()
        self.fourier_embedder = fourier_embedder  # shared reference
        fourier_dim = fourier_embedder.out_dim
        self.color_residual = color_residual

        self.mlp = nn.Sequential(
            nn.Linear(token_dim + fourier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 14))

        total = sum(p.numel() for p in self.parameters()
                    if p is not self.fourier_embedder)
        # Count only MLP params (fourier_embedder is shared)
        total = sum(p.numel() for p in self.mlp.parameters())
        print(f"[SpatialAwareDecoder] token({token_dim})+fourier({fourier_dim})"
              f" -> hidden({hidden_dim}) -> 14 | {total:,} MLP params")
        print(f"  Idea 3: per-Gaussian decoder with spatial inductive bias")
        print(f"  Each Gaussian: gather(token_feat) + Fourier(voxel_anchor) -> 14 params")

    def forward(self, transformer_tokens, scaffold_anchors, scaffold_token_ids):
        """
        transformer_tokens:   [B, 512, token_dim]
        scaffold_anchors:     [B, 512, 3]   — voxel centroid positions
        scaffold_token_ids:   [B, 40000]    — which voxel each Gaussian belongs to

        Returns: [B, 40000*14] to match GS_decoder output format
        """
        B, T, D = transformer_tokens.shape
        N = scaffold_token_ids.shape[1]  # 40000

        # Step 1: per-Gaussian anchor position
        # scaffold_token_ids: [B, N] → index into scaffold_anchors [B, 512, 3]
        # per_gaussian_anchor: [B, N, 3]
        idx_for_anchors = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, 3)
        per_gaussian_anchor = torch.gather(scaffold_anchors, 1, idx_for_anchors)  # [B, N, 3]

        # Step 2: Fourier encode per-Gaussian spatial anchor
        spatial_enc = self.fourier_embedder(per_gaussian_anchor)  # [B, N, fourier_dim]

        # Step 3: gather token features per Gaussian
        idx_for_tokens = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, D)
        token_feats = torch.gather(transformer_tokens, 1, idx_for_tokens)  # [B, N, D]

        # Step 4: per-Gaussian decode
        combined = torch.cat([token_feats, spatial_enc], dim=-1)  # [B, N, D+fourier_dim]
        BN = B * N
        raw = self.mlp(combined.reshape(BN, -1)).reshape(B, N, 14)

        # Step 5: apply activations (same as GS_decoder)
        pos   = raw[:, :, 0:3]
        color = (raw[:, :, 3:6] if self.color_residual
                 else torch.clamp(raw[:, :, 3:6], 0.0, 1.0))
        opac  = torch.sigmoid(raw[:, :, 6:7])
        scale = torch.exp(raw[:, :, 7:10])
        quat  = F.normalize(raw[:, :, 10:14], p=2, dim=-1)

        out = torch.cat([pos, color, opac, scale, quat], dim=-1)
        return out.reshape(B, -1)  # [B, N*14] — same format as GS_decoder


# ============================================================================
# PER-GAUSSIAN SEMANTIC HEADS (InfoNCE path — unchanged)
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
    Full Can3Tok VAE with all objectives and three new spatial inductive bias ideas.

    NEW FLAGS:
      decoder_pos_enc (bool, default False):
        Adds learnable positional embeddings PE[512, width] to decoder tokens
        before the transformer. Gives tokens structural identity so self-attention
        can learn spatially-organised communication patterns.

      predict_seg_labels (bool, default False):
        SegPredHead: gaussian_params [B,40000,14] -> seg_logits [B,40000,72].
        Cross-entropy loss forces decoder position outputs to be categorically
        discriminative. Inference: soft centroid lookup from seg_logits + pred_centroids.

      token_cond (bool, default False):
        Injects spatial conditioning into decoder tokens before transformer.

      token_cond_approach (str, default 'A'):
        'A'  — scaffold anchor Fourier encoding (requires position_scaffold=True)
        'B'  — soft category centroid (requires scene_layout_head=True)
        'AB' — both approaches applied additively

      query_decoder (bool, default False):
        Replaces flat GS_decoder MLP with SpatialAwareDecoder.
        Per-Gaussian: gather(token_feat) + Fourier(voxel_anchor) -> small MLP -> 14.
        Requires position_scaffold=True.
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
                 # ── NEW: three ideas ──────────────────────────────────────────
                 decoder_pos_enc=False,
                 predict_seg_labels=False,
                 token_cond=False,
                 token_cond_approach='A',
                 query_decoder=False):

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
        # ── New idea flags ──────────────────────────────────────────────────
        self.decoder_pos_enc_flag     = decoder_pos_enc
        self.predict_seg_labels_flag  = predict_seg_labels
        self.token_cond_flag          = token_cond
        self.token_cond_approach      = token_cond_approach.upper()
        self.query_decoder_flag       = query_decoder

        # Validate new flag dependencies
        if token_cond and 'A' in self.token_cond_approach and not position_scaffold:
            print("  [WARNING] token_cond approach A requires position_scaffold=True. "
                  "Approach A will be inactive unless scaffold data is provided.")
        if token_cond and 'B' in self.token_cond_approach and not scene_layout_head:
            print("  [WARNING] token_cond approach B requires scene_layout_head=True. "
                  "Approach B will be inactive until SceneLayoutHead is enabled.")
        if query_decoder and not position_scaffold:
            print("  [WARNING] query_decoder requires position_scaffold=True. "
                  "Query decoder will fall back to GS_decoder if scaffold data absent.")

        print(f"\n{'='*70}")
        print(f"  CAN3TOK")
        print(f"  semantic='{semantic_mode}' | color_residual={color_residual}")
        print(f"  scene_semantic_head={scene_semantic_head} | position_scaffold={position_scaffold}")
        print(f"  decoder_shape_prepend={decoder_shape_prepend} | "
              f"decoder_shape_cross_attn={decoder_shape_cross_attn}")
        print(f"  latent_disentangle={latent_disentangle}  semantic_dims={semantic_dims}")
        print(f"  scene_layout_head={scene_layout_head}  jepa_idea1={jepa_idea1}")
        print(f"  ── NEW IDEAS ──────────────────────────────────────────────────")
        print(f"  decoder_pos_enc:    {decoder_pos_enc}  (PE before transformer)")
        print(f"  predict_seg_labels: {predict_seg_labels}  (Idea 1: seg from Gaussian params)")
        print(f"  token_cond:         {token_cond}  approach={token_cond_approach}  "
              f"(Idea 2: spatial token conditioning)")
        print(f"  query_decoder:      {query_decoder}  (Idea 3: per-Gaussian spatial decoder)")
        print(f"{'='*70}")

        # ── EXISTING AUXILIARY HEADS ─────────────────────────────────────────
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

        self.scene_layout_module    = None
        self.last_scene_layout_pred = None
        if scene_layout_head:
            self.scene_layout_module = SceneLayoutHead(width=width)

        self.spatial_semantic_module    = None
        self.last_spatial_semantic_pred = None
        if jepa_idea1:
            if not position_scaffold:
                print(f"  [WARNING] jepa_idea1=True requires position_scaffold=True. Disabled.")
            else:
                self.spatial_semantic_module = SpatialSemanticHead(width=width, num_tokens=512)

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

        # ── NEW IDEA 0: POSITIONAL ENCODING ──────────────────────────────────
        self.decoder_pos_emb = None
        if decoder_pos_enc:
            # Learnable PE for 512 decoder tokens, initialised near zero
            # so it does not perturb existing training when loading from ckpt
            self.decoder_pos_emb = nn.Parameter(torch.zeros(512, width))
            nn.init.trunc_normal_(self.decoder_pos_emb, std=0.02)
            print(f"[DecoderPosEnc] learnable PE [512, {width}] — "
                  f"{512*width:,} params (init ~N(0, 0.02))")

        # ── NEW IDEA 1: SEGMENT PREDICTION HEAD ──────────────────────────────
        self.seg_pred_head = None
        self.last_seg_pred = None
        if predict_seg_labels:
            self.seg_pred_head = SegPredHead(in_dim=14, num_cats=72)

        # ── NEW IDEA 2: TOKEN CENTROID CONDITIONING ───────────────────────────
        self.token_cond_mlp_A      = None
        self.token_cond_mlp_B      = None
        self.token_cat_assign      = None
        fourier_out_dim = self.fourier_embedder.out_dim

        if token_cond:
            print(f"[TokenCond] approach='{token_cond_approach}' | "
                  f"fourier_dim={fourier_out_dim}")
            if 'A' in self.token_cond_approach:
                # Approach A: Fourier(scaffold_anchor) → token spatial bias
                self.token_cond_mlp_A = TokenCondMLP(fourier_out_dim, width)
                print(f"  Approach A: Fourier(scaffold_anchor[B,512,3]) -> bias[B,512,{width}]")

            if 'B' in self.token_cond_approach:
                # Approach B: learned token→category assignment × pred_centroids
                self.token_cat_assign = nn.Parameter(torch.zeros(512, 72))
                nn.init.trunc_normal_(self.token_cat_assign, std=0.01)
                self.token_cond_mlp_B = TokenCondMLP(fourier_out_dim, width)
                total_B = sum(p.numel() for p in self.token_cond_mlp_B.parameters()) + 512 * 72
                print(f"  Approach B: W[512,72] × pred_centroids[B,72,3] -> "
                      f"Fourier -> bias[B,512,{width}] | {total_B:,} params")
                print(f"    Requires scene_layout_head=True (uses last_scene_layout_pred)")

        # ── NEW IDEA 3: QUERY-BASED SPATIAL DECODER ───────────────────────────
        self.spatial_aware_decoder = None
        if query_decoder:
            self.spatial_aware_decoder = SpatialAwareDecoder(
                token_dim=width,
                fourier_embedder=self.fourier_embedder,
                hidden_dim=256,
                color_residual=color_residual)
            print(f"  Idea 3: GS_decoder MLP replaced by SpatialAwareDecoder")
            print(f"    Falls back to GS_decoder if scaffold data not provided")

        # ── PER-GAUSSIAN INFONCE HEADS (existing, unchanged) ─────────────────
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

        Args:
            latents:            [B, 512, 32]   — reshaped z from VAE
            scaffold_anchors:   [B, 512, 3]    — voxel centroids (for Idea 2A, 3)
            scaffold_token_ids: [B, 40000]     — per-Gaussian voxel assignment (for Idea 3)
            shape_embed:        [B, width]     — for decoder conditioning (existing)
        """
        latents = self.post_kl(latents)  # [B, 512, width]

        # ── [Existing] Option 2: shape cross-attention conditioning ───────────
        if (self.decoder_shape_cross_attn and
                self.shape_cross_attn_layers is not None and
                shape_embed is not None):
            shape_context = self.project_shape_for_cross_attn(shape_embed).unsqueeze(1)
            for cross_attn_layer in self.shape_cross_attn_layers:
                latents = cross_attn_layer(latents, shape_context)

        # ── NEW: Idea 0 — Positional Encoding ────────────────────────────────
        # Applied BEFORE transformer so attention can learn spatial structure.
        # PE is initialised near zero and learnable, safe to add to existing ckpts.
        if self.decoder_pos_emb is not None:
            latents = latents + self.decoder_pos_emb.unsqueeze(0)  # [B, 512, width]

        # ── NEW: Idea 2 — Token Centroid Conditioning ─────────────────────────
        # Applied BEFORE transformer so conditioned representations flow through
        # all 12 attention layers, maximising propagation of spatial context.
        if self.token_cond_flag:
            # Approach A: scaffold anchor Fourier encoding
            if ('A' in self.token_cond_approach and
                    self.token_cond_mlp_A is not None and
                    scaffold_anchors is not None):
                # scaffold_anchors: [B, 512, 3] → Fourier → [B, 512, fourier_dim]
                fourier_A    = self.fourier_embedder(scaffold_anchors)
                spatial_A    = self.token_cond_mlp_A(fourier_A)   # [B, 512, width]
                latents      = latents + spatial_A

            # Approach B: soft category centroid from SceneLayoutHead
            if ('B' in self.token_cond_approach and
                    self.token_cat_assign is not None and
                    self.token_cond_mlp_B is not None and
                    self.last_scene_layout_pred is not None):
                # W: [512, 72] soft assignment (scene-agnostic, learnable)
                W = F.softmax(self.token_cat_assign, dim=-1)        # [512, 72]
                pred_c = self.last_scene_layout_pred                 # [B, 72, 3]
                # token_centroids: [B, 512, 3]
                #   einsum 'tk,bkd->btd': for each batch b, for each token t,
                #   sum over categories k: W[t,k] * pred_c[b,k,:]
                token_centroids = torch.einsum('tk,bkd->btd', W, pred_c)
                fourier_B    = self.fourier_embedder(token_centroids)
                spatial_B    = self.token_cond_mlp_B(fourier_B)    # [B, 512, width]
                latents      = latents + spatial_B

        # ── [Existing] Option 1: shape prepend ───────────────────────────────
        shape_token_prepended = False
        if (self.decoder_shape_prepend and
                self.project_shape_for_prepend is not None and
                shape_embed is not None):
            shape_token = self.project_shape_for_prepend(shape_embed).unsqueeze(1)
            latents = torch.cat([shape_token, latents], dim=1)
            shape_token_prepended = True

        # ── Transformer ───────────────────────────────────────────────────────
        latents_out = self.transformer(latents)
        if shape_token_prepended:
            latents_out = latents_out[:, 1:, :]    # remove prepended shape token

        # ── NEW: Idea 3 — Spatial-Aware Decoder ──────────────────────────────
        # Uses per-Gaussian gather from transformer tokens + spatial Fourier encoding.
        # Falls back to GS_decoder if scaffold data unavailable.
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
            # Idea 3: per-Gaussian spatial decoder
            reconstruction = self.spatial_aware_decoder(
                latents_out, scaffold_anchors, scaffold_token_ids)
        else:
            # Standard GS_decoder (flat MLP)
            latents_flat = latents_out.reshape(latents_out.shape[0], -1)
            if need_hidden:
                reconstruction, hidden = self.GS_decoder(latents_flat, return_hidden=True)
            else:
                reconstruction = self.GS_decoder(latents_flat, return_hidden=False)

        # ── NEW: Idea 1 — Segment Prediction ─────────────────────────────────
        # Takes the 14-dim Gaussian params as input → lightweight per-Gaussian MLP.
        # Gradient flows back through Gaussian params into the decoder,
        # forcing position/color to be categorically discriminative.
        self.last_seg_pred = None
        if self.seg_pred_head is not None:
            B_r = reconstruction.shape[0]
            g_params = reconstruction.reshape(B_r, 40000, 14)
            self.last_seg_pred = self.seg_pred_head(g_params)  # [B, 40000, 72]

        # ── [Existing] Semantic features for InfoNCE ─────────────────────────
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

        NEW optional args:
          scaffold_anchors:   [B, 512, 3]  — needed for Idea 2A and Idea 3
          scaffold_token_ids: [B, 40000]   — needed for Idea 3

        NOTE: SceneLayoutHead computed BEFORE decode() so Approach B of
        Idea 2 can use pred_centroids inside the decode() call.
        """
        shape_embed, mu, log_var, z, posterior = self.encode(pc, feats, sample_posterior)
        _se = self._shape_embed_cache

        # ── SceneLayoutHead computed BEFORE decode() ─────────────────────────
        # Reason: Idea 2 Approach B needs pred_centroids inside decode() to
        # compute token_centroids. All other heads are computed after decode().
        self.last_scene_layout_pred = (
            self.scene_layout_module(_se)
            if self.scene_layout_module is not None else None)

        latents = z.reshape(z.shape[0], 512, 32)
        UV_gs_recover, per_gaussian_features = self.decode(
            latents, volume_queries,
            return_semantic_features=self.training,
            shape_embed=_se,
            scaffold_anchors=scaffold_anchors,
            scaffold_token_ids=scaffold_token_ids)

        # ── Remaining heads ───────────────────────────────────────────────────
        self.last_mean_color_pred = (
            self.mean_color_head(_se) if self.mean_color_head is not None else None)
        self.last_scene_semantic_pred = (
            self.scene_semantic_module(_se)
            if self.scene_semantic_module is not None else None)
        self.last_anchor_pred = (
            self.anchor_position_head(_se)
            if self.anchor_position_head is not None else None)
        # last_scene_layout_pred already set above
        self.last_spatial_semantic_pred = None

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features