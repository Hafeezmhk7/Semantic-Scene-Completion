# -*- coding: utf-8 -*-
"""
sal_perceiver.py - Can3Tok FULL IMPLEMENTATION with Dual Fourier Embeddings
============================================================================
CHANGED: Output 14 parameters instead of 11
- xyz (3) + rgb (3) + opacity (1) + scale (3) + rotation (4) = 14 params

CAN3TOK APPROACH (from sal_perceiver_try.py):
----------------------------------------------
Dual spatial encoding in the encoder:
1. Voxel centers → Fourier embedding → coarse spatial structure
2. Actual xyz → Fourier embedding → fine-grained positions
3. Concatenate: [fourier_voxel, fourier_xyz, gaussian_params] → input_proj

This provides hierarchical spatial understanding without decoder residuals.
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
    Transformer
)

from .tsal_base import ShapeAsLatentModule


# ============================================================================
# SEMANTIC PROJECTION HEADS
# ============================================================================

class SemanticProjectionHead(nn.Module):
    """Hidden state projection"""
    def __init__(self, hidden_dim=1024, num_gaussians=40000, feature_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_gaussians * feature_dim),
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHead] Hidden State → MLP:")
        print(f"  Input: [B, {hidden_dim}]")
        print(f"  Output: [B, {num_gaussians}, {feature_dim}]")
        print(f"  Parameters: {total_params / 1e6:.3f}M")
    
    def forward(self, hidden):
        B = hidden.shape[0]
        expanded = self.projection(hidden)
        features = expanded.reshape(B, self.num_gaussians, self.feature_dim)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features


class SemanticProjectionHeadGeometric(nn.Module):
    """Geometric projection - handles 14 params (with RGB)"""
    def __init__(self, gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.gaussian_dim = gaussian_dim
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.projection = nn.Sequential(
            nn.Linear(gaussian_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[SemanticProjectionHeadGeometric] Geometric → MLP:")
        print(f"  Input: [B, {num_gaussians}, {gaussian_dim}] (WITH RGB!)")
        print(f"  Output: [B, {num_gaussians}, {feature_dim}]")
        print(f"  Parameters: {total_params / 1e6:.3f}M")
    
    def forward(self, gaussians):
        B, N, D = gaussians.shape
        gaussians_flat = gaussians.reshape(B * N, D)
        features_flat = self.projection(gaussians_flat)
        features = features_flat.reshape(B, N, self.feature_dim)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features


# ============================================================================
# ENCODER - CAN3TOK FULL VERSION WITH DUAL FOURIER EMBEDDINGS
# ============================================================================
class CrossAttentionEncoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 fourier_embedder_ID: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        # Initialize learnable queries with coarse voxel structure
        voxel_reso = 4
        x_y = np.linspace(-8, 8, voxel_reso)
        xv, yv, zv = np.meshgrid(x_y, x_y, x_y, indexing='ij')
        voxel_centers = torch.tensor(
            np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T,
            device=device,
            dtype=dtype
        ).reshape([-1, 3])

        dummy_tensor2 = torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02
        dummy_tensor2[:, :192] = voxel_centers.reshape([-1]) * 0.01
        self.query = nn.Parameter(dummy_tensor2)
        
        self.point_feats = point_feats
        self.fourier_embedder = fourier_embedder  # For xyz positions
        self.fourier_embedder_ID = fourier_embedder_ID  # For voxel centers

        # ═══════════════════════════════════════════════════════════════
        # CAN3TOK FULL VERSION: DUAL FOURIER EMBEDDINGS
        # ═══════════════════════════════════════════════════════════════
        # Input = fourier(voxel_centers) + fourier(xyz) + gaussian_params
        self.input_proj = nn.Linear(
            self.fourier_embedder.out_dim + point_feats + self.fourier_embedder_ID.out_dim,
            #                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #                                              Added voxel Fourier dimension!
            width,
            device=device,
            dtype=dtype
        )

        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """
        CAN3TOK encoder with dual Fourier embeddings.
        
        Args:
            pc: [B, N, 18] - Full Gaussian data
                [:, :, 0:3] = voxel_centers
                [:, :, 3] = voxel_id
                [:, :, 4:7] = xyz positions
                [:, :, 7:] = other parameters
            feats: Same as pc
        """
        bs = pc.shape[0]
        
        # Extract spatial features
        voxel_centers = pc[:, :, 0:3]  # Coarse spatial anchor
        xyz_actual = pc[:, :, 4:7]     # Fine position
        gaussian_params = feats[:, :, 7:]  # rgb, opacity, scale, rot
        
        # ═══════════════════════════════════════════════════════════════
        # DUAL FOURIER EMBEDDINGS (CAN3TOK)
        # ═══════════════════════════════════════════════════════════════
        voxel_coords_emb = self.fourier_embedder_ID(voxel_centers)  # Coarse
        xyz_emb = self.fourier_embedder(xyz_actual)                  # Fine
        
        # Concatenate all features
        data = torch.cat([
            xyz_emb,             # Fine-grained position encoding
            voxel_coords_emb,    # Coarse spatial structure
            gaussian_params      # Gaussian parameters
        ], dim=-1).to(dtype=torch.float32)
        
        # Project to attention feature space
        data = self.input_proj(data)
        
        # Cross-attention + self-attention
        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


# ============================================================================
# DECODER
# ============================================================================
class CrossAttentionDecoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(
            self.fourier_embedder.out_dim,
            width,
            device=device,
            dtype=dtype
        )

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class GaussianSemanticAttentionHead(CrossAttentionDecoder):
    """Cross-attention semantic head"""
    
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False):
        
        super().__init__(
            device=device,
            dtype=dtype,
            num_latents=num_latents,
            out_channels=out_channels,
            fourier_embedder=fourier_embedder,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[GaussianSemanticAttentionHead] Cross-Attention:")
        print(f"  Queries: [B, 40k, 3] Gaussian positions")
        print(f"  Keys/Values: [B, {num_latents}, {width}] scene tokens")
        print(f"  Output: [B, 40k, {out_channels}]")
        print(f"  Parameters: {total_params / 1e6:.3f}M")
    
    def forward(self, gaussian_xyz, scene_tokens):
        features = super().forward(gaussian_xyz, scene_tokens)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features


# ============================================================================
# GS_DECODER - WITH ACTIVATION FUNCTIONS
# ============================================================================
class GS_decoder(nn.Module):
    """
    MLP decoder for Gaussian parameters WITH RGB COLOR.
    
    Output: 14 geometric params (xyz, rgb, opacity, scale, quat)
    ALL parameters are in POST-ACTIVATION format:
      - Color: sigmoid → [0, 1]
      - Opacity: sigmoid → [0, 1]
      - Scale: softplus → (0, +∞)
      - Quaternion: normalized → ||q|| = 1.0
    """
    def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56):
        super(GS_decoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skip
        self.output_ch = output_ch
        
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)])
        for i in range(D-1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.LayerNorm(W))
            self.pts_linears.append(nn.ReLU())
        
        self.output_linear = nn.Linear(in_features=W, out_features=output_ch)
        
    def forward(self, x, return_hidden=False):
        """
        Forward pass with proper activations to match dataset format.
        
        Args:
            x: [B, input_ch] latent features
            return_hidden: whether to return hidden features for semantic learning
            
        Returns:
            output: [B, N*14] Gaussian parameters in POST-ACTIVATION format
            hidden: (optional) [B, W] hidden features before output projection
        """
        # MLP forward
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
        
        hidden = x
        raw_output = self.output_linear(x)
        
        # Reshape to [B, N, 14]
        B = raw_output.shape[0]
        N = 40000
        raw_output = raw_output.reshape(B, N, 14)
        
        # Split into parameters
        position    = raw_output[:, :, 0:3]
        color_raw   = raw_output[:, :, 3:6]
        opacity_raw = raw_output[:, :, 6:7]
        scale_raw   = raw_output[:, :, 7:10]
        quat_raw    = raw_output[:, :, 10:14]
        
        # Apply activations
        color = torch.sigmoid(color_raw)
        opacity = torch.sigmoid(opacity_raw)
        scale = F.softplus(scale_raw) + 1e-7
        quat = F.normalize(quat_raw, p=2, dim=-1)
        
        # Recombine
        output = torch.cat([
            position,  # [B, N, 3] - absolute positions (NO residual)
            color,     # [B, N, 3]
            opacity,   # [B, N, 1]
            scale,     # [B, N, 3]
            quat,      # [B, N, 4]
        ], dim=-1)
        
        # Flatten back to [B, N*14]
        output = output.reshape(B, -1)
        
        if return_hidden:
            return output, hidden
        else:
            return output


# ============================================================================
# SHAPE AS LATENT PERCEIVER (base class)
# ============================================================================
class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = True,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        
        self.fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs,
            include_pi=include_pi,
            input_dim=3
        )
        self.fourier_embedder_ID = FourierEmbedder(
            num_freqs=num_freqs,
            include_pi=include_pi,
            input_dim=3
        )

        init_scale = init_scale * math.sqrt(1.0 / width)
        
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            fourier_embedder_ID=self.fourier_embedder_ID,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.embed_dim = embed_dim
        if embed_dim > 0:
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )
        
        print(f"\n⚠️  GS_DECODER OUTPUT: 40000 × 14 = 560,000 (WITH RGB COLOR!)")
        self.GS_decoder = GS_decoder(3, 1024, width*512, [4], 40000*14)
        
        self.kl_emb_proj_mean = nn.Linear((num_latents-1)*embed_dim, 64*64*4, dtype=dtype)
        self.kl_emb_proj_var = nn.Linear((num_latents-1)*embed_dim, 64*64*4, dtype=dtype)
        
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

    def encode(self, pc, feats=None, sample_posterior=True):
        latents, center_pos = self.encoder(pc, feats)
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            latents = posterior.sample() if sample_posterior else posterior.mode()
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
# ALIGNED SHAPE LATENT PERCEIVER - NO RESIDUAL FIX
# ============================================================================
class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):
    def __init__(self, *,
                device: Optional[torch.device],
                dtype: Optional[torch.dtype],
                num_latents: int,
                point_feats: int = 0,
                embed_dim: int = 0,
                num_freqs: int = 8,
                include_pi: bool = True,
                width: int,
                heads: int,
                num_encoder_layers: int,
                num_decoder_layers: int,
                init_scale: float = 0.25,
                qkv_bias: bool = True,
                flash: bool = True,
                use_ln_post: bool = False,
                use_checkpoint: bool = False,
                semantic_mode: str = 'none'):

        super().__init__(
            device=device,
            dtype=dtype,
            num_latents=1 + num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=include_pi,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.width = width
        self.semantic_mode = semantic_mode
        
        print(f"\n{'='*70}")
        print(f"🚀 CAN3TOK FULL IMPLEMENTATION - DUAL FOURIER EMBEDDINGS")
        print(f"[AlignedShapeLatentPerceiver] semantic_mode='{semantic_mode}'")
        print(f"{'='*70}")
        
        self.semantic_projection_hidden = None
        self.semantic_projection_geometric = None
        self.semantic_attention_head = None
        
        if semantic_mode == 'hidden':
            self.semantic_projection_hidden = SemanticProjectionHead(
                hidden_dim=1024, num_gaussians=40000, feature_dim=32
            )
            print(f"✓ Initialized HIDDEN STATE projection head ONLY")
            
        elif semantic_mode == 'geometric':
            self.semantic_projection_geometric = SemanticProjectionHeadGeometric(
                gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128
            )
            print(f"✓ Initialized GEOMETRIC projection head ONLY (WITH RGB!)")
            
        elif semantic_mode == 'attention':
            self.semantic_attention_head = GaussianSemanticAttentionHead(
                device=device, dtype=dtype, num_latents=num_latents,
                out_channels=32, fourier_embedder=self.fourier_embedder,
                width=width, heads=heads, init_scale=init_scale,
                qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint
            )
            print(f"✓ Initialized CROSS-ATTENTION semantic head ONLY")
        
        elif semantic_mode == 'none':
            print(f"✓ NO semantic head initialized (pure VAE mode)")
            print(f"✓ Maximum memory savings!")
        else:
            raise ValueError(f"Unknown semantic_mode: {semantic_mode}")
        
        print(f"{'='*70}")
        print(f"✓ Encoder: Dual Fourier embeddings (voxel + xyz)")
        print(f"✓ Decoder: Direct MLP output (NO residual)")
        print(f"{'='*70}\n")

    def encode(self, pc, feats=None, sample_posterior=True):
        shape_embed, latents = self.encode_latents(pc, feats)
        kl_embed, posterior = self.encode_kl_embed(latents, sample_posterior)
        
        kl_embed = kl_embed.reshape([kl_embed.shape[0], -1])
        mu = self.kl_emb_proj_mean(kl_embed)
        log_var = self.kl_emb_proj_var(kl_embed)
        
        std = torch.exp(0.5 * log_var).to(log_var.device)
        eps = torch.randn_like(std).to(std.device)
        z = mu + std * eps.to(std.device)
        
        return shape_embed, mu, log_var, z, posterior

    def decode(self, latents, volume_queries=None, return_semantic_features=False):
        """
        CAN3TOK decoder - NO residual position fix.
        
        The dual Fourier embeddings in the encoder provide spatial awareness.
        The decoder outputs absolute positions directly.
        """
        latents = self.post_kl(latents)
        latents_transformed = self.transformer(latents)
        latents_flat = latents_transformed.reshape(latents_transformed.shape[0], -1)
        
        should_compute_semantic = (
            return_semantic_features and
            self.training and
            (self.semantic_projection_hidden is not None or
             self.semantic_projection_geometric is not None or
             self.semantic_attention_head is not None)
        )
        
        if should_compute_semantic:
            reconstruction, hidden = self.GS_decoder(latents_flat, return_hidden=True)
        else:
            reconstruction = self.GS_decoder(latents_flat, return_hidden=False)

        # ═══════════════════════════════════════════════════════════════
        # NO RESIDUAL FIX - Direct output from decoder
        # ═══════════════════════════════════════════════════════════════
        # The model learns absolute positions through the dual Fourier
        # embeddings in the encoder (voxel + xyz).
        # No residual addition needed.
        # ═══════════════════════════════════════════════════════════════

        if should_compute_semantic:
            B = reconstruction.shape[0]
            reconstruction_gaussians = reconstruction.reshape(B, 40000, 14)

            if self.semantic_mode == 'hidden' and self.semantic_projection_hidden is not None:
                semantic_features = self.semantic_projection_hidden(hidden)
            elif self.semantic_mode == 'geometric' and self.semantic_projection_geometric is not None:
                semantic_features = self.semantic_projection_geometric(reconstruction_gaussians)
            elif self.semantic_mode == 'attention' and self.semantic_attention_head is not None:
                gaussian_positions = reconstruction_gaussians[:, :, 0:3]
                semantic_features = self.semantic_attention_head(
                    gaussian_positions, latents_transformed
                )
            else:
                semantic_features = None
            
            return reconstruction, semantic_features
        else:
            return reconstruction

    def encode_latents(self, pc, feats=None):
        x, _ = self.encoder(pc, feats)
        shape_embed = x[:, 0]
        latents = x[:, 1:]
        return shape_embed, latents

    def encode_kl_embed(self, latents, sample_posterior=True):
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            kl_embed = posterior.sample() if sample_posterior else posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior

    def forward(self, pc, feats, volume_queries, sample_posterior=True):
        shape_embed, mu, log_var, z, posterior = self.encode(
            pc, feats, sample_posterior=sample_posterior
        )

        latents_reshaped = z.reshape(z.shape[0], 512, 32)
        
        has_semantic = (
            self.semantic_projection_hidden is not None or
            self.semantic_projection_geometric is not None or
            self.semantic_attention_head is not None
        )

        if self.training and has_semantic:
            UV_gs_recover, per_gaussian_features = self.decode(
                latents_reshaped, volume_queries, return_semantic_features=True
            )
        else:
            UV_gs_recover = self.decode(
                latents_reshaped, volume_queries, return_semantic_features=False
            )
            per_gaussian_features = None
        
        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features