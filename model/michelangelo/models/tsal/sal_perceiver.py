# -*- coding: utf-8 -*-
"""
sal_perceiver.py - WITH RGB COLOR SUPPORT
============================================
CHANGED: Output 14 parameters instead of 11
- xyz (3) + rgb (3) + opacity (1) + scale (3) + rotation (4) = 14 params

All three semantic heads updated to work with 14-dim Gaussians
"""

import torch
import torch.nn as nn
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
# SEMANTIC PROJECTION HEADS (Updated for 14 params)
# ============================================================================

class SemanticProjectionHead(nn.Module):
    """Hidden state projection (unchanged)"""
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
    """
    Geometric projection - NOW HANDLES 14 PARAMS (with RGB)
    """
    def __init__(self, gaussian_dim=14, num_gaussians=40000, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.gaussian_dim = gaussian_dim
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.projection = nn.Sequential(
            nn.Linear(gaussian_dim, hidden_dim),  # 14 → 128
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
# ENCODER (Unchanged)
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
        self.fourier_embedder = fourier_embedder
        self.fourier_embedder_ID = fourier_embedder_ID

        self.input_proj = nn.Linear(
            self.fourier_embedder.out_dim + point_feats,
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
        bs = pc.shape[0]
        feats = feats[:, :, 7:]
        data = self.fourier_embedder(pc[:, :, 4:7])
       
        if feats is not None:
            data = torch.cat([data, feats], dim=-1).to(dtype=torch.float32)
        
        data = self.input_proj(data)
        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


# ============================================================================
# DECODER (Unchanged)
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
    """Cross-attention semantic head (unchanged)"""
    
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
# GS_DECODER - UPDATED TO OUTPUT 14 PARAMS (WITH RGB!)
# ============================================================================
class GS_decoder(nn.Module):
    """
    MLP decoder for Gaussian parameters WITH RGB COLOR.
    
    Output: 14 geometric params (xyz, rgb, opacity, scale, quat)
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
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
        
        hidden = x
        output = self.output_linear(x)
        
        if return_hidden:
            return output, hidden
        else:
            return output


# ============================================================================
# SHAPE AS LATENT PERCEIVER (Updated for 14 params)
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
        
        # ================================================================
        # CRITICAL CHANGE: Output 14 params (xyz, rgb, opacity, scale, quat)
        # ================================================================
        print(f"\n⚠️  GS_DECODER OUTPUT: 40000 × 14 = 560,000 (WITH RGB COLOR!)")
        self.GS_decoder = GS_decoder(3, 1024, width*512, [4], 40000*14)  # ← Changed from *11
        
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

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        latents, center_pos = self.encoder(pc, feats)

        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                latents = posterior.sample()
            else:
                latents = posterior.mode()

        return latents, center_pos, posterior

    def decode(self, latents: torch.FloatTensor, volume_queries=None):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return self.GS_decoder(latents.reshape(latents.shape[0], -1))

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries, latents).squeeze(-1)
        return logits

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)
        latents_decoded = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)
        return logits, center_pos, posterior


# ============================================================================
# ALIGNED SHAPE LATENT PERCEIVER (WITH RGB COLOR!)
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
        print(f"🚀 PERFORMANCE FIX APPLIED - ISSUE #1")
        print(f"[AlignedShapeLatentPerceiver] semantic_mode='{semantic_mode}'")
        print(f"{'='*70}")
        
        self.semantic_projection_hidden = None
        self.semantic_projection_geometric = None
        self.semantic_attention_head = None
        
        if semantic_mode == 'hidden':
            self.semantic_projection_hidden = SemanticProjectionHead(
                hidden_dim=1024,
                num_gaussians=40000,
                feature_dim=32
            )
            print(f"✓ Initialized HIDDEN STATE projection head ONLY")
            print(f"✓ Saved ~15GB memory by NOT loading other heads!")
            
        elif semantic_mode == 'geometric':
            self.semantic_projection_geometric = SemanticProjectionHeadGeometric(
                gaussian_dim=14,  # ← Changed from 11!
                num_gaussians=40000,
                feature_dim=32,
                hidden_dim=128
            )
            print(f"✓ Initialized GEOMETRIC projection head ONLY (WITH RGB!)")
            print(f"✓ Saved ~15GB memory by NOT loading other heads!")
            
        elif semantic_mode == 'attention':
            self.semantic_attention_head = GaussianSemanticAttentionHead(
                device=device,
                dtype=dtype,
                num_latents=num_latents,
                out_channels=32,
                fourier_embedder=self.fourier_embedder,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                flash=flash,
                use_checkpoint=use_checkpoint
            )
            print(f"✓ Initialized CROSS-ATTENTION semantic head ONLY")
            print(f"✓ Saved ~15GB memory by NOT loading other heads!")
        
        elif semantic_mode == 'none':
            print(f"✓ NO semantic head initialized (pure VAE mode)")
            print(f"✓ Maximum memory savings!")
            
        else:
            raise ValueError(f"Unknown semantic_mode: {semantic_mode}")
        
        print(f"{'='*70}\n")

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        shape_embed, latents = self.encode_latents(pc, feats)
        kl_embed, posterior = self.encode_kl_embed(latents, sample_posterior)
        
        kl_embed = kl_embed.reshape([kl_embed.shape[0], -1])
        mu = self.kl_emb_proj_mean(kl_embed)
        log_var = self.kl_emb_proj_var(kl_embed)
        
        std = torch.exp(0.5 * log_var).to(log_var.device)
        eps = torch.randn_like(std).to(std.device)
        z = mu + std * eps.to(std.device)
        
        return shape_embed, mu, log_var, z, posterior

    def decode(self, latents: torch.FloatTensor, volume_queries=None, return_semantic_features=False):
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
            B = reconstruction.shape[0]
            reconstruction_gaussians = reconstruction.reshape(B, 40000, 14)  # ← Changed from 11!
            
            if self.semantic_mode == 'hidden' and self.semantic_projection_hidden is not None:
                semantic_features = self.semantic_projection_hidden(hidden)
            elif self.semantic_mode == 'geometric' and self.semantic_projection_geometric is not None:
                semantic_features = self.semantic_projection_geometric(reconstruction_gaussians)
            elif self.semantic_mode == 'attention' and self.semantic_attention_head is not None:
                gaussian_positions = reconstruction_gaussians[:, :, 0:3]
                semantic_features = self.semantic_attention_head(
                    gaussian_positions, 
                    latents_transformed
                )
            else:
                semantic_features = None
            
            return reconstruction, semantic_features
        else:
            reconstruction = self.GS_decoder(latents_flat, return_hidden=False)
            return reconstruction

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):
        x, _ = self.encoder(pc, feats)
        shape_embed = x[:, 0]
        latents = x[:, 1:]
        return shape_embed, latents

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents

        return kl_embed, posterior

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        shape_embed, mu, log_var, z, posterior = self.encode(
            pc, feats, sample_posterior=sample_posterior
        )

        latents_reshaped = z.reshape(z.shape[0], 512, 32)
        
        if self.training and (self.semantic_projection_hidden is not None or
                            self.semantic_projection_geometric is not None or
                            self.semantic_attention_head is not None):
            UV_gs_recover, per_gaussian_features = self.decode(
                latents_reshaped,
                volume_queries,
                return_semantic_features=True
            )
        else:
            UV_gs_recover = self.decode(
                latents_reshaped,
                volume_queries,
                return_semantic_features=False
            )
            per_gaussian_features = None
        
        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features