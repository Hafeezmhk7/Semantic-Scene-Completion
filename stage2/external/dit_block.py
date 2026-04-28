# ---------------------------------------------------------------
# Building blocks extracted from DiT (Peebles & Xie, ICCV 2023)
# https://github.com/facebookresearch/DiT  —  Apache 2.0 License
#
# Changes from the original:
#   * Replaced timm.Attention / timm.Mlp with pure-PyTorch equivalents
#     so there is no timm dependency in Stage 2.
#   * Replaced the image-specific FinalLayer (patch_size * patch_size *
#     out_channels output) with TokenFinalLayer that outputs token_dim
#     directly — suitable for any sequence of latent tokens.
#   * Removed everything image-specific: LabelEmbedder, DiT, PatchEmbed,
#     2-D sinusoidal positional embedding utilities, and model configs.
#   * modulate() and TimestepEmbedder are kept verbatim from DiT.
# ---------------------------------------------------------------

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift.  [B,N,D] x [B,D] → [B,N,D]"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """
    Two-layer MLP with GELU activation.
    Replaces timm.models.vision_transformer.Mlp to avoid the dependency.
    """
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU(approximate="tanh")
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# TimestepEmbedder  (verbatim from DiT)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    t ∈ [0, 1]  →  sinusoidal frequencies  →  2-layer MLP  →  [B, hidden_size]
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Sinusoidal timestep embedding (from GLIDE / DiT).
        t : [B,] float tensor of timesteps in [0, 1]
        returns [B, dim]
        """
        half   = dim // 2
        freqs  = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args      = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


# ---------------------------------------------------------------------------
# DiTBlock  (adapted: timm.Attention → nn.MultiheadAttention)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Source: DiT (Peebles & Xie, 2023), adapted to use nn.MultiheadAttention
    instead of timm.Attention so the block has no external model dependencies.

    forward(x, c):
        x  [B, N, D]  token sequence
        c  [B, D]     conditioning vector (timestep embedding)
    returns [B, N, D]
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        # adaLN-Zero: output zero-initialised → each block starts as identity
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        # Self-attention
        x_mod         = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _   = self.attn(x_mod, x_mod, x_mod)
        x             = x + gate_msa.unsqueeze(1) * attn_out
        # Feed-forward
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ---------------------------------------------------------------------------
# TokenFinalLayer  (replaces image-specific FinalLayer)
# ---------------------------------------------------------------------------

class TokenFinalLayer(nn.Module):
    """
    Final layer for token-based DiT models.

    Replaces DiT's FinalLayer which outputs patch_size² × out_channels per token
    (image-specific). This version projects each token [D] → [token_dim] with
    AdaLN modulation, suitable for any latent token sequence.

    forward(x, c):
        x  [B, N, D]     token sequence after all DiT blocks
        c  [B, D]        conditioning vector
    returns [B, N, token_dim]
    """
    def __init__(self, hidden_size: int, token_dim: int):
        super().__init__()
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear           = nn.Linear(hidden_size, token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


# ---------------------------------------------------------------------------
# Weight initialisation (mirrors DiT's initialize_weights)
# ---------------------------------------------------------------------------

def init_dit_weights(model: nn.Module) -> None:
    """
    Initialise weights following the DiT recipe:
      • Linear layers: Xavier uniform, bias = 0
      • adaLN_modulation final Linear: weights = 0, bias = 0  (adaLN-Zero trick)
      • final_layer.linear: weights = 0, bias = 0
      • TimestepEmbedder MLP linears: normal(std=0.02)
    Call after constructing the model, before training.
    """
    def _basic_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(_basic_init)

    # Zero-init adaLN-Zero modulation outputs
    for m in model.modules():
        if hasattr(m, 'adaLN_modulation') and isinstance(m.adaLN_modulation, nn.Sequential):
            last = m.adaLN_modulation[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.weight, 0)
                nn.init.constant_(last.bias, 0)

    # Zero-init final layer output projection
    for name, m in model.named_modules():
        if 'final_layer' in name and isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    # Normal init for TimestepEmbedder MLP
    for m in model.modules():
        if isinstance(m, TimestepEmbedder):
            nn.init.normal_(m.mlp[0].weight, std=0.02)
            nn.init.normal_(m.mlp[2].weight, std=0.02)