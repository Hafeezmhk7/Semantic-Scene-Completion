"""
stage2/models/flat_dit.py
==========================
Stage 2 models for the FLAT-latent Stage 1 experiments (1-5), i.e. checkpoints
with local_disentangle=False, whose encoder produces a SINGLE latent Z [B,512,32]
and NO separate z_s.

  SceneDiT             unconditional generation of the full latent Z [B,512,32].
                       Objective 1 for experiments 1-5. For experiments 1-4 the
                       512 tokens are a global mixture; for experiment 5 (local
                       encoder) they are Hilbert blocks. Either way Stage 2 sees a
                       [512,32] tensor and learns P(Z) directly with one DiT.

  CompletionDiTUncond  scene completion for the STRUCTURED flat case (experiment 5
                       only), where token k is a spatial Hilbert block, so masking
                       tokens masks regions. Self-attention only, no layout
                       conditioning. (Completion is NOT meaningful for the global
                       flat experiments 1-4 and is rejected upstream.)

SceneDiT is LayoutDiT widened from 16 to 512 tokens; CompletionDiTUncond is the
B1 CompletionDiT with the cross-attention removed. Both reuse the shared DiT
building blocks and support the same learned_ape / 1d / 3d positional options.
"""

import random
import torch
import torch.nn as nn

from ..external.dit_block import (
    DiTBlock, TimestepEmbedder, TokenFinalLayer, init_dit_weights,
)

N_Z       = 512    # full latent token count (matches Stage 1 _N_LATENT_TOKENS)
TOKEN_DIM = 32     # embed_dim
_VALID_ROPE = ('learned_ape', '1d', '3d')


def _make_rope(rope_type: str, head_dim: int, n_tokens: int):
    """Return a single self-attention RoPE module (or None for learned_ape)."""
    if rope_type == 'learned_ape':
        return None
    from ..external.rope import RoPE1D, RoPE3D
    if rope_type == '1d':
        return RoPE1D(head_dim=head_dim, max_seq_len=n_tokens)
    if rope_type == '3d':
        # 512 tokens fill the 8x8x8 grid exactly. NOTE: that grid is row-major,
        # while structured tokens are Hilbert-ordered, so 3D RoPE here is an
        # approximation of spatial proximity (see report notes).
        return RoPE3D(head_dim=head_dim, seq_len=n_tokens)
    raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")


# ============================================================================
# SceneDiT  —  unconditional generation of the full latent
# ============================================================================

class SceneDiT(nn.Module):
    """
    Single DiT over the full latent Z [B, n_tokens, 32]. The only conditioning is
    the flow-matching timestep (adaLN-Zero). There is no z_s and no cross-attention.

    forward(x, t) -> velocity [B, n_tokens, 32]
    """

    def __init__(self, hidden_size: int = 384, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, n_tokens: int = N_Z,
                 rope_type: str = 'learned_ape'):
        super().__init__()
        if rope_type not in _VALID_ROPE:
            raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")
        self.hidden_size = hidden_size
        self.n_tokens    = n_tokens
        self.rope_type   = rope_type

        self.token_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.t_embedder     = TimestepEmbedder(hidden_size)

        head_dim    = hidden_size // num_heads
        rope_module = _make_rope(rope_type, head_dim, n_tokens)
        if rope_type == 'learned_ape':
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, hidden_size))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_module('rope', rope_module)
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     rope_type=rope_type, rope_module=rope_module)
            for _ in range(depth)
        ])
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)
        init_dit_weights(self)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.token_embedder(x)
        if self.pos_embed is not None:
            h = h + self.pos_embed
        c = self.t_embedder(t)
        for block in self.blocks:
            h = block(h, c)
        return self.final_layer(h, c)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (f"SceneDiT(tokens={self.n_tokens}, rope={self.rope_type}, "
                f"hidden={self.hidden_size}, depth={len(self.blocks)}, "
                f"params={self.num_params()/1e6:.2f}M)")


# ============================================================================
# CompletionDiTUncond  —  completion for the structured flat case (exp 5)
# ============================================================================

class CompletionDiTUncond(nn.Module):
    """
    Inpainting DiT with NO layout conditioning. Mask a fraction of the 512 spatial
    tokens, replace them with a learned mask token, denoise the unobserved tokens
    using self-attention only.

    forward(x, t, obs_mask) -> velocity [B, 512, 32]
    """

    def __init__(self, hidden_size: int = 384, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, n_tokens: int = N_Z,
                 rope_type: str = 'learned_ape'):
        super().__init__()
        if rope_type not in _VALID_ROPE:
            raise ValueError(f"rope_type must be one of {_VALID_ROPE}, got '{rope_type}'")
        self.hidden_size = hidden_size
        self.n_tokens    = n_tokens
        self.rope_type   = rope_type

        self.token_embedder = nn.Linear(TOKEN_DIM, hidden_size)
        self.mask_token     = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.t_embedder     = TimestepEmbedder(hidden_size)

        head_dim    = hidden_size // num_heads
        rope_module = _make_rope(rope_type, head_dim, n_tokens)
        if rope_type == 'learned_ape':
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, hidden_size))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_module('rope', rope_module)
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     rope_type=rope_type, rope_module=rope_module)
            for _ in range(depth)
        ])
        self.final_layer = TokenFinalLayer(hidden_size, TOKEN_DIM)
        init_dit_weights(self)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                obs_mask: torch.Tensor) -> torch.Tensor:
        h   = self.token_embedder(x)
        msk = obs_mask.unsqueeze(-1).float()              # [B, N, 1]
        h   = h * msk + self.mask_token * (1.0 - msk)     # unobserved -> mask token
        if self.pos_embed is not None:
            h = h + self.pos_embed
        c = self.t_embedder(t)
        for block in self.blocks:
            h = block(h, c)
        return self.final_layer(h, c)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (f"CompletionDiTUncond(tokens={self.n_tokens}, rope={self.rope_type}, "
                f"hidden={self.hidden_size}, depth={len(self.blocks)}, "
                f"params={self.num_params()/1e6:.2f}M)")


# ============================================================================
# Masking + training step for the unconditioned completion
# ============================================================================

def sample_block_mask(B: int, n_tokens: int = N_Z, device=None,
                      coverage_range: tuple = (0.3, 0.8)) -> torch.Tensor:
    """
    Random binary observed-mask over tokens (same logic as B1's sample_voxel_mask,
    renamed for the structured-token setting). 1=observed (held fixed),
    0=unobserved (denoise). For experiment 5 each token is a Hilbert block, so this
    is masking spatial regions.
    """
    masks = []
    for _ in range(B):
        cov   = random.uniform(*coverage_range)
        n_obs = int(n_tokens * cov)
        perm  = torch.randperm(n_tokens, device=device)
        m     = torch.zeros(n_tokens, device=device)
        m[perm[:n_obs]] = 1.0
        masks.append(m)
    return torch.stack(masks)


def completion_training_step_uncond(model, z_clean: torch.Tensor, path_sampler,
                                    n_tokens: int = N_Z,
                                    coverage_range: tuple = (0.3, 0.8)) -> torch.Tensor:
    """
    One completion training step WITHOUT layout conditioning (experiment 5).

    z_clean : [B, 512, 32]  clean Stage 1 latent (encoder mode)
    Masks 30-80% of tokens as observed, applies flow-matching noise to the
    unobserved tokens only, and returns the MSE on the unobserved positions.
    path_sampler is accepted for interface symmetry with the conditioned step
    (the linear path is computed inline).
    """
    B, device = z_clean.shape[0], z_clean.device
    obs_mask  = sample_block_mask(B, n_tokens, device, coverage_range)   # [B, N]
    mask_exp  = obs_mask.unsqueeze(-1)                                   # [B, N, 1]

    t     = torch.rand(B, device=device)
    z0    = torch.randn_like(z_clean)
    t_exp = t.view(B, 1, 1)
    z_t   = t_exp * z_clean + (1.0 - t_exp) * z0
    z_t   = z_t * (1.0 - mask_exp) + z_clean * mask_exp                  # observed = clean

    v_target = z_clean - z0
    v_pred   = model(z_t, t, obs_mask)
    loss = ((v_pred - v_target) ** 2 * (1.0 - mask_exp)).sum()
    loss = loss / ((1.0 - mask_exp).sum() + 1e-8)
    return loss


# ============================================================================
# DCHead  —  recover the DC (mean) colour for FLAT generation
# ============================================================================

class DCHead(nn.Module):
    """
    Conditional DC (mean colour) model for the FLAT generation case (experiments
    1-5), where Stage 1's mean_color_head reads shape_embed, which is NOT part of
    the latent Z that SceneDiT generates. There is therefore no shape_embed at
    generation-from-noise to recover the per-scene DC colour, and Stage 1 cannot be
    retrained.

    This head instead models q(DC | Z) on the generatable latent, trained on
    (Z_encoder mode, mean_color_GT) pairs with the frozen Stage 1 encoder. No Stage
    1 change. mean_color_GT (the per-scene RGB the decoder's AC residual was defined
    against) is the target, so DC_pred + AC_decoder reconstructs absolute colour.

    Parameterisation: a diagonal Gaussian over the LOGIT of the RGB DC, predicted
    from a permutation-invariant pool of Z (mean + std over the 512 tokens):
        mu, logvar = head(pool(Z))                  # logit space, 3 dims each
        DC = sigmoid(mu + exp(0.5 logvar) * eps)

    Why a conditional Gaussian rather than a point regressor: it is automatically
    correct in both regimes, which matters because whether Z carries DC information
    depends on whether the dataset mean-subtracts colour before the encoder.
      * Z carries DC info  -> mu tracks it, logvar small: palette coupled to the
        generated geometry (the conditional mean E[DC|Z], MMSE-optimal).
      * Z is DC-invariant  -> mu collapses to the global mean, logvar grows to the
        marginal variance: sampling recovers the empirical palette P(DC).
    A deterministic regressor would collapse to the global mean in the second
    regime and silently destroy palette diversity.

    forward(z)              -> (mu, logvar)   in logit space
    forward(z, target)      -> Gaussian NLL   (scalar; for training)
    sample(z, mode)         -> DC in [0,1]^3  ('sample' draws from q, 'mean' = E[DC|Z])
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),        nn.SiLU(),
            nn.Linear(hidden, 6),                                   # 3 mu + 3 logvar
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _pool(z: torch.Tensor) -> torch.Tensor:
        # z [B, T, ed] -> [B, 2*ed]  (mean + std over tokens; global, order-invariant)
        return torch.cat([z.mean(dim=1), z.std(dim=1)], dim=-1)

    def forward(self, z: torch.Tensor, target: torch.Tensor = None):
        h          = self.net(self._pool(z))
        mu, logvar = h[:, :3], h[:, 3:].clamp(-8.0, 4.0)
        if target is None:
            return mu, logvar
        eps = 1e-4
        p   = target.clamp(eps, 1.0 - eps)
        y   = torch.log(p) - torch.log1p(-p)                       # logit(target)
        nll = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))
        return nll.sum(dim=-1).mean()

    @torch.no_grad()
    def sample(self, z: torch.Tensor, mode: str = "sample") -> torch.Tensor:
        mu, logvar = self.forward(z, None)
        logit = mu if mode == "mean" else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return torch.sigmoid(logit)                                # [B, 3] in [0,1]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return f"DCHead(embed_dim={self.embed_dim}, params={self.num_params()/1e3:.1f}K)"


# ============================================================================
# Size presets
# ============================================================================

def SceneDiT_S(**kw): return SceneDiT(hidden_size=256, depth=8,  num_heads=8,  **kw)
def SceneDiT_B(**kw): return SceneDiT(hidden_size=384, depth=12, num_heads=12, **kw)  # default
def SceneDiT_L(**kw): return SceneDiT(hidden_size=512, depth=16, num_heads=16, **kw)

SceneDiT_models = {
    "SceneDiT-S": SceneDiT_S,
    "SceneDiT-B": SceneDiT_B,
    "SceneDiT-L": SceneDiT_L,
}

def CompletionDiTUncond_S(**kw): return CompletionDiTUncond(hidden_size=256, depth=8,  num_heads=8,  **kw)
def CompletionDiTUncond_B(**kw): return CompletionDiTUncond(hidden_size=384, depth=12, num_heads=12, **kw)  # default
def CompletionDiTUncond_L(**kw): return CompletionDiTUncond(hidden_size=512, depth=16, num_heads=16, **kw)

CompletionDiTUncond_models = {
    "CompletionDiTUncond-S": CompletionDiTUncond_S,
    "CompletionDiTUncond-B": CompletionDiTUncond_B,
    "CompletionDiTUncond-L": CompletionDiTUncond_L,
}