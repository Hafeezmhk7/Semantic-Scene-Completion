"""
Position-conditioned refinement heads for per-Gaussian COLOR and ROTATION.

WHY THIS EXISTS
---------------
In the token-local decoder every Gaussian inside a token is produced by a single
shared linear read-out of one per-token vector. There is NO per-Gaussian input, so
the 20 Gaussians of a region can only differ by a fixed template; the high-frequency
per-Gaussian signal (colour texture, surface-normal orientation) collapses to the
per-region mean. That is washed-out colour and round ("mean covariance") splats.
Position is the one attribute decoded accurately (anchor + offset), so we USE the
decoded position as the missing per-Gaussian coordinate: colour and orientation are
re-expressed as FIELDS over space, evaluated at each Gaussian's own location.

A standard MLP cannot fit a high-frequency function of a coordinate (spectral bias,
Rahaman et al. 2019); a Fourier-feature encoding of the coordinate fixes this
(Tancik et al. 2020). So each head conditions a small SHARED MLP on
    [ per-token content feature , gamma(position) ]
and predicts a RESIDUAL added to the decoder's base colour / quaternion. The final
layers are zero-initialised, so at start the heads are the identity and behaviour is
exactly the baseline; they only add value as they train. The MLP is shared across all
tokens and scenes (it learns a general "content + where -> local appearance" rule),
which is why this generalises, unlike a separate network per token index.

INTEGRATION
-----------
Built in AlignedShapeLatentPerceiver.__init__ behind set_pos_cond_heads(...), and
applied inside decode() right AFTER the final position (anchor + offset) is assembled,
conditioning on the per-token transformer feature H_out broadcast to each Gaussian.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Random Fourier feature map of a 3D coordinate (Tancik et al. 2020).

        gamma(x) = [ sin(2*pi * B x) , cos(2*pi * B x) ] ,   B ~ N(0, sigma^2)

    B is FIXED (registered buffer, not trained); learnability is better spent on the
    MLP than on the frequencies. Input is normalised by `pos_scale` (~ scene radius)
    so the frequency content is scene-appropriate.
    """

    def __init__(self, in_dim: int = 3, n_freqs: int = 32, sigma: float = 6.0,
                 pos_scale: float = 10.0):
        super().__init__()
        # [in_dim, n_freqs] frequency matrix, fixed.
        B = torch.randn(in_dim, n_freqs) * float(sigma)
        self.register_buffer("B", B)
        self.pos_scale = float(pos_scale)
        self.out_dim = 2 * n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim] absolute coordinate -> [..., 2*n_freqs]
        x = x / self.pos_scale
        proj = 2.0 * math.pi * (x @ self.B.to(x.dtype))     # [..., n_freqs]
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class PositionConditionedHeads(nn.Module):
    """Per-Gaussian colour / rotation refinement conditioned on (token feature, position).

    forward(recon_14, token_feat_pg):
        recon_14      : [B, N, 14] decoder output, FINAL positions in channels 0:3,
                        colour (residual) in 3:6, quaternion in 10:14.
        token_feat_pg : [B, N, token_feat_dim] per-token feature broadcast to each
                        of its Gaussians (the decoder's H_out, repeated g times).
    Returns recon_14 with channels 3:6 (colour) and/or 10:14 (rotation) refined by a
    position-conditioned residual. Zero-init final layers => identity at start.
    """

    def __init__(self, token_feat_dim: int, n_freqs: int = 32, sigma: float = 6.0,
                 pos_scale: float = 10.0, hidden: int = 128,
                 do_color: bool = True, do_rotation: bool = True):
        super().__init__()
        self.do_color = bool(do_color)
        self.do_rotation = bool(do_rotation)
        self.fourier = FourierFeatures(3, n_freqs, sigma, pos_scale)
        cond_dim = token_feat_dim + self.fourier.out_dim

        def _mlp(out_dim):
            m = nn.Sequential(
                nn.Linear(cond_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden),   nn.GELU(),
                nn.Linear(hidden, out_dim),
            )
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)        # identity (zero residual) at init
            return m

        self.color_mlp = _mlp(3) if self.do_color else None
        self.rot_mlp = _mlp(4) if self.do_rotation else None

        n_par = sum(p.numel() for p in self.parameters())
        print(f"[PositionConditionedHeads] cond_dim={cond_dim} "
              f"(token {token_feat_dim} + fourier {self.fourier.out_dim}) hidden={hidden} "
              f"| color={self.do_color} rotation={self.do_rotation} "
              f"| sigma={sigma} pos_scale={pos_scale} | {n_par/1e3:.1f}K params (zero-init)")

    def forward(self, recon_14: torch.Tensor, token_feat_pg: torch.Tensor) -> torch.Tensor:
        pos = recon_14[..., 0:3]                                   # FINAL positions
        gamma = self.fourier(pos)                                  # [B, N, 2*n_freqs]
        cond = torch.cat([token_feat_pg, gamma], dim=-1)           # [B, N, cond_dim]

        out = recon_14.clone()
        if self.color_mlp is not None:
            out[..., 3:6] = out[..., 3:6] + self.color_mlp(cond)   # residual on colour
        if self.rot_mlp is not None:
            q = out[..., 10:14] + self.rot_mlp(cond)               # residual on quaternion
            out[..., 10:14] = F.normalize(q, p=2, dim=-1)
        return out