"""
Can3Tok Training — MAIN NEW IDEA: decoder_zs_cross_attn
=========================================================
DATASET MODES (--train_data):
  "chunks"   — train_grid1.0cm_chunk8x8_stride6x6/ (default, 3888 chunks)
               Requires norm_factor.npy (run precompute_norm_from_chunks.py)
               Normalization: GLOBAL scene frame via norm_factor.npy
  "full"     — train/ (800 full scenes, per-scene normalization)
  "combined" — both sources concatenated (4688 total)
  Validation always uses val/ (held-out full scenes).

RECONSTRUCTION OBJECTIVE (--use_chamfer_loss):
  element-wise (default): torch.norm(pred[i] - target[i]) per slot. Requires a
      stable slot<->target correspondence. Pair with --morton_order so target
      slot i is a spatially-stable location (otherwise opacity rank makes the
      per-slot target unlearnable and the loss plateaus).
  chamfer (--use_chamfer_loss): permutation-invariant nearest-neighbour matching
      on position. Order-free, so --morton_order is not required.

REPORTING CONVENTION (raw, matches training):
  Both training AND validation now report the raw torch.norm per batch, averaged
  over the number of batches (NOT divided by number of scenes). This makes the
  train and val Pos/Col/Opa/Scl/Rot numbers directly comparable in scale.
  No per-element or per-channel rescaling is applied to the loss itself.
  (Minor caveat: train batch=90, val batch can differ, so the raw norm differs
   by sqrt(batch ratio); this is a scale-only artefact, not a model difference.)

BF16 MIXED PRECISION:
  torch.autocast wraps training forward, eval forward, cross-recon decode.
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path
import math

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
import torch.utils.data as Data

from semantic_losses import (compute_semantic_loss, compute_scene_infonce_loss,
                             compute_zs_token_infonce_loss,
                             compute_zs_layout_infonce_loss)
from distribution_loss import compute_distribution_loss
from pca_feature_visualization import visualize_semantic_features, visualize_z_s_space
try:
    from pca_feature_visualization import visualize_zs_tokens
except ImportError:
    print("[WARNING] visualize_zs_tokens not found — z_s token PCA disabled.")
    def visualize_zs_tokens(*args, **kwargs):
        return None
from gs_ply_reconstructor import save_reconstructed_gaussians
try:
    from render_loss import compute_render_loss
except Exception as _rl_e:  # render_loss.py or its deps missing -> only matters if --render_loss
    compute_render_loss = None
    _RENDER_LOSS_IMPORT_ERROR = _rl_e
from accelerate import Accelerator, DistributedDataParallelKwargs

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# PARAMETER INDICES
# ============================================================================
PARAM_SLICES = {
    'position': slice(0, 3), 'color': slice(3, 6),
    'opacity':  slice(6, 7), 'scale': slice(7, 10), 'rotation': slice(10, 14),
}
GEOMETRIC_INDICES = (list(range(4, 7)) + list(range(7, 10)) + [10]
                     + list(range(11, 14)) + list(range(14, 18)))
GEO_ONLY_SLICES = {
    'position': slice(0, 3), 'opacity': slice(6, 7),
    'scale': slice(7, 10),   'rotation': slice(10, 14),
}

# ============================================================================
# LOSS HELPERS
# ============================================================================
def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0):
    if color_weight == 1.0:
        return torch.norm(prediction - target, p=2) / batch_size
    return (torch.norm(prediction[:,:,0:3] - target[:,:,0:3], p=2)
          + torch.norm(prediction[:,:,3:6] - target[:,:,3:6], p=2) * color_weight
          + torch.norm(prediction[:,:,6:]  - target[:,:,6:],  p=2)) / batch_size


def _chunked_nn_indices(src, ref, chunk=2048):
    """
    For each point in src [Ns,3], the index of its nearest point in ref [Nr,3].
    Chunked over src so the [Ns,Nr] distance matrix is never materialized in full.
    argmin is non-differentiable (correct for Chamfer — gradients flow through the
    gathered values, not the indices).
    """
    Ns = src.shape[0]
    out = torch.empty(Ns, dtype=torch.long, device=src.device)
    for s in range(0, Ns, chunk):
        e = min(s + chunk, Ns)
        d = torch.cdist(src[s:e].unsqueeze(0), ref.unsqueeze(0)).squeeze(0)  # [c, Nr]
        out[s:e] = d.argmin(dim=1)
    return out


def chamfer_reconstruction_loss(prediction, target, batch_size, color_weight=1.0, chunk=2048):
    """
    Permutation-invariant reconstruction loss.

    Matches predicted and target Gaussians by nearest neighbour IN POSITION
    (columns 0:3), then applies the SAME channel weighting as
    compute_reconstruction_loss on the matched pairs
    (position + color*color_weight + rest[6:]). No opacity-specific scaling.
    Bidirectional (pred->target and target->pred). The matching is
    non-differentiable; gradients flow through the gathered values.

    NOTE: matches on the position columns AS PASSED IN. Intended for absolute
    position targets (default path; no position_scaffold / position_layout_residual).
    A warning is printed at startup if combined with those residual modes.
    """
    B = prediction.shape[0]
    total = prediction.new_zeros(())
    for b in range(B):
        pp = prediction[b]            # [N,14]
        tt = target[b]                # [N,14]
        with torch.no_grad():
            idx_p2t = _chunked_nn_indices(pp[:, 0:3], tt[:, 0:3], chunk)  # pred -> nearest target
            idx_t2p = _chunked_nn_indices(tt[:, 0:3], pp[:, 0:3], chunk)  # target -> nearest pred
        matched_t = tt.index_select(0, idx_p2t)   # [N,14]
        matched_p = pp.index_select(0, idx_t2p)   # [N,14]
        loss_fwd = (torch.norm(pp[:, 0:3] - matched_t[:, 0:3], p=2)
                  + torch.norm(pp[:, 3:6] - matched_t[:, 3:6], p=2) * color_weight
                  + torch.norm(pp[:, 6:]  - matched_t[:, 6:],  p=2))
        loss_bwd = (torch.norm(tt[:, 0:3] - matched_p[:, 0:3], p=2)
                  + torch.norm(tt[:, 3:6] - matched_p[:, 3:6], p=2) * color_weight
                  + torch.norm(tt[:, 6:]  - matched_p[:, 6:],  p=2))
        total = total + 0.5 * (loss_fwd + loss_bwd)
    return total / batch_size


# ============================================================================
# GAUGE-INVARIANT COVARIANCE RECONSTRUCTION  (--geom_loss)
# ============================================================================
# A 3D Gaussian is N(mu, Sigma) with Sigma = R diag(s^2) R^T, a 3x3 SPD matrix.
# The data stores the GAUGE-DEPENDENT factorisation (quaternion q, scale s), and
# the default element-wise L2 penalises differences in (q, s) that leave Sigma --
# hence the Gaussian itself -- unchanged: the quaternion double cover (q == -q),
# axis relabelling (permute s together with R's columns), column sign flips, and
# near-isotropy (R nearly unconstrained). That gauge term is unlearnable, which is
# why rotation never converges under L2. These losses compare the Gaussians via
# gauge-invariant distances on Sigma instead:
#   bures        : exact 2-Wasserstein / Bures shape term,
#                  tr(Sp + St - 2 (St^1/2 Sp St^1/2)^1/2)
#   bures_codiag : co-diagonalisation upper bound  ||Sp^1/2 - St^1/2||_F^2
#                  (equals Bures when Sp, St commute; cheaper and very stable)
#   logeuclid    : log-Euclidean metric  ||log Sp - log St||_F^2
#   quat_antipodal: keeps s and q SEPARATE but uses the double-cover-aware rotation
#                  distance min(||q-q'||^2, ||q+q'||^2)  (cheap ablation baseline)
#
# SPD square roots use a Newton-Schulz iteration (pure matmul -> differentiable and
# stable at coincident eigenvalues) under trace normalisation, which guarantees
# convergence because the eigenvalues of A/tr(A) lie in (0,1). logeuclid needs a
# matrix log and falls back to eigh (slower, less stable near equal eigenvalues).
# All covariance math runs in fp32 with autocast disabled.
#
# Position / colour / opacity terms are IDENTICAL to compute_reconstruction_loss
# (same torch.norm reduction), so the loss magnitude and the KL / cross-recon /
# ortho balance are preserved; only the scale+rotation block is replaced. The
# per-component Pos/Col/Opa/Scl/Rot diagnostics are computed separately and are
# unaffected. Requires --scale_norm_mode linear (covariance assembly needs s>0).

def _build_R_from_quat(q, eps=1e-8):
    """[...,4] (w,x,y,z), unit-normalised, -> rotation matrix [...,3,3] (3DGS order)."""
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R

def _assemble_sigma(scale, quat, eps=1e-6):
    """Sigma = R diag(scale^2) R^T + eps*I; invariant to the (scale,quat) gauge."""
    R     = _build_R_from_quat(quat)
    s2    = scale * scale                       # [...,3]
    RS    = R * s2.unsqueeze(-2)                 # scale the columns of R
    Sigma = RS @ R.transpose(-1, -2)
    eye   = torch.eye(3, device=Sigma.device, dtype=Sigma.dtype)
    return Sigma + eps * eye

def _ns_spd_sqrt(A, iters=10):
    """SPD matrix square root of A [...,3,3] via Newton-Schulz (trace-normalised).
    Pure matmul: differentiable and stable even at coincident/degenerate eigenvalues."""
    eye = torch.eye(3, device=A.device, dtype=A.dtype).expand_as(A)
    tr  = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)
    Y   = A / tr[..., None, None]                # eigenvalues now in (0,1) -> converges
    Z   = eye.clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * tr[..., None, None].sqrt()

def _spd_log_eigh(A, eps=1e-6):
    """Matrix log of SPD A [...,3,3] via eigendecomposition (fp32).

    Hardened for a *diagnostic* (eval-only): sanitise non-finite entries, symmetrise,
    and floor the matrix to SPD before the solve, then fall back to a CPU eigh if the
    GPU cuSOLVER path still errors (it throws CUSOLVER_STATUS_INVALID_VALUE on any
    non-finite / near-degenerate batch, e.g. random init at epoch 0)."""
    eye = torch.eye(3, device=A.device, dtype=A.dtype)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = 0.5 * (A + A.transpose(-1, -2)) + eps * eye        # symmetric + SPD floor
    try:
        evals, evecs = torch.linalg.eigh(A)
    except Exception:                                       # cuSOLVER fragility -> CPU LAPACK
        evals, evecs = torch.linalg.eigh(A.detach().double().cpu())
        evals = evals.to(A.device, A.dtype)
        evecs = evecs.to(A.device, A.dtype)
    evals = evals.clamp(min=eps)
    return (evecs * evals.log().unsqueeze(-2)) @ evecs.transpose(-1, -2)

def _trace_last2(X):
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)

def _shape_term(sp, qp, st, qt, mode, eps, ns_iters, shape_weight):
    """Norm-like scalar for the scale+rotation / covariance block of the loss.
    sp/st: per-Gaussian scale [...,3]; qp/qt: per-Gaussian quaternion [...,4]."""
    with torch.autocast(device_type='cuda', enabled=False):
        sp = sp.float(); qp = qp.float(); st = st.float(); qt = qt.float()
        if mode == 'quat_antipodal':
            scale_n = torch.norm(sp - st, p=2)
            qpn = qp / (qp.norm(dim=-1, keepdim=True) + 1e-8)
            qtn = qt / (qt.norm(dim=-1, keepdim=True) + 1e-8)
            d2  = torch.minimum(((qpn - qtn) ** 2).sum(-1),
                                ((qpn + qtn) ** 2).sum(-1))         # per Gaussian
            return scale_n + shape_weight * d2.sum().clamp(min=0).sqrt()
        Sp = _assemble_sigma(sp, qp, eps)
        St = _assemble_sigma(st, qt, eps)
        if mode == 'logeuclid':
            d2 = ((_spd_log_eigh(Sp, eps) - _spd_log_eigh(St, eps)) ** 2
                  ).flatten(start_dim=-2).sum(-1)
        elif mode == 'bures_codiag':
            d2 = ((_ns_spd_sqrt(Sp, ns_iters) - _ns_spd_sqrt(St, ns_iters)) ** 2
                  ).flatten(start_dim=-2).sum(-1)
        elif mode == 'bures':
            At = _ns_spd_sqrt(St, ns_iters)
            sM = _ns_spd_sqrt(At @ Sp @ At, ns_iters)
            d2 = (_trace_last2(Sp) + _trace_last2(St) - 2.0 * _trace_last2(sM)).clamp(min=0)
        else:
            raise ValueError(f"unknown geom_loss shape mode: {mode}")
        return shape_weight * d2.sum().clamp(min=0).sqrt()

def compute_reconstruction_loss_geom(prediction, target, batch_size, color_weight,
                                     geom_loss, eps=1e-6, ns_iters=10, shape_weight=1.0):
    """Element-wise (slot-aligned) reconstruction with a gauge-invariant scale+rotation
    term. Position/colour/opacity match compute_reconstruction_loss exactly."""
    p, t = prediction, target
    pos = torch.norm(p[:, :, 0:3] - t[:, :, 0:3], p=2)
    col = torch.norm(p[:, :, 3:6] - t[:, :, 3:6], p=2)
    opa = torch.norm(p[:, :, 6:7] - t[:, :, 6:7], p=2)
    shape = _shape_term(p[:, :, 7:10], p[:, :, 10:14],
                        t[:, :, 7:10], t[:, :, 10:14],
                        geom_loss, eps, ns_iters, shape_weight)
    return (pos + color_weight * col + opa + shape) / batch_size

def chamfer_reconstruction_loss_geom(prediction, target, batch_size, color_weight,
                                     geom_loss, chunk=2048, eps=1e-6, ns_iters=10,
                                     shape_weight=1.0):
    """Chamfer assignment (nearest neighbour by position) with the gauge-invariant
    scale+rotation term. Bidirectional; matching is non-differentiable."""
    B = prediction.shape[0]
    total = prediction.new_zeros(())
    for b in range(B):
        pp, tt = prediction[b], target[b]
        with torch.no_grad():
            idx_p2t = _chunked_nn_indices(pp[:, 0:3], tt[:, 0:3], chunk)
            idx_t2p = _chunked_nn_indices(tt[:, 0:3], pp[:, 0:3], chunk)
        matched_t = tt.index_select(0, idx_p2t)
        matched_p = pp.index_select(0, idx_t2p)
        loss_fwd = (torch.norm(pp[:, 0:3] - matched_t[:, 0:3], p=2)
                  + torch.norm(pp[:, 3:6] - matched_t[:, 3:6], p=2) * color_weight
                  + torch.norm(pp[:, 6:7] - matched_t[:, 6:7], p=2)
                  + _shape_term(pp[:, 7:10], pp[:, 10:14],
                                matched_t[:, 7:10], matched_t[:, 10:14],
                                geom_loss, eps, ns_iters, shape_weight))
        loss_bwd = (torch.norm(tt[:, 0:3] - matched_p[:, 0:3], p=2)
                  + torch.norm(tt[:, 3:6] - matched_p[:, 3:6], p=2) * color_weight
                  + torch.norm(tt[:, 6:7] - matched_p[:, 6:7], p=2)
                  + _shape_term(tt[:, 7:10], tt[:, 10:14],
                                matched_p[:, 7:10], matched_p[:, 10:14],
                                geom_loss, eps, ns_iters, shape_weight))
        total = total + 0.5 * (loss_fwd + loss_bwd)
    return total / batch_size


# ============================================================================
# PERMUTATION-INVARIANT SET RECONSTRUCTION LOSS  (--set_loss)
# ============================================================================
# Why: rendering depends only on the SET of Gaussians, not their order, and
# within a token block the assignment of which colour / orientation belongs to
# which Gaussian is a gauge freedom of the 3DGS fit (diagnostic_oracle_field.py:
# within-block colour/normal is NOT a function of within-block position, held-out
# CV-R2 < 0). A slot-aligned (element-wise) loss forces the decoder to break that
# gauge, which it cannot, so it emits the per-block mean -> washed colour + round
# blobs. This loss instead MATCHES predicted and target Gaussians WITHIN each
# token block by entropic optimal transport (Sinkhorn) and scores the matched
# pairs, so the decoder is graded on producing the right *set* of colours /
# orientations, not the (arbitrary) per-slot assignment. This is the point-cloud
# reconstruction principle (CD/EMD are permutation-invariant; "MSE cannot be
# directly applied to point clouds") brought to parameter-space 3DGS attributes.
#
# Efficiency: bures_codiag's covariance sqrt is available in closed form,
# Sigma^1/2 = R diag(|s|) R^T, so position/colour/opacity and the gauge-invariant
# shape term fold into ONE 16-dim per-Gaussian feature whose squared Euclidean
# distance IS the weighted (pos + colour + opacity + Bures-shape) cost. Matching
# is per block (g x g, g ~ 20-79; Hilbert order already fixes between-block
# assignment) and the sqrt is computed g times per block, never g^2.

def _sqrt_sigma_flat(scale, quat):
    """Sigma^{1/2} = R diag(|s|) R^T flattened to [...,9]. Exact PSD square root
    of Sigma = R diag(s^2) R^T (no Newton-Schulz needed); equals bures_codiag's
    sqrt up to the eps floor, which is irrelevant for a matching cost."""
    R  = _build_R_from_quat(quat)
    RS = R * scale.abs().unsqueeze(-2)                  # scale columns of R by |s|
    H  = RS @ R.transpose(-1, -2)                       # symmetric PSD
    return H.flatten(start_dim=-2)                      # [...,9]

def _gaussian_set_features(g14, m_pos, m_col, m_opa, m_shape):
    """[...,14] -> [...,16] feature with ||f_i - f_j||^2 =
    m_pos^2|dpos|^2 + m_col^2|dcol|^2 + m_opa^2|dopa|^2 + m_shape^2|dSigma^1/2|_F^2."""
    return torch.cat([
        g14[..., 0:3] * m_pos,
        g14[..., 3:6] * m_col,
        g14[..., 6:7] * m_opa,
        _sqrt_sigma_flat(g14[..., 7:10], g14[..., 10:14]) * m_shape,
    ], dim=-1)

def _log_sinkhorn(cost, eps, iters):
    """Entropic-OT transport plan for batched square cost [...,n,n] with uniform
    marginals (mass 1/n each), computed in the log domain for stability. Returns
    P [...,n,n] with row and column sums 1/n (total mass 1). Detach at call site:
    matching is fixed, gradient flows through the differentiable cost (DETR recipe)."""
    n = cost.shape[-1]
    log_m = cost.new_full(cost.shape[:-1], -math.log(n))     # [...,n] uniform (log)
    C = cost / eps
    f = torch.zeros_like(log_m)
    g = torch.zeros_like(log_m)
    for _ in range(iters):
        f = log_m - torch.logsumexp(g.unsqueeze(-2) - C, dim=-1)
        g = log_m - torch.logsumexp(f.unsqueeze(-1) - C, dim=-2)
    return (f.unsqueeze(-1) + g.unsqueeze(-2) - C).exp()

def compute_reconstruction_loss_set(prediction, target, batch_size, color_weight,
                                    shape_weight, block_size, pos_weight=1.0, opa_weight=1.0,
                                    sinkhorn_eps=0.05, sinkhorn_iters=50,
                                    return_components=False, return_opos=False):
    """Permutation-invariant per-block set reconstruction loss (Sinkhorn EMD).

    Same global scale / form as compute_reconstruction_loss (an L2 residual norm
    divided by batch), but each predicted Gaussian is assigned to a target by the
    within-block optimal-transport plan instead of slot order. The identity plan
    recovers the element-wise loss exactly; any reassignment can only lower it.
    block_size = g = Gaussians the decoder packs per token (= ceil(N / n_tokens))."""
    p = prediction.float(); t = target.float()
    B, N, _ = p.shape
    g = int(block_size)
    n_full = (N // g) * g
    total_sq = p.new_zeros(())
    opos = p.new_zeros(())          # opacity-weighted mean-sq position error (visible Gaussians)
    comp = {k: p.new_zeros(()) for k in ('pos', 'col', 'opa', 'shape')}

    if n_full >= g:
        nb = N // g
        pf = p[:, :n_full].reshape(B, nb, g, 14)
        tf = t[:, :n_full].reshape(B, nb, g, 14)
        with torch.autocast(device_type='cuda', enabled=False):
            with torch.no_grad():                               # matching is detached
                Fp = _gaussian_set_features(pf, pos_weight, color_weight, opa_weight, shape_weight)
                Ft = _gaussian_set_features(tf, pos_weight, color_weight, opa_weight, shape_weight)
                cost = torch.cdist(Fp, Ft) ** 2                 # [B,nb,g,g] weighted sq dist
                cnorm = cost / (cost.mean(dim=(-2, -1), keepdim=True) + 1e-9)
                P = _log_sinkhorn(cnorm, sinkhorn_eps, sinkhorn_iters)
                idx = P.argmax(dim=-1)                          # [B,nb,g] best target per pred
            # hard-gather the matched target, then weighted L2 (= 0 at pred==target, no
            # target averaging; gradient flows through the prediction only, DETR-style)
            mt = torch.gather(tf, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 14))
            dpos = (pf[..., 0:3] - mt[..., 0:3]).pow(2).sum() * (pos_weight ** 2)
            dcol = (pf[..., 3:6] - mt[..., 3:6]).pow(2).sum() * (color_weight ** 2)
            dopa = (pf[..., 6:7] - mt[..., 6:7]).pow(2).sum() * (opa_weight ** 2)
            Hp = _sqrt_sigma_flat(pf[..., 7:10], pf[..., 10:14])
            Hm = _sqrt_sigma_flat(mt[..., 7:10], mt[..., 10:14])
            dsh = (Hp - Hm).pow(2).sum() * (shape_weight ** 2)
            total_sq = total_sq + dpos + dcol + dopa + dsh
            if return_components:
                comp['pos'], comp['col'], comp['opa'], comp['shape'] = dpos, dcol, dopa, dsh
            if return_opos:
                # Opacity-weighted position error: focus the limited positional capacity
                # on VISIBLE Gaussians. Weight each Sinkhorn-matched pair's squared
                # position error by the MATCHED-TARGET opacity (GT visibility, in [0,1];
                # using GT not pred opacity so the model cannot dodge the term by dimming
                # mis-placed Gaussians). Normalised weighted mean -> scale-stable scalar,
                # differentiable through the predicted positions only (mt is a constant).
                w_opa  = mt[..., 6:7].clamp(0.0, 1.0)                       # [B,nb,g,1]
                sq_pos = (pf[..., 0:3] - mt[..., 0:3]).pow(2).sum(-1, keepdim=True)
                opos   = (w_opa * sq_pos).sum() / (w_opa.sum() + 1e-6)

    if n_full < N:                                              # tiny remainder: slot-aligned
        pr = p[:, n_full:]; tr = t[:, n_full:]
        Hp = _sqrt_sigma_flat(pr[..., 7:10], pr[..., 10:14])
        Ht = _sqrt_sigma_flat(tr[..., 7:10], tr[..., 10:14])
        total_sq = total_sq + (pr[..., 0:3] - tr[..., 0:3]).pow(2).sum() * (pos_weight ** 2) \
                            + (pr[..., 3:6] - tr[..., 3:6]).pow(2).sum() * (color_weight ** 2) \
                            + (pr[..., 6:7] - tr[..., 6:7]).pow(2).sum() * (opa_weight ** 2) \
                            + (Hp - Ht).pow(2).sum() * (shape_weight ** 2)

    loss = total_sq.clamp(min=0).sqrt() / batch_size
    if return_components:
        comps = {k: (v.clamp(min=0).sqrt() / batch_size).item() for k, v in comp.items()}
        return (loss, comps, opos) if return_opos else (loss, comps)
    if return_opos:
        return loss, opos
    return loss


@torch.no_grad()
def set_matched_individual_losses(prediction, target, block_size, color_weight, shape_weight,
                                  pos_weight=1.0, opa_weight=1.0, eps=0.05, iters=50):
    """Per-attribute reconstruction error UNDER THE SET (Sinkhorn) MATCHING.

    This is the meaningful per-component readout when --set_loss is on: the index-matched
    compute_individual_losses() compares predicted slot i to target slot i and is BLIND to
    the within-block permutation the set loss introduces (so its Col / Rot look frozen).
    This matches each predicted Gaussian to its set-assigned target and reports the same
    raw norms (same scale as compute_individual_losses). Scale+rotation also live in the
    gauge-invariant covariance ('shape'); 'rotation' is reported as the mean thin-axis
    (surface-normal) angular error in DEGREES under the matching, which is gauge-aware
    (sign-invariant) unlike a raw quaternion L2."""
    p = prediction.float(); t = target.float()
    B, N, _ = p.shape; g = int(block_size); n_full = (N // g) * g
    out = {k: 0.0 for k in ('position', 'color', 'opacity', 'scale', 'rotation', 'shape')}

    def _thin_normal(scale, quat):
        R = _build_R_from_quat(quat)
        k = scale.abs().argmin(dim=-1)                              # smallest-scale (thin) axis
        n = torch.gather(R, -1, k[..., None, None].expand(*k.shape, 3, 1)).squeeze(-1)
        return n / (n.norm(dim=-1, keepdim=True) + 1e-9)

    if n_full >= g:
        nb = N // g
        pf = p[:, :n_full].reshape(B, nb, g, 14); tf = t[:, :n_full].reshape(B, nb, g, 14)
        Fp = _gaussian_set_features(pf, pos_weight, color_weight, opa_weight, shape_weight)
        Ft = _gaussian_set_features(tf, pos_weight, color_weight, opa_weight, shape_weight)
        cost = torch.cdist(Fp, Ft) ** 2
        cnorm = cost / (cost.mean(dim=(-2, -1), keepdim=True) + 1e-9)
        idx = _log_sinkhorn(cnorm, eps, iters).argmax(dim=-1)
        mt = torch.gather(tf, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 14))
        out['position'] = (pf[..., 0:3] - mt[..., 0:3]).pow(2).sum().sqrt().item()
        out['color']    = (pf[..., 3:6] - mt[..., 3:6]).pow(2).sum().sqrt().item()
        out['opacity']  = (pf[..., 6:7] - mt[..., 6:7]).pow(2).sum().sqrt().item()
        out['scale']    = (pf[..., 7:10] - mt[..., 7:10]).pow(2).sum().sqrt().item()
        Hp = _sqrt_sigma_flat(pf[..., 7:10], pf[..., 10:14])
        Hm = _sqrt_sigma_flat(mt[..., 7:10], mt[..., 10:14])
        out['shape']    = (Hp - Hm).pow(2).sum().sqrt().item()
        # gauge-aware orientation: angle between predicted and matched-target thin-axis normals
        cos = (_thin_normal(pf[..., 7:10], pf[..., 10:14]) *
               _thin_normal(mt[..., 7:10], mt[..., 10:14])).sum(-1).abs().clamp(max=1.0)
        out['rotation'] = torch.rad2deg(torch.arccos(cos)).mean().item()   # mean degrees
    return out


@torch.no_grad()
def compute_covariance_diagnostics(prediction, target, use_chamfer=False, chunk=2048,
                                   eps=1e-6, ns_iters=10):
    """GAUGE-INVARIANT validation metric (diagnostic only, no grad).

    Returns the mean per-Gaussian distance between predicted and target COVARIANCES
    Sigma = R diag(s^2) R^T, which -- unlike the raw quaternion/scale L2 -- is blind
    to the double-cover / axis-permutation gauge and therefore actually measures
    whether orientation+shape are reconstructed. Also reports the target's mean
    anisotropy (max/min scale-axis ratio): if this is ~1, orientation is moot.

      cov_bures    : mean sqrt(Bures^2) over Gaussians (Newton-Schulz sqrt)
      cov_logeuclid: mean ||log Sp - log St||_F over Gaussians (eigh)
      aniso        : mean (s_max / s_min) of the TARGET Gaussians

    Matching mirrors the per-component readout: nearest-neighbour by position under
    Chamfer, slot-aligned (index) otherwise. Computed in fp32, autocast disabled.
    """
    B = prediction.shape[0]
    with torch.autocast(device_type='cuda', enabled=False):
        p = prediction.float(); t = target.float()
        if use_chamfer:
            mt = []
            for b in range(B):
                idx = _chunked_nn_indices(p[b, :, 0:3], t[b, :, 0:3], chunk)
                mt.append(t[b].index_select(0, idx))
            T = torch.stack(mt, 0)
        else:
            T = t
        sp, qp = p[:, :, 7:10], p[:, :, 10:14]
        st, qt = T[:, :, 7:10], T[:, :, 10:14]
        Sp = _assemble_sigma(sp, qp, eps)
        St = _assemble_sigma(st, qt, eps)
        At = _ns_spd_sqrt(St, ns_iters)
        sM = _ns_spd_sqrt(At @ Sp @ At, ns_iters)
        bures = (_trace_last2(Sp) + _trace_last2(St) - 2.0 * _trace_last2(sM)
                 ).clamp(min=0).sqrt().mean()
        le = ((_spd_log_eigh(Sp, eps) - _spd_log_eigh(St, eps)) ** 2
              ).flatten(start_dim=-2).sum(-1).sqrt().mean()
        s_sorted = st.abs().clamp(min=1e-8).sort(dim=-1).values
        aniso = (s_sorted[..., -1] / s_sorted[..., 0]).mean()
    return {'cov_bures': bures.item(), 'cov_logeuclid': le.item(), 'aniso': aniso.item()}


def compute_individual_losses(prediction, target):
    return {k: torch.norm(prediction[:,:,sl] - target[:,:,sl], p=2).item()
            for k, sl in PARAM_SLICES.items()}

def compute_individual_losses_matched(prediction, target, chunk=2048):
    """
    Per-attribute reconstruction error UNDER NEAREST-NEIGHBOUR MATCHING (by position).

    WHY THIS EXISTS: with Chamfer loss the model is trained to match the predicted
    SET to the target SET, with no constraint that predicted slot i corresponds to
    target slot i. The index-matched compute_individual_losses() therefore reports
    garbage for position (and a misleading "looks converged" for near-constant
    channels like opacity/scale) because it compares pred[i] to target[i] across two
    differently-ordered sets. This version matches each predicted Gaussian to its
    nearest target Gaussian in position (cols 0:3) and reports per-attribute norms on
    those matched pairs, which is the meaningful per-component readout for Chamfer.
    Diagnostic only (no grad). Direction: pred -> nearest target.
    """
    B = prediction.shape[0]
    out = {k: 0.0 for k in PARAM_SLICES}
    with torch.no_grad():
        for b in range(B):
            pp = prediction[b]
            tt = target[b]
            idx = _chunked_nn_indices(pp[:, 0:3], tt[:, 0:3], chunk)  # pred -> nearest target
            matched_t = tt.index_select(0, idx)
            for k, sl in PARAM_SLICES.items():
                out[k] += torch.norm(pp[:, sl] - matched_t[:, sl], p=2).item()
    return out

def scene_semantic_kl_loss(p_hat, p_s, eps=1e-8):
    return (p_s * (torch.log(p_s + eps) - torch.log(p_hat.clamp(min=eps)))).sum(-1).mean()

def compute_cross_recon_loss(pred_cross_3d, target, batch_size):
    loss = torch.tensor(0.0, device=pred_cross_3d.device)
    for sl in GEO_ONLY_SLICES.values():
        loss = loss + torch.norm(pred_cross_3d[:,:,sl] - target[:,:,sl], p=2) / batch_size
    return loss

def compute_orthogonality_loss(mu_s, mu_g, proj_dim=64):
    B = mu_s.shape[0]
    if B < 2: return torch.tensor(0.0, device=mu_s.device)
    with torch.no_grad():
        p = min(proj_dim, B - 1, mu_s.shape[1], mu_g.shape[1])
        is_ = torch.randperm(mu_s.shape[1], device=mu_s.device)[:p]
        ig  = torch.randperm(mu_g.shape[1], device=mu_g.device)[:p]
    ps = F.normalize(mu_s[:,is_] - mu_s[:,is_].mean(0,True), p=2, dim=0)
    pg = F.normalize(mu_g[:,ig]  - mu_g[:,ig].mean(0,True),  p=2, dim=0)
    return ((ps.T @ pg) ** 2).mean()

def compute_layout_loss(pred_c, gt_c, gt_valid):
    return ((((pred_c - gt_c)**2).mean(-1)) * gt_valid).sum() / (gt_valid.sum() + 1e-8)

def compute_scale_penalty(pred_3d, threshold=0.5):
    return (torch.clamp(pred_3d[:,:,7:10] - threshold, min=0.0)**2).mean()

def compute_seg_pred_loss(seg_logits, segment_labels):
    B, N, C = seg_logits.shape
    fl = seg_logits.reshape(B*N, C); ll = segment_labels.reshape(B*N).long()
    valid = ll >= 0
    if valid.sum() == 0: return torch.tensor(0.0, device=seg_logits.device)
    return F.cross_entropy(fl[valid], ll[valid])

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Can3Tok Training')
parser.add_argument('--batch_size',           type=int,   default=64)
parser.add_argument('--num_epochs',           type=int,   default=1000)
parser.add_argument('--lr',                   type=float, default=1e-4)
parser.add_argument('--kl_weight',            type=float, default=1e-5)
parser.add_argument('--kl_anneal_steps',      type=int,   default=0,
    help='Number of optimizer steps over which to ramp kl_weight from 0 to its '
         'target value (linear warm-up). 0 = no annealing (fixed kl_weight).')
parser.add_argument('--weight_decay',         type=float, default=1e-2)
parser.add_argument('--warmup_steps',         type=int,   default=100)
parser.add_argument('--lr_min_ratio',         type=float, default=0.1)
parser.add_argument('--lr_restart_T0',        type=int,   default=0,
    help='Cosine warm restart period in EPOCHS. 0 = single cosine decay.')
parser.add_argument('--eval_every',           type=int,   default=20)
parser.add_argument('--failure_threshold',    type=float, default=100.0)
parser.add_argument('--train_scenes',         type=int,   default=None)
parser.add_argument('--val_scenes',           type=int,   default=None)
parser.add_argument('--chunk_val_scenes',     type=int,   default=None,
    help='Held-out chunk count for the chunk-val split (chunks/combined only). These are '
         'the chunks sorted AFTER the first --train_scenes, so they are DISJOINT from '
         'training by construction. None = use all remaining chunks (e.g. 3888 total - '
         '3800 train = 88). For a clean split do NOT set --random_subset_seed: a random '
         'training subset overlaps the skipped val chunks, which the disjointness check '
         'will reject.')
parser.add_argument('--sampling_method',      type=str,   default='opacity',
                    choices=['random','opacity','hybrid','uniform','fps',
                             'uniform_instance','fps_instance'],
                    help="Which TARGET_POINTS Gaussians to keep per scene. 'opacity' = top-K "
                         "by opacity (density follows the opacity field; non-uniform, hard to "
                         "compress). 'uniform'/'fps' = density-uniform FPS-style grid sampling "
                         "(equalizes per-region density; recommended). 'uniform_instance'/"
                         "'fps_instance' = the same, stratified by instance label (object-"
                         "coherent + uniform). Uniform variants need --order_frame_radius>0.")
parser.add_argument('--sample_voxel_res',      type=int,   default=96,
                    help="Grid resolution for the uniform/FPS sampler (--sampling_method "
                         "uniform/fps/*_instance). Finer = more uniform but needs occupied "
                         "voxels >= num_gaussians; 96 over a [-10,10] frame works for ~40k.")
parser.add_argument('--random_subset_seed', type=int, default=None,
    help='Random seed for selecting a subset of scenes. None = sorted first-N.')
# Spatial crop
parser.add_argument('--crop_percentile', type=float, default=100.0,
    help='Spatial crop: keep inner crop_percentile%% of Gaussians by distance '
         'from centroid before opacity sampling. 100.0 = disabled (default).')
# Gaussian ordering (Morton / Z-order)
parser.add_argument('--morton_order', action='store_true', default=False,
    help='Reorder opacity-selected Gaussians along a space-filling curve so slot i maps to '
         'a spatially-stable location. Makes element-wise loss learnable. Off by default. '
         'Curve chosen by --order_curve.')
parser.add_argument('--order_curve', type=str, default='hilbert', choices=['hilbert', 'morton'],
    help='Space-filling curve used when --morton_order is set. "hilbert" (default) has '
         'provably better locality than "morton" (Z-order): consecutive slots are always '
         'spatially adjacent, giving a smoother slot->position target that fits and '
         'generalizes better. "morton" kept for ablation/back-compat.')
parser.add_argument('--order_frame_radius', type=float, default=10.0,
    help='Frame for the space-filling sort. >0 (default 10.0, the canonical normalization '
         'radius) = FIXED canonical frame [-R,R]: the curve traverses the same absolute '
         'cells in the same order for every scene, so the slot->position target is '
         'consistent across scenes (matches the original Can3Tok HilbertSort3D). '
         '<=0 = legacy PER-SCENE min-max (each scene stretched to the grid; ordering is '
         'scene-idiosyncratic, which hurts cross-scene generalization).')
parser.add_argument('--canonical_voxel', action='store_true', default=False,
    help="Re-express each scene as a CANONICAL density-adaptive voxel set before training: "
         "keep one representative (most opaque) Gaussian per occupied voxel of an "
         "order_frame_radius-framed grid, Hilbert-ordered, padded to --num_gaussians with "
         "zero-opacity dummies. Converts the non-identifiable raw per-Gaussian arrangement "
         "into an identifiable target (occupied cell + small offset). Best paired with a "
         "render-PRIMARY objective (padding is invisible to the render). Replaces opacity "
         "sampling + reorder.")
parser.add_argument('--voxel_res', type=int, default=64,
    help="Canonical voxel grid resolution per axis (--canonical_voxel). Higher = more, "
         "smaller cells = less merging, more detail, more representatives (closer to the raw "
         "count). 64 over a [-10,10] frame ~= 0.31-unit cells. Tune so occupied voxels are "
         "near --num_gaussians.")
parser.add_argument('--voxel_snap', action='store_true', default=False,
    help="With --canonical_voxel, SNAP each representative to its voxel centre (position "
         "becomes a pure grid, fully determined by occupancy; GaussianCube-style). Off = keep "
         "the representative's true sub-voxel position (retains fine offset).")
# Reconstruction objective
parser.add_argument('--use_chamfer_loss', action='store_true', default=False,
    help='Use permutation-invariant Chamfer reconstruction loss (NN matching on '
         'position) instead of element-wise L2. Order-free (no --morton_order '
         'needed). Slower due to the NN search. Off by default (element-wise).')
parser.add_argument('--chamfer_chunk', type=int, default=2048,
    help='Chunk size for the Chamfer NN search (memory control). Lower if OOM. '
         'Only used when --use_chamfer_loss.')
parser.add_argument('--geom_loss', type=str, default='l2',
    choices=['l2', 'quat_antipodal', 'bures', 'bures_codiag', 'logeuclid'],
    help="Geometric reconstruction metric for the scale+rotation block. 'l2' "
         "(default) keeps the current element-wise behaviour unchanged. The "
         "others are GAUGE-INVARIANT: they compare the covariance "
         "Sigma=R diag(s^2) R^T instead of raw (quat, scale), removing the "
         "unlearnable quaternion double-cover / axis-permutation gauge that "
         "stalls rotation. bures=exact 2-Wasserstein shape term; "
         "bures_codiag=stable co-diagonalisation upper bound (recommended); "
         "logeuclid=log-Euclidean; quat_antipodal=cheap double-cover-aware "
         "rotation ablation. Composes orthogonally with --use_chamfer_loss "
         "(assignment). Requires --scale_norm_mode linear.")
parser.add_argument('--geom_eps', type=float, default=1e-6,
    help='Isotropic regulariser added to each covariance (Sigma + eps*I) for '
         'numerical stability of the matrix square root / log.')
parser.add_argument('--geom_ns_iters', type=int, default=10,
    help='Newton-Schulz iterations for the SPD matrix square root '
         '(bures / bures_codiag). More = more accurate, slower.')
parser.add_argument('--geom_shape_weight', type=float, default=1.0,
    help='Multiplier on the covariance/rotation shape term relative to the '
         'position/colour/opacity terms.')
# ── Permutation-invariant per-block SET reconstruction loss (Sinkhorn EMD) ────
# Replaces the slot-aligned recon loss with within-block optimal-transport
# matching, so colour/orientation are graded as a SET (the gauge-correct target)
# rather than per slot. Folds the gauge-invariant covariance sqrt into the OT
# cost, so it subsumes the geom shape term (uses --color_loss_weight and
# --geom_shape_weight as the colour/shape multipliers). Default OFF (baseline).
parser.add_argument('--set_loss', action='store_true',
    help='Use the permutation-invariant per-block set (Sinkhorn EMD) reconstruction '
         'loss instead of the slot-aligned one. Targets the washed-colour / round-blob '
         'failure, which is a within-block permutation gauge, not lost information.')
parser.add_argument('--set_block_size', type=int, default=0,
    help='Gaussians per token block for set matching. 0 = auto = ceil(num_gaussians/512), '
         'i.e. the decoder fan-out g (where the gauge lives).')
parser.add_argument('--set_pos_weight', type=float, default=1.0,
    help='Position multiplier in the set-matching feature. Lower lets colour/shape '
         'drive the within-block reassignment more freely.')
parser.add_argument('--set_opa_weight', type=float, default=1.0,
    help='Opacity multiplier in the set-matching feature.')
parser.add_argument('--set_sinkhorn_eps', type=float, default=0.05,
    help='Entropic regularisation for Sinkhorn (relative to per-block mean cost). '
         'Lower = harder, more permutation-like matching (needs more iters).')
parser.add_argument('--set_sinkhorn_iters', type=int, default=50,
    help='Sinkhorn iterations per block.')
parser.add_argument('--set_diag_every', type=int, default=10,
    help='Compute the (expensive) set-matched per-component training readout every N steps '
         'and carry it forward between (the loss still matches every step). Reduces the '
         'redundant second Sinkhorn/step; the epoch-mean readout stays representative.')

# ── Virtual-camera rendering loss (default OFF; complements the set loss) ──────
parser.add_argument('--render_loss', action='store_true',
    help='Add a virtual-camera rendering loss: render predicted vs GT Gaussians from the '
         'same synthetic cameras and minimise image L1 + D-SSIM. Needs gsplat. Verify with '
         'render_check.py first.')
parser.add_argument('--render_loss_weight', type=float, default=0.5,
    help='Weight of the render loss added to total_loss.')
parser.add_argument('--render_warmup_epochs', type=int, default=0,
    help='Only apply the render loss after this many epochs (fine-tune phase). 0 = from start.')
parser.add_argument('--render_views', type=int, default=4,
    help='Cameras per scene (azimuth ring).')
parser.add_argument('--render_img', type=int, default=128,
    help='Render resolution (square). Keep modest at 40k Gaussians.')
parser.add_argument('--render_max_scenes', type=int, default=8,
    help='Scenes per step to render (random subset each step; cost control). 0 = whole batch.')
parser.add_argument('--render_ssim_weight', type=float, default=0.2,
    help='Weight of (1 - SSIM); the L1 term has weight 1.0.')
parser.add_argument('--render_lpips_weight', type=float, default=0.0,
    help="Weight of the LPIPS perceptual term inside the render loss (L1 + ssim*D-SSIM "
         "+ lpips*LPIPS). 0 = off. ~1.0 for a render-PRIMARY objective. Needs the `lpips` "
         "package; degrades gracefully to L1+D-SSIM if its weights cannot be fetched.")
parser.add_argument('--param_loss_weight', type=float, default=1.0,
    help="Global multiplier on the PARAMETER-space reconstruction loss (the set/Chamfer/L2 "
         "recon term). 1.0 = current behaviour (parameter-primary). For a render-PRIMARY "
         "objective set this small (e.g. 0.1) and raise --render_loss_weight (e.g. 5-10) so "
         "the gauge-invariant render signal drives the fit and parameters only regularize.")
parser.add_argument('--render_fov', type=float, default=50.0)
parser.add_argument('--render_dist_mult', type=float, default=2.6,
    help='Camera distance = render_dist_mult * chunk radius.')
parser.add_argument('--render_up_axis', type=int, default=2, choices=[0, 1, 2],
    help='World up axis for camera elevation (match what looked correct in render_check).')
parser.add_argument('--render_quat_order', default='wxyz', choices=['wxyz', 'xyzw'],
    help="Quaternion order gsplat expects. Use xyzw ONLY if render_check showed scrambled "
         "geometry with wxyz.")
cov_grp = parser.add_mutually_exclusive_group()
cov_grp.add_argument('--cov_metric', dest='cov_metric', action='store_true', default=True,
    help='[DEFAULT ON] Report the gauge-invariant covariance error (mean per-Gaussian '
         'Bures + log-Euclidean) and target anisotropy at each eval. Independent of '
         '--geom_loss, so it enables L2-vs-Bures checkpoint comparison on one metric.')
cov_grp.add_argument('--no_cov_metric', dest='cov_metric', action='store_false',
    help='Disable the covariance eval metric (saves a little eval time).')
parser.add_argument('--aug_yaw', action='store_true', default=False,
    help='Augment TRAINING scenes with a random yaw rotation about the gravity axis '
         '(positions rotate by R, each Gaussian orientation composes with R so the '
         'covariance transforms consistently). Targets the train/val overfitting gap. '
         'Disables the dataset preload cache (fresh rotation per access).')
parser.add_argument('--aug_yaw_axis', type=str, default='z', choices=['x', 'y', 'z'],
    help='Up/gravity axis for yaw augmentation (default z).')
parser.add_argument('--aug_yaw_max_deg', type=float, default=180.0,
    help='Yaw sampled uniformly in [-max, max] degrees (180 = full circle).')
# Dataset source
parser.add_argument('--train_data',           type=str,   default='chunks',
                    choices=['chunks', 'full', 'combined'],
    help='"chunks" = train_grid/ (global norm), "full" = train/ (per-scene norm), '
         '"combined" = both.')
# MAIN NEW IDEA
parser.add_argument('--decoder_zs_cross_attn', action='store_true', default=False)
# Per-Gaussian InfoNCE
parser.add_argument('--semantic_mode',        type=str,   default='none',
                    choices=['none','hidden','geometric','dist'])
parser.add_argument('--segment_loss_weight',  type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample',   type=int,   default=2000)
parser.add_argument('--sampling_strategy',    type=str,   default='balanced',
                    choices=['random','balanced'])
# Scene z_s InfoNCE
parser.add_argument('--z_s_infonce_weight',      type=float, default=0.0)
parser.add_argument('--z_s_infonce_temperature', type=float, default=0.07)
parser.add_argument('--z_s_infonce_delta',       type=float, default=0.4)
# Strategy B flags
parser.add_argument('--decoder_layout_cross_attn', action='store_true', default=False)
parser.add_argument('--decoder_layout_additive',   action='store_true', default=False)
parser.add_argument('--structured_layout_tokens',  action='store_true', default=False)
parser.add_argument('--zs_layout_infonce_weight',      type=float, default=0.0)
parser.add_argument('--zs_layout_infonce_temperature', type=float, default=0.07)
# z_s pool / token InfoNCE
parser.add_argument('--zs_pool_infonce_weight',      type=float, default=0.0)
parser.add_argument('--zs_pool_infonce_temperature', type=float, default=0.07)
parser.add_argument('--zs_token_infonce_weight',      type=float, default=0.0)
parser.add_argument('--zs_token_infonce_temperature', type=float, default=0.07)
# Core
parser.add_argument('--color_residual',       action='store_true', default=False)
parser.add_argument('--mean_color_weight',    type=float, default=1.0)
parser.add_argument('--scene_semantic_head',  action='store_true', default=False)
parser.add_argument('--scene_semantic_weight',type=float, default=0.3)
parser.add_argument('--position_scaffold',    action='store_true', default=False)
parser.add_argument('--anchor_loss_weight',   type=float, default=1.0)
parser.add_argument('--scaffold_mode', type=str, default='voxel',
                    choices=['voxel', 'hilbert_block'],
                    help="position_scaffold anchor construction. 'voxel' = legacy fixed "
                         "8^3 grid (anchors ~ cell centres). 'hilbert_block' = adaptive "
                         "per-block centroids over the space-filling-ordered points "
                         "(real per-scene cluster centres; requires --morton_order). "
                         "Pairs with --anchor_relative_decode.")
parser.add_argument('--anchor_relative_decode', action='store_true', default=False,
                    help="Decode position as anchor + offset_scale*tanh(offset) "
                         "(Scaffold-GS style BOUNDED local offset), so the per-token "
                         "anchor must carry coarse position. Auto-enables "
                         "--position_scaffold; recon loss uses ABSOLUTE position targets.")
parser.add_argument('--anchor_teacher_force', action='store_true', default=False,
                    help="Diagnostic upper bound: feed GT block centroids as the anchor "
                         "inside decode (needs the Stage-1 wrapper to forward "
                         "scaffold_anchors). Isolates the decoder from anchor prediction.")
parser.add_argument('--offset_scale_init', type=float, default=2.0,
                    help="Initial value of the learnable global offset scale used by "
                         "--anchor_relative_decode (it adapts during training).")
parser.add_argument('--micro_pattern',         action='store_true', default=False,
                    help="GaussianCube-style framed canonical micro-pattern: decode "
                         "pos = anchor + R_block.(s_block.c[slot]) + micro*tanh(resid), "
                         "where c is a fixed unit-ball point set and (s_block,R_block) is "
                         "a per-token frame. Requires --anchor_relative_decode and "
                         "scaffold_mode='hilbert_block'. Retrain (not a fine-tune of the "
                         "free-offset decoder).")
parser.add_argument('--micro_pattern_no_rotation', action='store_true', default=False,
                    help="Ablation: drop the per-token rotation (anisotropic scale only) "
                         "from --micro_pattern. Default keeps rotation, needed to orient "
                         "the flattened pattern to non-axis-aligned surfaces.")
parser.add_argument('--micro_offset_scale',    type=float, default=0.3,
                    help="Initial bound of the SMALL per-Gaussian residual on top of the "
                         "framed canonical pattern (--micro_pattern). Learnable; if it "
                         "grows large the pattern prior is not fitting the local surfaces.")
parser.add_argument('--embed_dim',             type=int,   default=None,
                    help="Per-token latent width (capacity knob). None = use the YAML "
                         "value (32 -> latent 512*32 = 16384). The latent token count is "
                         "pinned to the 512-block count, so the structured-latent total = "
                         "512 * embed_dim: set 64 to DOUBLE capacity to 32768 (the real "
                         "lever for the position floor). Needs a Stage-2 retrain at the new "
                         "latent size. KL scales ~linearly with this, so KL*weight grows.")
parser.add_argument('--opacity_pos_weight',    type=float, default=0.0,
                    help="Weight of an opacity-weighted position term added to the loss: "
                         "sum_i o_i*|dp_i|^2 / sum_i o_i over Sinkhorn-matched pairs, "
                         "weighted by GT opacity. Spends positional capacity on VISIBLE "
                         "Gaussians (improves renders) rather than the aggregate Pos mean. "
                         "Requires --set_loss. 0 = off; try ~0.05 to start.")
parser.add_argument('--latent_disentangle',   action='store_true', default=False)
parser.add_argument('--structured_latent',     action='store_true', default=False,
                    help="Per-token latent: token k of z == encoder token k == Hilbert "
                         "block k. Skips the dense kl_emb_proj_mean global remix. Overrides "
                         "--latent_disentangle (off). Pairs with --anchor_relative_decode.")
parser.add_argument('--local_encoder',         action='store_true', default=False,
                    help="Windowed spatially-local encoder: geometry query k attends only "
                         "to a window of Hilbert blocks around block k (compositional, "
                         "generalizing features). Implies --structured_latent.")
parser.add_argument('--local_window',          type=int,   default=1,
                    help="Half-width (in blocks) of the local encoder attention window. "
                         "0 = pure block-diagonal (each token sees only its own block); "
                         "-1 = GLOBAL (every geometry query attends to all points) for the "
                         "locality ablation -- identical to the windowed encoder otherwise.")
parser.add_argument('--semantic_dims',        type=int,   default=512)
parser.add_argument('--cross_recon_weight',   type=float, default=0.3)
parser.add_argument('--ortho_weight',         type=float, default=0.1)
parser.add_argument('--scene_layout_head',    action='store_true', default=False)
parser.add_argument('--layout_loss_weight',   type=float, default=0.3)
parser.add_argument('--position_layout_residual', action='store_true', default=False)
parser.add_argument('--decoder_pos_enc',      action='store_true', default=False)
parser.add_argument('--predict_seg_labels',   action='store_true', default=False)
parser.add_argument('--seg_pred_weight',      type=float, default=0.3)
parser.add_argument('--token_cond',           action='store_true', default=False)
parser.add_argument('--token_cond_approach',  type=str,   default='B',
                    choices=['A','B','AB'])
parser.add_argument('--decoder_fourier_pe',   action='store_true', default=False)
parser.add_argument('--token_local_decoder', action='store_true', default=False,
    help='Replace flat GS_decoder with shared per-token MLP. See token_local_decoder.py.')
parser.add_argument('--token_cond_adaln',     action='store_true', default=False)
parser.add_argument('--semantic_token_heads', action='store_true', default=False)
# Legacy
parser.add_argument('--jepa_idea1',           action='store_true', default=False)
parser.add_argument('--jepa_idea1_weight',    type=float, default=1.0)
parser.add_argument('--query_decoder',        action='store_true', default=False)
parser.add_argument('--label_input',          action='store_true', default=False)
parser.add_argument('--no_label_input',       dest='label_input', action='store_false')
parser.add_argument('--scale_norm_mode',      type=str,   default='linear',
                    choices=['log','linear'])
parser.add_argument('--color_loss_weight',    type=float, default=1.0)
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)
parser.add_argument('--scale_penalty_threshold', type=float, default=0.5)
preload_grp = parser.add_mutually_exclusive_group()
preload_grp.add_argument('--preload', dest='preload', action='store_true', default=True,
    help='[DEFAULT ON] Preprocess every scene once into RAM. Fast per-epoch, but EACH '
         'DDP rank holds a full copy, so a large multi-dataset mix can exceed --mem and '
         'get SIGKILL-ed (exit -9).')
preload_grp.add_argument('--no_preload', dest='preload', action='store_false',
    help='Load + process each scene on-the-fly per access instead of caching in RAM. '
         'Much lower host memory (use for the full chunks+full multi-dataset mix); '
         'epochs are slower (disk I/O + sampling/sort per access, hidden by the workers).')
norm_grp = parser.add_mutually_exclusive_group()
norm_grp.add_argument('--use_canonical_norm', dest='use_canonical_norm',
                      action='store_true', default=True)
norm_grp.add_argument('--no_canonical_norm',  dest='use_canonical_norm',
                      action='store_false')
chunk_norm_grp = parser.add_mutually_exclusive_group()
chunk_norm_grp.add_argument('--chunk_norm_factor', dest='chunk_norm_factor',
    action='store_true', default=True,
    help='[DEFAULT ON] Use norm_factor.npy global frame for grid chunks.')
chunk_norm_grp.add_argument('--no_chunk_norm_factor', dest='chunk_norm_factor',
    action='store_false',
    help='Disable norm_factor.npy for chunks: force per-scene normalisation.')
color_norm_grp = parser.add_mutually_exclusive_group()
color_norm_grp.add_argument('--normalize_colors',    dest='normalize_colors',
                            action='store_true', default=True)
color_norm_grp.add_argument('--no_normalize_colors', dest='normalize_colors',
                            action='store_false')
parser.add_argument('--pca_vis_freq',         type=int,   default=50)
parser.add_argument('--pca_brightness',       type=float, default=1.25)
parser.add_argument('--pca_num_scenes',       type=int,   default=3)
parser.add_argument('--recon_ply_freq',       type=int,   default=50)
parser.add_argument('--recon_ply_num_scenes', type=int,   default=3)
parser.add_argument('--recon_ply_max_sh',     type=int,   default=3)
parser.add_argument('--num_gaussians',        type=int,   default=10000,
                    help="Gaussians per scene. Sets the dataset sample count AND the "
                         "decoder/encoder Gaussian count together (kept consistent). The "
                         "latent stays 512 tokens; only g=ceil(num_gaussians/512) per token "
                         "scales. Default 10000 reproduces prior runs.")
# Position-conditioned per-Gaussian colour/rotation refinement heads (off by default).
parser.add_argument('--pos_cond_heads', action='store_true',
                    help="Enable Fourier position-conditioned refinement of per-Gaussian "
                         "colour and rotation (fixes mean-collapse / washed colour + round "
                         "splats). Identity at init, so it only adds value as it trains.")
parser.add_argument('--pos_cond_color',     type=int,   default=1,
                    help="1=refine colour with the position head (default), 0=off.")
parser.add_argument('--pos_cond_rotation',  type=int,   default=1,
                    help="1=refine rotation with the position head (default), 0=off. "
                         "Most useful with a gauge-invariant --geom_loss (bures*/logeuclid).")
parser.add_argument('--pos_cond_n_freqs',   type=int,   default=32,
                    help="Number of Fourier frequency pairs for the position encoding.")
parser.add_argument('--pos_cond_sigma',     type=float, default=6.0,
                    help="Std of the Fourier frequency matrix (bandwidth). Higher = higher "
                         "frequency detail. Tune if colour/orientation stay too smooth/noisy.")
parser.add_argument('--pos_cond_pos_scale', type=float, default=10.0,
                    help="Position normaliser for the Fourier encoding; set to ~scene radius.")
parser.add_argument('--pos_cond_hidden',    type=int,   default=128,
                    help="Hidden width of the shared refinement MLPs.")
parser.add_argument('--use_wandb',            action='store_true', default=False)
parser.add_argument('--wandb_project',        type=str,   default='Can3Tok-SceenSplat-7K')
parser.add_argument('--wandb_entity',         type=str,   default='3D-SSC')
parser.add_argument('--resume_checkpoint',    type=str,   default=None)
parser.add_argument('--resume_epoch',         type=int,   default=None)
# Multi-dataset support
parser.add_argument('--extra_train_paths',    type=str,   default='',
    help='Colon-separated list of extra scene root directories added on top of '
         '--train_data. Semantics disabled automatically.')
parser.add_argument('--extra_train_scenes',   type=str,   default='',
    help='Colon-separated max scenes per extra path (0 = all). Example: "1290:906"')

args = parser.parse_args()

# Validation of flags
if args.geom_loss != 'l2' and args.scale_norm_mode != 'linear':
    raise ValueError(
        f"--geom_loss {args.geom_loss} assembles Sigma = R diag(scale^2) R^T and "
        f"needs positive linear scale, but --scale_norm_mode is "
        f"'{args.scale_norm_mode}'. Set --scale_norm_mode linear "
        f"(SCALE_NORM_MODE=\"linear\" in the job).")
if args.local_encoder:
    args.structured_latent = True
if args.structured_latent and args.latent_disentangle:
    print("[INFO] --structured_latent/--local_encoder overrides --latent_disentangle (off); "
          "cross_recon / ortho / z_s-InfoNCE auto-disable below.")
    args.latent_disentangle = False

if args.decoder_zs_cross_attn and not args.latent_disentangle:
    raise ValueError("--decoder_zs_cross_attn requires --latent_disentangle")
if args.cross_recon_weight > 0 and not args.latent_disentangle:
    args.cross_recon_weight = 0.0
if args.ortho_weight > 0 and not args.latent_disentangle:
    args.ortho_weight = 0.0
if args.z_s_infonce_weight > 0 and not args.latent_disentangle:
    args.z_s_infonce_weight = 0.0
if args.zs_token_infonce_weight > 0 and not args.latent_disentangle:
    print("[WARNING] zs_token_infonce_weight > 0 requires latent_disentangle. Setting to 0.")
    args.zs_token_infonce_weight = 0.0
_any_B = args.decoder_layout_cross_attn or args.decoder_layout_additive
if args.zs_layout_infonce_weight > 0 and not _any_B:
    if args.latent_disentangle:
        print("[INFO] zs_layout_infonce_weight > 0 with Strategy A: routing z_s as layout tokens")
    else:
        print("[WARNING] zs_layout_infonce_weight > 0 requires decoder_layout_* or latent_disentangle. Setting to 0.")
        args.zs_layout_infonce_weight = 0.0
if _any_B and args.latent_disentangle:
    print("[INFO] decoder_layout_cross/additive=True with latent_disentangle=True.")
if args.semantic_dims % (args.embed_dim or 32) != 0:
    raise ValueError(f"--semantic_dims must be divisible by embed_dim ({args.embed_dim or 32})")
if args.semantic_token_heads and not args.latent_disentangle:
    raise ValueError("--semantic_token_heads requires --latent_disentangle")
if args.micro_pattern:
    if not args.anchor_relative_decode:
        print("[INFO] --micro_pattern requires --anchor_relative_decode. Enabling.")
        args.anchor_relative_decode = True
    if not args.position_scaffold:
        print("[INFO] --micro_pattern requires --position_scaffold. Enabling.")
        args.position_scaffold = True
    if args.scaffold_mode != 'hilbert_block':
        raise ValueError(
            "--micro_pattern requires --scaffold_mode hilbert_block: the canonical "
            "slot index (i % g) only aligns with the decoder layout and the per-token "
            "anchor when token_ids = arange(N)//g, which is the hilbert_block scaffold. "
            f"Got scaffold_mode='{args.scaffold_mode}'.")
if args.opacity_pos_weight > 0 and not args.set_loss:
    raise ValueError(
        "--opacity_pos_weight needs --set_loss: the opacity-weighted position term "
        "reuses the per-block Sinkhorn matching to pair predicted and target Gaussians. "
        "Enable --set_loss, or set --opacity_pos_weight 0.")
if args.position_layout_residual and not args.scene_layout_head:
    args.scene_layout_head = True
if args.token_cond and 'B' in args.token_cond_approach.upper() and not args.scene_layout_head:
    args.scene_layout_head = True
# Chamfer matches on the position columns as passed. In scaffold / layout-residual
# modes those columns are residuals, so NN matching would be on residual coords.
if args.use_chamfer_loss and (args.position_scaffold or args.position_layout_residual):
    print("[WARNING] --use_chamfer_loss with position_scaffold/position_layout_residual: "
          "Chamfer matches on residual position columns, which is likely not intended. "
          "Use chamfer with absolute-position targets (no scaffold/residual).")

if args.anchor_relative_decode and not args.position_scaffold:
    print("[INFO] --anchor_relative_decode requires --position_scaffold. Enabling.")
    args.position_scaffold = True

need_scaffold_data = args.position_scaffold
semantic_requested    = (args.semantic_mode != 'none')
semantic_loss_enabled = (args.segment_loss_weight > 0 or args.instance_loss_weight > 0)
enable_semantic       = semantic_requested and semantic_loss_enabled
effective_semantic_mode = args.semantic_mode if enable_semantic else 'none'
need_segment_labels = (enable_semantic or args.scene_semantic_head or args.predict_seg_labels)

# ============================================================================
# ACCELERATE
# ============================================================================
_ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=False)
accelerator = Accelerator(kwargs_handlers=[_ddp_kwargs])

# ============================================================================
# W&B
# ============================================================================
wandb_enabled = False
if args.use_wandb and accelerator.is_main_process:
    try:
        import wandb
        job_id   = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"can3tok_{job_id}"
        flags = [
            (args.color_residual,             "_colorresidual"),
            (args.latent_disentangle,         f"_disent{args.semantic_dims}"),
            (args.decoder_zs_cross_attn,      "_zsCA"),
            (args.decoder_fourier_pe,         "_fourierpe"),
            (args.scene_layout_head,          "_layout"),
            (args.semantic_token_heads,       "_semTok"),
            (args.z_s_infonce_weight > 0,     "_zsNCE"),
            (args.zs_token_infonce_weight > 0,  "_zsTokNCE"),
            (args.decoder_layout_cross_attn,    "_layCA"),
            (args.decoder_layout_additive,      "_layAdd"),
            (args.zs_layout_infonce_weight > 0,   "_layNCE"),
            (args.zs_pool_infonce_weight > 0,      "_poolNCE"),
            (args.morton_order,               "_morton"),
            (args.use_chamfer_loss,           "_chamfer"),
            (enable_semantic,                 f"_pgNCE{args.segment_loss_weight}"),
        ]
        for flag, label in flags:
            if flag: run_name += label
        run_name += f"_{args.train_data}_inferencefixed"
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                               name=run_name, config=vars(args))
        wandb_enabled = True
        print("W&B enabled")
    except Exception as e:
        print(f"W&B failed: {e}")

# ============================================================================
# DEVICE + PATHS
# ============================================================================
device    = accelerator.device
data_path = "/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs"

job_id = os.environ.get('SLURM_JOB_ID', None)
tag    = (f"RGB_job_{job_id}_{effective_semantic_mode}" if job_id
          else f"RGB_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
flags = [
    (args.color_residual,             "_colorresidual"),
    (args.latent_disentangle,         f"_disent{args.semantic_dims}"),
    (args.decoder_zs_cross_attn,      "_zsCA"),
    (args.decoder_fourier_pe,         "_fourierpe"),
    (args.scene_layout_head,          "_layout"),
    (args.semantic_token_heads,       "_semTok"),
    (args.z_s_infonce_weight > 0,     "_zsNCE"),
    (args.zs_token_infonce_weight > 0,  "_zsTokNCE"),
    (args.decoder_layout_cross_attn,    "_layCA"),
    (args.decoder_layout_additive,      "_layAdd"),
    (args.zs_layout_infonce_weight > 0,   "_layNCE"),
    (args.zs_pool_infonce_weight > 0,      "_poolNCE"),
    (args.morton_order,               "_morton"),
    (args.use_chamfer_loss,           "_chamfer"),
    (enable_semantic,                 f"_pgNCE"),
]
for flag, label in flags:
    if flag: tag += label
tag += f"_{args.train_data}_inferencefixed"

save_path = f"/home/yli11/scratch-project/Hafeez_thesis/Can3Tok/checkpoints_stage1/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================
if accelerator.is_main_process:
    print(f"\n{'='*70}")
    print(f"CAN3TOK — train_data='{args.train_data}'")
    print(f"  decoder_zs_cross_attn={args.decoder_zs_cross_attn}")
    print(f"  color_residual={args.color_residual}")
    print(f"  latent_disentangle={args.latent_disentangle} semantic_dims={args.semantic_dims}")
    print(f"  scene_layout_head={args.scene_layout_head}")
    print(f"  decoder_fourier_pe={args.decoder_fourier_pe}")
    print(f"  token_local_decoder={args.token_local_decoder}")
    print(f"  crop_percentile={args.crop_percentile}")
    print(f"  morton_order={args.morton_order}"
          f"{' curve='+args.order_curve.upper()+(' frame=[-%g,%g]'%(args.order_frame_radius,args.order_frame_radius) if args.order_frame_radius>0 else ' frame=per-scene') if args.morton_order else ''}")
    print(f"  RECON OBJECTIVE = {'CHAMFER (permutation-invariant)' if args.use_chamfer_loss else 'element-wise L2'}"
          f"{' chunk='+str(args.chamfer_chunk) if args.use_chamfer_loss else ''}")
    print(f"  cross_recon={args.cross_recon_weight} ortho={args.ortho_weight}")
    print(f"  Save: {save_path}")
    print(f"{'='*70}\n")

# ============================================================================
# MODEL
# ============================================================================
print("Loading model config...")
config_path  = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config = get_config_from_file(config_path).model
p = model_config.params.shape_module_cfg.params
if args.embed_dim is not None:
    p.embed_dim = int(args.embed_dim)
# Structured-latent total = 512 (block count) * embed_dim. Pinned token count means
# embed_dim is a pure capacity knob; report what was selected.
print(f"[latent] embed_dim={p.embed_dim} -> 512 tokens x {p.embed_dim} = "
      f"{512 * p.embed_dim} latent numbers "
      f"({'baseline' if p.embed_dim == 32 else f'{(512*p.embed_dim)//16384}x baseline'})")
p.semantic_mode           = effective_semantic_mode
p.color_residual          = args.color_residual
p.scene_semantic_head     = args.scene_semantic_head
p.position_scaffold       = args.position_scaffold
p.latent_disentangle      = args.latent_disentangle
p.semantic_dims           = args.semantic_dims
p.scene_layout_head       = args.scene_layout_head
p.jepa_idea1              = args.jepa_idea1
p.decoder_pos_enc         = args.decoder_pos_enc
p.predict_seg_labels      = args.predict_seg_labels
p.token_cond              = args.token_cond
p.token_cond_approach     = args.token_cond_approach
p.query_decoder           = args.query_decoder
p.decoder_fourier_pe      = args.decoder_fourier_pe
p.token_cond_adaln        = args.token_cond_adaln
p.semantic_token_heads    = args.semantic_token_heads
p.token_local_decoder     = args.token_local_decoder
p.anchor_relative_decode  = args.anchor_relative_decode
p.anchor_teacher_force    = args.anchor_teacher_force
p.offset_scale_init       = args.offset_scale_init
p.micro_pattern           = args.micro_pattern
p.micro_pattern_rotation  = not args.micro_pattern_no_rotation
p.micro_offset_scale      = args.micro_offset_scale
p.structured_latent       = args.structured_latent
p.local_encoder           = args.local_encoder
p.local_window            = args.local_window
p.decoder_zs_cross_attn       = args.decoder_zs_cross_attn
p.decoder_layout_cross_attn   = args.decoder_layout_cross_attn
p.decoder_layout_additive     = args.decoder_layout_additive
p.structured_layout_tokens    = args.structured_layout_tokens
p.position_layout_residual    = args.position_layout_residual

cfg_point_feats = p.point_feats
expected_feats  = 12 if args.label_input else 11
if cfg_point_feats != expected_feats:
    raise ValueError(f"point_feats mismatch: yaml={cfg_point_feats}, expected {expected_feats}.")
print(f"  point_feats={cfg_point_feats} OK")

from model.michelangelo.models.tsal.sal_perceiver_dist_changes import set_num_gaussians
set_num_gaussians(args.num_gaussians)   # must run BEFORE the model is built
if args.pos_cond_heads:
    from model.michelangelo.models.tsal.sal_perceiver_dist_changes import set_pos_cond_heads
    set_pos_cond_heads(enabled=True, n_freqs=args.pos_cond_n_freqs, sigma=args.pos_cond_sigma,
                       pos_scale=args.pos_cond_pos_scale, hidden=args.pos_cond_hidden,
                       color=bool(args.pos_cond_color), rotation=bool(args.pos_cond_rotation))
gs_autoencoder = instantiate_from_config(model_config)
gs_autoencoder.to(device)
optimizer = torch.optim.AdamW(
    gs_autoencoder.parameters(), lr=args.lr, betas=[0.9,0.999], weight_decay=args.weight_decay)

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================
start_epoch   = 0
best_val_loss = float('inf')
best_epoch    = 0

if args.resume_checkpoint:
    print(f"\nResuming from: {args.resume_checkpoint}")
    ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
    for flag_name, current_val, default_val in [
        ('color_residual',             args.color_residual,          False),
        ('label_input',                args.label_input,             False),
        ('latent_disentangle',         args.latent_disentangle,      False),
        ('semantic_dims',              args.semantic_dims,           512),
        ('position_layout_residual',   args.position_layout_residual, False),
    ]:
        saved = ckpt.get(flag_name, default_val)
        if saved != current_val:
            raise ValueError(f"{flag_name} mismatch: ckpt={saved}, current={current_val}.")
    strict = all([
        ckpt.get('scene_semantic_head',   False) == args.scene_semantic_head,
        ckpt.get('semantic_mode', 'none') == effective_semantic_mode,
        ckpt.get('scene_layout_head',     False) == args.scene_layout_head,
        ckpt.get('decoder_fourier_pe',    False) == args.decoder_fourier_pe,
        ckpt.get('token_cond',            False) == args.token_cond,
        ckpt.get('token_cond_adaln',      False) == args.token_cond_adaln,
        ckpt.get('semantic_token_heads',  False) == args.semantic_token_heads,
        ckpt.get('token_local_decoder',  False) == args.token_local_decoder,
        ckpt.get('decoder_zs_cross_attn', False) == args.decoder_zs_cross_attn,
        ckpt.get('micro_pattern',         False) == args.micro_pattern,
    ])
    if not strict:
        print(f"  Architecture changed — loading strict=False")
    gs_autoencoder.load_state_dict(ckpt['model_state_dict'], strict=strict)
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("  Loaded optimizer state from checkpoint.")
    else:
        print("  [WARN] checkpoint has no 'optimizer_state_dict' (this is a best/model-only "
              "checkpoint, e.g. best_model.pth) — starting with a FRESH optimizer. This is "
              "expected and fine for fine-tuning (e.g. switching on the render loss); the "
              "Adam moments will rebuild within a few steps.")
    start_epoch   = ckpt.get('epoch', 0) + 1
    if args.resume_epoch is not None: start_epoch = args.resume_epoch
    best_val_loss = ckpt.get('val_l2_error', ckpt.get('best_val_l2', float('inf')))
    best_epoch    = ckpt.get('epoch', 0)
    print(f"  Resumed epoch {start_epoch} (val L2: {best_val_loss:.4f})")

# ============================================================================
# LR SCHEDULER
# ============================================================================
def build_lr_lambda(warmup_steps, total_steps, lr_min_ratio):
    """Single cosine decay with linear warmup (original behaviour)."""
    cosine_steps = max(total_steps - warmup_steps, 1)
    def f(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)
        t = step - warmup_steps
        return lr_min_ratio + (1-lr_min_ratio) * 0.5*(1 + math.cos(math.pi*t/cosine_steps))
    return f

def build_lr_lambda_restart(warmup_steps, restart_T0_steps, lr_min_ratio):
    """Cosine warm restarts with linear warmup."""
    T = max(restart_T0_steps, 1)
    def f(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)
        t = (step - warmup_steps) % T
        cosine_val = 0.5 * (1 + math.cos(math.pi * t / (T / 2)))
        return lr_min_ratio + (1 - lr_min_ratio) * cosine_val
    return f

# NOTE: _bpe, scheduler created AFTER datasets so _bpe reflects combined size.

# ============================================================================
# DATASETS
# ============================================================================
from gs_dataset_scenesplat import gs_dataset
# Per-scene Gaussian count is a class attribute read as self.TARGET_POINTS; set it once
# here so every dataset instance samples args.num_gaussians points and the Hilbert-block
# scaffold (g = ceil(N / 512)) scales with it. SCAFFOLD_TOKENS stays 512 (latent tokens).
gs_dataset.TARGET_POINTS = args.num_gaussians

_train_only_kwargs = dict(random_subset_seed=args.random_subset_seed,
                          preload=args.preload,
                          aug_yaw=args.aug_yaw, aug_yaw_axis=args.aug_yaw_axis,
                          aug_yaw_max_deg=args.aug_yaw_max_deg)

_ds_kwargs = dict(
    resol=100,
    sampling_method=args.sampling_method,
    normalize=args.use_canonical_norm,
    normalize_colors=args.normalize_colors,
    use_chunk_norm_factor=args.chunk_norm_factor,
    target_radius=10.0,
    scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input,
    color_residual=args.color_residual,
    position_scaffold=args.position_scaffold,
    scaffold_mode=args.scaffold_mode,
    scene_layout_head=args.scene_layout_head,
    jepa_idea1=args.jepa_idea1,
    position_layout_residual=args.position_layout_residual,
    crop_percentile=args.crop_percentile,   # spatial crop before opacity sampling
    morton_order=args.morton_order,         # Z-order reorder of selected Gaussians
    order_curve=args.order_curve,           # which space-filling curve (hilbert default)
    order_frame_radius=args.order_frame_radius,  # fixed canonical frame for the sort
    canonical_voxel=args.canonical_voxel,   # density-adaptive voxel canonicalization
    voxel_res=args.voxel_res,
    voxel_snap=args.voxel_snap,
    sample_voxel_res=args.sample_voxel_res,
)
# NOTE: preload is intentionally NOT in _ds_kwargs. It lives in _train_only_kwargs
# (so the big train/extra sets honor --no_preload for memory), while the val and
# chunk-val sets below pass only _ds_kwargs and therefore default to preload=True --
# i.e. they are sampled ONCE at construction and frozen for the whole run, so the
# validation metric is a fixed, comparable measure rather than re-sampled per eval.

_chunk_root = os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6")
_full_root  = os.path.join(data_path, "train")

# Training dataset
if args.train_data == 'chunks':
    if accelerator.is_main_process:
        print(f"\n--- Training Dataset: CHUNKS ({_chunk_root}) ---")
    gs_dataset_train = gs_dataset(
        root=_chunk_root, random_permute=True, train=True,
        max_scenes=args.train_scenes, skip_scenes=None,
        **_train_only_kwargs, **_ds_kwargs)
    _n_train_chunks = len(gs_dataset_train)

elif args.train_data == 'full':
    if accelerator.is_main_process:
        print(f"\n--- Training Dataset: FULL SCENES ({_full_root}) ---")
    gs_dataset_train = gs_dataset(
        root=_full_root, random_permute=True, train=True,
        max_scenes=args.train_scenes, skip_scenes=None,
        **_train_only_kwargs, **_ds_kwargs)
    _n_train_chunks = 0

else:  # combined
    _max_full  = max(1, args.train_scenes // 2) if args.train_scenes else None
    _max_chunk = (args.train_scenes - _max_full)  if args.train_scenes else None
    if accelerator.is_main_process:
        print(f"\n--- Training Dataset: COMBINED (full + chunks) ---")
    _ds_full  = gs_dataset(root=_full_root,  random_permute=True, train=True,
                           max_scenes=_max_full,  skip_scenes=None,
                           **_train_only_kwargs, **_ds_kwargs)
    _ds_chunk = gs_dataset(root=_chunk_root, random_permute=True, train=True,
                           max_scenes=_max_chunk, skip_scenes=None,
                           **_train_only_kwargs, **_ds_kwargs)
    gs_dataset_train = Data.ConcatDataset([_ds_full, _ds_chunk])
    _n_train_chunks  = len(_ds_chunk)

# Val full scenes (PRIMARY — thesis target)
if accelerator.is_main_process:
    print(f"\n--- Validation Dataset: val/ (held-out full scenes) ---")
gs_dataset_val = gs_dataset(
    root=os.path.join(data_path, "val"),
    random_permute=False, train=False,
    max_scenes=args.val_scenes, skip_scenes=None, **_ds_kwargs)

# Val held-out chunks (in-distribution diagnostic)
gs_dataset_val_chunk  = None
valChunkDataLoader    = None
_has_chunk_val        = False

class _ChunkSplitError(Exception):
    """Chunk train/val split is contaminated (train and val share chunks). Fatal:
    must never be swallowed by the soft dataset-construction error handler below."""
    pass

if args.train_data in ('chunks', 'combined') and _n_train_chunks > 0:
    if accelerator.is_main_process:
        print(f"\n--- Validation Dataset: held-out chunks "
              f"(skip_scenes={_n_train_chunks}) ---")
    try:
        # Clean split: training takes the first --train_scenes sorted chunks, so the
        # chunks sorted AFTER that (skip_scenes=_n_train_chunks) are disjoint from
        # training. With 3888 total and train_scenes=3800 this yields the last 88.
        # chunk_val_scenes=None -> take all remaining; otherwise cap at that many.
        gs_dataset_val_chunk = gs_dataset(
            root=_chunk_root,
            random_permute=False, train=False,
            skip_scenes=_n_train_chunks,
            max_scenes=args.chunk_val_scenes,
            **_ds_kwargs)
        if len(gs_dataset_val_chunk) > 0:
            _has_chunk_val = True
            # ── HARD DISJOINTNESS GUARANTEE ──────────────────────────────────────
            # Training chunks and held-out val chunks must never overlap. Overlap
            # occurs when --random_subset_seed is set: training then samples a RANDOM
            # train_scenes out of all chunks, while val skips the first train_scenes
            # SORTED — so the skipped range is mostly inside the random training set.
            # Reject it loudly rather than report a contaminated chunk metric.
            _train_chunk_dirs = set(
                gs_dataset_train.scene_dirs if args.train_data == 'chunks'
                else _ds_chunk.scene_dirs)
            _overlap = _train_chunk_dirs & set(gs_dataset_val_chunk.scene_dirs)
            if _overlap:
                raise _ChunkSplitError(
                    f"\n{'!'*70}\n"
                    f"CHUNK SPLIT CONTAMINATED: {len(_overlap)} chunk(s) are in BOTH the "
                    f"training set and the held-out chunk-val set.\n"
                    f"Cause: --random_subset_seed={args.random_subset_seed} makes training "
                    f"sample a RANDOM {args.train_scenes} chunks, while the chunk-val skips "
                    f"the first {args.train_scenes} SORTED chunks, so they overlap.\n"
                    f"Fix:   unset --random_subset_seed (set RANDOM_SUBSET_SEED=\"\" in the "
                    f"job). Training then takes the first {args.train_scenes} sorted chunks "
                    f"and val takes the remaining {len(gs_dataset_val_chunk)} — disjoint by "
                    f"construction.\n{'!'*70}")
            if accelerator.is_main_process:
                print(f"  Clean chunk split verified: {len(_train_chunk_dirs)} train / "
                      f"{len(gs_dataset_val_chunk)} val chunks, 0 overlap ✓")
        else:
            if accelerator.is_main_process:
                print(f"  [INFO] No held-out chunks available. Chunk val disabled.")
            gs_dataset_val_chunk = None
    except _ChunkSplitError:
        raise  # contamination is fatal — never disable-and-continue on a dirty split
    except Exception as e:
        if accelerator.is_main_process:
            print(f"  [WARNING] Could not create held-out chunk val dataset: {e}")
        gs_dataset_val_chunk = None
        _has_chunk_val = False

# Extra training datasets (multi-path support)
_extra_train_datasets = []
_extra_path_list      = []
_extra_n_scenes_map   = {}

if args.extra_train_paths:
    _raw_paths  = [p.strip() for p in args.extra_train_paths.split(':') if p.strip()]
    _raw_scenes = ([s.strip() for s in args.extra_train_scenes.split(':') if s.strip()]
                   if args.extra_train_scenes else [])
    while len(_raw_scenes) < len(_raw_paths):
        _raw_scenes.append('0')
    _raw_scenes = _raw_scenes[:len(_raw_paths)]

    for _ep, _es_str in zip(_raw_paths, _raw_scenes):
        _max_s = (int(_es_str) if _es_str and _es_str != '0' else None)
        if accelerator.is_main_process:
            print(f"\n--- Extra Training Dataset: {os.path.basename(_ep)} ---")
            print(f"    Path       : {_ep}")
            print(f"    Max scenes : {'all' if _max_s is None else _max_s}")
            print(f"    Semantics  : disabled (label_dist=zeros, segment=-1)")
        try:
            _extra_ds = gs_dataset(
                root=_ep, random_permute=True, train=True,
                max_scenes=_max_s, skip_scenes=None,
                disable_semantics=True,
                preload=args.preload,
                aug_yaw=args.aug_yaw, aug_yaw_axis=args.aug_yaw_axis,
                aug_yaw_max_deg=args.aug_yaw_max_deg,
                **_ds_kwargs)
            _extra_train_datasets.append(_extra_ds)
            _extra_path_list.append(_ep)
            _extra_n_scenes_map[_ep] = len(_extra_ds)
        except Exception as _exc:
            if accelerator.is_main_process:
                print(f"  [WARNING] Could not load extra dataset at {_ep}: {_exc}  (skipping)")

if _extra_train_datasets:
    _gs_dataset_train_combined = Data.ConcatDataset(
        [gs_dataset_train] + _extra_train_datasets)
else:
    _gs_dataset_train_combined = gs_dataset_train

# Scheduler (created here so _bpe uses the actual combined dataset size)
_bpe = max(1, math.ceil(
    len(_gs_dataset_train_combined) / (args.batch_size * accelerator.num_processes)))
_total_steps  = _bpe * args.num_epochs
_elapsed      = _bpe * start_epoch

if args.lr_restart_T0 > 0:
    _restart_T0_steps = args.lr_restart_T0 * _bpe
    _warmup_adj = max(0, args.warmup_steps - _elapsed)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda_restart(
            warmup_steps=_warmup_adj,
            restart_T0_steps=_restart_T0_steps,
            lr_min_ratio=args.lr_min_ratio))
    if accelerator.is_main_process:
        print(f"\n  LR: peak={args.lr:.2e} | floor={args.lr*args.lr_min_ratio:.2e}")
        print(f"  Scheduler: COSINE WARM RESTARTS  T0={args.lr_restart_T0} epochs "
              f"({_restart_T0_steps} steps)  _bpe={_bpe}")
        print(f"  Restart cycle: peak->floor->peak every {args.lr_restart_T0} epochs")
        print(f"  Expected restarts over {args.num_epochs} epochs: "
              f"{args.num_epochs // args.lr_restart_T0}")
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=build_lr_lambda(
        warmup_steps=max(0, args.warmup_steps - _elapsed),
        total_steps=max(_total_steps - _elapsed, 1),
        lr_min_ratio=args.lr_min_ratio))
    if accelerator.is_main_process:
        print(f"\n  LR: peak={args.lr:.2e} | floor={args.lr*args.lr_min_ratio:.2e}")
        print(f"  Scheduler: single cosine  _bpe={_bpe}  total_steps={_total_steps}  "
              f"combined_train_scenes={len(_gs_dataset_train_combined)}")

# DataLoaders
trainDataLoader = Data.DataLoader(
    dataset=_gs_dataset_train_combined, batch_size=args.batch_size,
    shuffle=True, num_workers=9, pin_memory=True, persistent_workers=True)

valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val, batch_size=args.batch_size,
    shuffle=False, num_workers=9, pin_memory=True, persistent_workers=True)

if _has_chunk_val:
    valChunkDataLoader = Data.DataLoader(
        dataset=gs_dataset_val_chunk, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# ============================================================================
# NORMALIZATION VERIFICATION
# ============================================================================
if accelerator.is_main_process:
    print(f"\n{'='*70}")
    print(f"  NORMALIZATION VERIFICATION")
    print(f"{'='*70}")

    def _check_nf(label, dirs, expected_present):
        sample = min(50, len(dirs))
        nf_ok  = sum(1 for d in dirs[:sample]
                     if os.path.exists(os.path.join(d, 'norm_factor.npy')))
        if expected_present:
            status = ('ALL PRESENT — global frame' if nf_ok == sample
                      else f'MISSING in {sample-nf_ok}/{sample} — position WILL NOT converge!')
        else:
            status = ('ABSENT — per-scene fallback (correct for full scenes)'
                      if nf_ok == 0 else f'present in {nf_ok}/{sample} (unusual but OK)')
        print(f"  {label:<30s}: {nf_ok}/{sample}  {status}")
        if expected_present and nf_ok >= 2:
            _ex = dirs[0]
            _nf = np.load(os.path.join(_ex, 'norm_factor.npy'))
            print(f"    Example {os.path.basename(_ex)}: "
                  f"center=({_nf[0]:.3f},{_nf[1]:.3f},{_nf[2]:.3f}) "
                  f"scale={_nf[3]:.4f}")
        return nf_ok == sample if expected_present else True

    if args.train_data == 'chunks':
        _ok_train = _check_nf("Training chunks", gs_dataset_train.scene_dirs, True)
    elif args.train_data == 'combined':
        _ok_train = _check_nf("Training chunks (combined)", _ds_chunk.scene_dirs, True)
        _check_nf("Training full scenes (combined)", _ds_full.scene_dirs, False)
    else:
        _check_nf("Training full scenes", gs_dataset_train.scene_dirs, False)

    _check_nf("Val full scenes (primary)", gs_dataset_val.scene_dirs, False)

    if _has_chunk_val:
        _ok_chunk_val = _check_nf("Val held-out chunks", gs_dataset_val_chunk.scene_dirs, True)

    print(f"{'='*70}\n")

# ============================================================================
# DATASET SUMMARY
# ============================================================================
if accelerator.is_main_process:
    _n_train_main  = len(gs_dataset_train)
    _n_train_extra = sum(len(d) for d in _extra_train_datasets)
    _n_train_total = len(_gs_dataset_train_combined)
    n_val          = len(gs_dataset_val)
    print(f"{'='*70}")
    print(f"  DATASET SUMMARY  (train_data='{args.train_data}')")
    print(f"{'='*70}")
    if _extra_train_datasets:
        print(f"  Training scenes    : {_n_train_total}  "
              f"({_n_train_main} main  +  {_n_train_extra} extra)")
        for _ep in _extra_path_list:
            print(f"    + {os.path.basename(_ep)}: {_extra_n_scenes_map[_ep]} scenes  "
                  f"(semantics disabled)")
    else:
        print(f"  Training scenes    : {_n_train_main}")
    print(f"  Val full scenes    : {n_val}  (PRIMARY — thesis target)")
    if _has_chunk_val:
        print(f"  Val held-out chunks: {len(gs_dataset_val_chunk)}  "
              f"(CLEAN split: first {_n_train_chunks} sorted = train, "
              f"remaining = val, disjoint)")
    else:
        print(f"  Val held-out chunks: N/A")
    print(f"  Gaussian order     : {(args.order_curve.upper()+' curve') if args.morton_order else 'opacity rank'}")
    print(f"  Recon assignment   : {'CHAMFER (order-free)' if args.use_chamfer_loss else 'element-wise (slot)'}")
    print(f"  Recon geom metric  : {args.geom_loss}"
          + ("" if args.geom_loss == 'l2'
             else f"  (gauge-invariant covariance; eps={args.geom_eps}, "
                  f"ns_iters={args.geom_ns_iters}, shape_w={args.geom_shape_weight})"))
    print(f"  Yaw augmentation   : {('ON (axis=%s, +/-%.0f deg)' % (args.aug_yaw_axis, args.aug_yaw_max_deg)) if args.aug_yaw else 'off'}")
    print(f"  Covariance metric  : {'on (Bures + log-Euclidean + anisotropy each eval)' if args.cov_metric else 'off'}")
    print(f"  Spatial crop       : {('inner %.0f%%' % args.crop_percentile) if args.crop_percentile < 100 else 'disabled'}")
    print(f"  Batches/epoch      : {_bpe}  "
          f"(batch={args.batch_size} x {accelerator.num_processes} GPUs)")
    print(f"{'='*70}\n")

# ============================================================================
# ACCELERATE PREPARE
# ============================================================================
if _has_chunk_val:
    (gs_autoencoder, optimizer, trainDataLoader,
     valDataLoader, valChunkDataLoader, scheduler) = accelerator.prepare(
        gs_autoencoder, optimizer, trainDataLoader,
        valDataLoader, valChunkDataLoader, scheduler)
else:
    (gs_autoencoder, optimizer, trainDataLoader,
     valDataLoader, scheduler) = accelerator.prepare(
        gs_autoencoder, optimizer, trainDataLoader,
        valDataLoader, scheduler)

raw_model = accelerator.unwrap_model(gs_autoencoder)

# ============================================================================
# MIXED PRECISION SETUP
# ============================================================================
_mp             = accelerator.mixed_precision
_autocast_dtype = (torch.bfloat16 if _mp == 'bf16' else
                   torch.float16  if _mp == 'fp16' else torch.float32)
_use_autocast   = (_mp != 'no')
if accelerator.is_main_process:
    print(f"\n{'='*70}")
    print(f"  GPU / COMPUTE SETUP")
    print(f"{'='*70}")
    print(f"  Num GPUs (accelerator processes) : {accelerator.num_processes}")
    print(f"  Distributed type                 : {accelerator.distributed_type}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"  GPU {i}: {props.name}  |  {total_mem_gb:.1f} GB VRAM  "
                  f"|  SM {props.major}.{props.minor}  "
                  f"|  {props.multi_processor_count} SMs")
    else:
        print(f"  CUDA not available — running on CPU")
    print(f"  Mixed precision : {_mp}")
    print(f"  Autocast dtype  : {_autocast_dtype}")
    print(f"  Autocast enabled: {_use_autocast}")
    print(f"{'='*70}\n")

# ============================================================================
# CHECKPOINT METADATA
# ============================================================================
_ckpt_meta = {
    'semantic_mode':              effective_semantic_mode,
    'enable_semantic':            enable_semantic,
    'label_input':                args.label_input,
    'color_residual':             args.color_residual,
    'scene_semantic_head':        args.scene_semantic_head,
    'position_scaffold':          args.position_scaffold,
    'latent_disentangle':         args.latent_disentangle,
    'semantic_dims':              args.semantic_dims,
    'scene_layout_head':          args.scene_layout_head,
    'decoder_fourier_pe':         args.decoder_fourier_pe,
    'token_cond':                 args.token_cond,
    'token_cond_approach':        args.token_cond_approach,
    'token_cond_adaln':           args.token_cond_adaln,
    'semantic_token_heads':       args.semantic_token_heads,
    'token_local_decoder':        args.token_local_decoder,
    'scaffold_mode':              args.scaffold_mode,
    'anchor_relative_decode':     args.anchor_relative_decode,
    'anchor_teacher_force':       args.anchor_teacher_force,
    'offset_scale_init':          args.offset_scale_init,
    'micro_pattern':              args.micro_pattern,
    'micro_pattern_rotation':     (not args.micro_pattern_no_rotation),
    'micro_offset_scale':         args.micro_offset_scale,
    'structured_latent':          args.structured_latent,
    'local_encoder':              args.local_encoder,
    'local_window':               args.local_window,
    'num_gaussians':              args.num_gaussians,
    'pos_cond_heads':             args.pos_cond_heads,
    'pos_cond_color':             bool(args.pos_cond_color),
    'pos_cond_rotation':          bool(args.pos_cond_rotation),
    'decoder_zs_cross_attn':      args.decoder_zs_cross_attn,
    'z_s_infonce_weight':         args.z_s_infonce_weight,
    'z_s_infonce_temperature':    args.z_s_infonce_temperature,
    'z_s_infonce_delta':          args.z_s_infonce_delta,
    'zs_token_infonce_weight':    args.zs_token_infonce_weight,
    'zs_token_infonce_temperature': args.zs_token_infonce_temperature,
    'decoder_layout_cross_attn':  args.decoder_layout_cross_attn,
    'decoder_layout_additive':    args.decoder_layout_additive,
    'zs_layout_infonce_weight':   args.zs_layout_infonce_weight,
    'zs_pool_infonce_weight':      args.zs_pool_infonce_weight,
    'zs_pool_infonce_temperature': args.zs_pool_infonce_temperature,
    'structured_layout_tokens':   args.structured_layout_tokens,
    'zs_layout_infonce_temperature': args.zs_layout_infonce_temperature,
    'inference_fixed':            True,
    'position_layout_residual':   args.position_layout_residual,
    'mean_color_weight':          args.mean_color_weight,
    'scene_semantic_weight':      args.scene_semantic_weight,
    'anchor_loss_weight':         args.anchor_loss_weight,
    'cross_recon_weight':         args.cross_recon_weight,
    'ortho_weight':               args.ortho_weight,
    'layout_loss_weight':         args.layout_loss_weight,
    'color_loss_weight':          args.color_loss_weight,
    'scale_penalty_weight':       args.scale_penalty_weight,
    'scale_penalty_threshold':    args.scale_penalty_threshold,
    'use_canonical_norm':         args.use_canonical_norm,
    'chunk_norm_factor':          args.chunk_norm_factor,
    'scale_norm_mode':            args.scale_norm_mode,
    'geom_loss':                  args.geom_loss,
    'geom_eps':                   args.geom_eps,
    'geom_ns_iters':              args.geom_ns_iters,
    'geom_shape_weight':          args.geom_shape_weight,
    'cov_metric':                 args.cov_metric,
    'aug_yaw':                    args.aug_yaw,
    'aug_yaw_axis':               args.aug_yaw_axis,
    'aug_yaw_max_deg':            args.aug_yaw_max_deg,
    'train_data':                 args.train_data,
    'n_train_chunks':             _n_train_chunks,
    'chunk_val_scenes':           args.chunk_val_scenes,
    'kl_anneal_steps':            args.kl_anneal_steps,
    'crop_percentile':            args.crop_percentile,
    'morton_order':               args.morton_order,
    'order_curve':                args.order_curve,
    'order_frame_radius':         args.order_frame_radius,
    'use_chamfer_loss':           args.use_chamfer_loss,
    'chamfer_chunk':              args.chamfer_chunk,
}

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, raw_model, dataloader, device, accelerator,
                   epoch=None, do_vis=True, vis_tag=None):
    """
    Evaluate the autoencoder on a dataloader.

    vis_tag : optional str. When set (e.g. "chunk"), all saved visualisations
        (reconstructed PLYs, PCA PLYs) go to suffixed folders
        reconstructed_gaussians_<tag>/ and pca_visualisations_<tag>/ so the
        held-out-chunk reconstructions do not overwrite the full-scene ones.
        When None (default), the original folder names are used unchanged.

    REPORTING: avg_l2_error and the per-attribute losses (position/color/...) are
    reported as the RAW torch.norm per batch, averaged over the number of batches
    (divide by n_batches, NOT n_scenes). This matches the training-side convention
    so train and val numbers are on the same scale.
    """
    model.eval()
    _eval_dtype    = _autocast_dtype
    _eval_autocast = _use_autocast

    total_l2 = total_kl = total_color = total_scene_sem = 0.0
    total_cov_bures = total_cov_le = total_aniso = 0.0   # gauge-invariant covariance metric
    total_anchor = total_layout = total_seg = total_z_s_nce = total_zs_tok_nce = total_zs_lay_nce = 0.0
    per_param    = {k: 0.0 for k in PARAM_SLICES}
    n_scenes     = 0
    n_batches    = 0   # for raw, training-style averaging of recon / per-param

    recon_preds  = []; recon_means  = []
    gt_preds     = []        # ground-truth 14-dim Gaussians, saved ONCE at epoch 0
    pca_input    = []; pca_recon    = []
    pca_sem_feat = []
    z_s_proj_acc = []; label_dist_acc = []
    zs_tokens_acc = []; zs_layout_acc = []; zs_pool_acc = []

    _do_vis    = do_vis
    # Suffix for output folders so chunk-val visualisations land in their own
    # directory (reconstructed_gaussians_chunk/ etc.) instead of overwriting the
    # full-scene ones. Empty string -> original paths (backward compatible).
    _vsfx      = f"_{vis_tag}" if vis_tag else ""
    do_recon   = (_do_vis and epoch is not None and epoch % args.recon_ply_freq == 0)
    # Ground truth never changes, so save it ONCE at epoch 0 (for the SAME scenes the
    # recon uses) into ground_truth_gaussians[_<tag>]/. Every later recon epoch is then
    # compared against this single GT folder, with no repeated GT writes.
    do_gt      = (do_recon and epoch == 0)
    do_pca     = (_do_vis and epoch is not None and epoch % args.pca_vis_freq   == 0)
    do_sem_pca = (do_pca and enable_semantic)
    do_z_s_vis     = (do_pca and raw_model.shape_model.z_s_infonce_head is not None)
    do_zs_tok_vis  = (do_pca and args.zs_token_infonce_weight > 0 and args.latent_disentangle)
    _any_B_eval    = args.decoder_layout_cross_attn or args.decoder_layout_additive
    do_zs_lay_vis  = (do_pca and _any_B_eval)
    do_zs_pool_vis = (do_pca and args.zs_pool_infonce_weight > 0)

    _pos_abs_min = _pos_abs_max = _pos_gt_range = 0.0

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            UV_gs_batch   = batch_data['features'].float().to(device)
            mean_color_gt = batch_data['mean_color'].float().to(device)
            label_dist_v  = batch_data['label_dist'].float().to(device)
            B = UV_gs_batch.shape[0]

            sa_gpu  = (batch_data['scaffold_anchors'].float().to(device)
                       if need_scaffold_data else None)
            sti_gpu = (batch_data['scaffold_token_ids'].long().to(device)
                       if args.position_scaffold else None)

            _rsf = True if do_sem_pca else None

            with torch.autocast('cuda', dtype=_eval_dtype, enabled=_eval_autocast):
                (shape_embed, mu, log_var, z,
                 UV_gs_recover, pg_feats) = model(
                    UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:,:,:3],
                    scaffold_anchors=sa_gpu, scaffold_token_ids=sti_gpu,
                    return_semantic_features=_rsf)

            mcp  = raw_model.shape_model.last_mean_color_pred
            ssp  = raw_model.shape_model.last_scene_semantic_pred
            anch = raw_model.shape_model.last_predicted_anchors_from_tokens
            slp  = raw_model.shape_model.last_scene_layout_pred
            sgp  = raw_model.shape_model.last_seg_pred
            zsp  = raw_model.shape_model.last_z_s_infonce_proj

            target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            if args.anchor_relative_decode:
                target  = target_abs
                pred_3d = UV_gs_recover.reshape(B,-1,14)
            elif args.position_scaffold:
                pos_off = batch_data['position_offsets'].float().to(device)
                target  = target_abs.clone(); target[:,:,0:3] = pos_off
                pred_3d = UV_gs_recover.reshape(B,-1,14).clone()
                if anch is not None and sti_gpu is not None:
                    idx_3d = sti_gpu.unsqueeze(-1).expand(-1,-1,3)
                    pred_3d[:,:,0:3] -= torch.gather(anch, 1, idx_3d)
            elif args.position_layout_residual:
                pos_res = batch_data['position_residuals'].float().to(device)
                target  = target_abs.clone(); target[:,:,0:3] = pos_res
                pred_3d = UV_gs_recover.reshape(B,-1,14)
            else:
                target  = target_abs
                pred_3d = UV_gs_recover.reshape(B,-1,14)

            pred_abs   = UV_gs_recover.reshape(B,-1,14)
            if args.set_loss:
                _set_g = args.set_block_size if args.set_block_size > 0 else -(-args.num_gaussians // 512)
                recon_loss = compute_reconstruction_loss_set(
                    pred_3d, target, B, args.color_loss_weight, args.geom_shape_weight, _set_g,
                    args.set_pos_weight, args.set_opa_weight,
                    args.set_sinkhorn_eps, args.set_sinkhorn_iters)
            elif args.geom_loss == 'l2':
                if args.use_chamfer_loss:
                    recon_loss = chamfer_reconstruction_loss(
                        pred_3d, target, B, args.color_loss_weight, args.chamfer_chunk)
                else:
                    recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
            else:
                if args.use_chamfer_loss:
                    recon_loss = chamfer_reconstruction_loss_geom(
                        pred_3d, target, B, args.color_loss_weight, args.geom_loss,
                        args.chamfer_chunk, args.geom_eps, args.geom_ns_iters, args.geom_shape_weight)
                else:
                    recon_loss = compute_reconstruction_loss_geom(
                        pred_3d, target, B, args.color_loss_weight, args.geom_loss,
                        args.geom_eps, args.geom_ns_iters, args.geom_shape_weight)
            kl_loss    = -0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp(), dim=1)

            if mcp is not None and args.color_residual:
                total_color += F.mse_loss(mcp.float(), mean_color_gt).item() * B
            if ssp is not None and args.scene_semantic_head:
                p_s = batch_data['label_dist'].float().to(device)
                total_scene_sem += scene_semantic_kl_loss(ssp.float(), p_s).item() * B
            if anch is not None and args.position_scaffold:
                total_anchor += F.mse_loss(anch.float(), sa_gpu).item() * B
            if slp is not None and args.scene_layout_head:
                gt_c = batch_data['category_centroids'].float().to(device)
                gt_v = batch_data['category_valid'].float().to(device)
                total_layout += compute_layout_loss(slp.float(), gt_c, gt_v).item() * B
            if args.predict_seg_labels and sgp is not None:
                total_seg += compute_seg_pred_loss(
                    sgp, batch_data['segment_labels'].long().to(device)).item() * B

            z_s_tokens_eval = None
            if args.latent_disentangle and args.semantic_dims > 0:
                _ed_eval = raw_model.shape_model.embed_dim
                _n_tok = args.semantic_dims // _ed_eval
                z_s_tokens_eval = z.reshape(B, -1, _ed_eval)[:, :_n_tok, :].detach()
            if args.zs_token_infonce_weight > 0 and z_s_tokens_eval is not None:
                zl_tok, _ = compute_zs_token_infonce_loss(
                    z_s_tokens_eval, label_dist_v, args.zs_token_infonce_temperature)
                total_zs_tok_nce += zl_tok.item() * B

            z_lay_proj_eval = raw_model.shape_model.last_z_layout_proj
            if args.zs_layout_infonce_weight > 0 and z_lay_proj_eval is not None:
                zl_lay, _ = compute_zs_layout_infonce_loss(
                    z_lay_proj_eval, label_dist_v, args.zs_layout_infonce_temperature)
                total_zs_lay_nce += zl_lay.item() * B

            if args.z_s_infonce_weight > 0 and zsp is not None:
                zl, _ = compute_scene_infonce_loss(zsp, label_dist_v,
                                                   args.z_s_infonce_temperature,
                                                   args.z_s_infonce_delta)
                total_z_s_nce += zl.item() * B

            total_l2 += recon_loss.item()
            total_kl += kl_loss.sum().item()
            n_scenes  += B
            n_batches += 1

            if n_scenes <= B:
                _pos_abs_min  = pred_abs[:,:,0:3].cpu().float().min().item()
                _pos_abs_max  = pred_abs[:,:,0:3].cpu().float().max().item()
                _pos_gt_range = (UV_gs_batch[:,:,4:7].cpu().max()-UV_gs_batch[:,:,4:7].cpu().min()).item()/2

            # Under Chamfer the set is permutation-free, so index-matched per-component
            # error is meaningless; match by position first. Element-wise keeps slot order.
            if args.set_loss:
                _set_g = args.set_block_size if args.set_block_size > 0 else -(-args.num_gaussians // 512)
                ind = set_matched_individual_losses(
                    pred_3d, target, _set_g, args.color_loss_weight, args.geom_shape_weight,
                    args.set_pos_weight, args.set_opa_weight, args.set_sinkhorn_eps, args.set_sinkhorn_iters)
            elif args.use_chamfer_loss:
                ind = compute_individual_losses_matched(pred_3d, target, args.chamfer_chunk)
            else:
                ind = compute_individual_losses(pred_3d, target)
            for k in per_param: per_param[k] += ind[k]

            if args.cov_metric:
                try:
                    _cov = compute_covariance_diagnostics(
                        pred_3d, target, args.use_chamfer_loss, args.chamfer_chunk,
                        args.geom_eps, args.geom_ns_iters)
                    total_cov_bures += _cov['cov_bures']
                    total_cov_le    += _cov['cov_logeuclid']
                    total_aniso     += _cov['aniso']
                except Exception as _cov_err:
                    print(f"  [cov_metric skipped this eval: {type(_cov_err).__name__}: {_cov_err}]")

            if do_recon and len(recon_preds) < args.recon_ply_num_scenes:
                pnp = pred_abs.cpu().float().numpy(); mnp = mean_color_gt.cpu().numpy()
                gnp = target_abs.cpu().float().numpy() if do_gt else None   # GT 14-dim (same scenes)
                for si in range(B):
                    if len(recon_preds) >= args.recon_ply_num_scenes: break
                    recon_preds.append(pnp[si]); recon_means.append(mnp[si])
                    if do_gt: gt_preds.append(gnp[si])

            if do_pca and len(pca_input) < args.pca_num_scenes:
                for si in range(B):
                    if len(pca_input) >= args.pca_num_scenes: break
                    pca_input.append(UV_gs_batch.cpu().numpy()[si])
                    pca_recon.append(pred_abs.cpu().float().numpy()[si])
                    if do_sem_pca and pg_feats is not None:
                        pca_sem_feat.append(pg_feats.cpu().float().numpy()[si])

            if do_z_s_vis and zsp is not None:
                z_s_proj_acc.append(zsp.detach().cpu().float().numpy())
                label_dist_acc.append(label_dist_v.cpu().numpy())
            if do_zs_tok_vis and z_s_tokens_eval is not None:
                zs_tokens_acc.append(z_s_tokens_eval.cpu().float().numpy())
                if not do_z_s_vis:
                    label_dist_acc.append(label_dist_v.cpu().numpy())

            z_lay_raw_eval = raw_model.shape_model.last_z_layout
            if do_zs_lay_vis and z_lay_raw_eval is not None:
                zs_layout_acc.append(z_lay_raw_eval.detach().cpu().float().numpy())
                if not do_z_s_vis and not do_zs_tok_vis:
                    label_dist_acc.append(label_dist_v.cpu().numpy())

            if do_zs_pool_vis:
                _ph = getattr(raw_model.shape_model, 'last_zs_pool_hidden', None)
                if _ph is None:
                    _ph = getattr(raw_model.shape_model, 'last_z_layout_pool_hidden', None)
                if _ph is not None:
                    zs_pool_acc.append(_ph.detach().cpu().float().numpy())
                    if not label_dist_acc:
                        label_dist_acc.append(label_dist_v.cpu().numpy())

    # PLY / PCA saves (full-scene val by default; chunk val when vis_tag="chunk")
    if do_recon and recon_preds and accelerator.is_main_process:
        try:
            all_preds = np.stack(recon_preds, 0)
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si,:,3:6] = np.clip(all_preds[si,:,3:6] + recon_means[si], 0, 1)
            recon_dir = Path(save_path)/f"reconstructed_gaussians{_vsfx}"/f"epoch_{epoch:03d}"
            save_reconstructed_gaussians(predictions=all_preds, output_dir=recon_dir,
                epoch=epoch, num_scenes=len(all_preds),
                max_sh_degree=args.recon_ply_max_sh, color_mode="1")
            print(f"  Recon PLYs{(' ['+vis_tag+']') if vis_tag else ''}: {recon_dir}")
        except Exception as e: print(f"  PLY error: {e}")

    # Ground-truth render: saved ONCE at epoch 0, using the SAME scenes, the SAME
    # per-scene mean color, and the SAME writer/SH settings as the recon, so the two
    # folders are directly comparable scene-for-scene (scene00.ply vs scene00.ply).
    if do_gt and gt_preds and accelerator.is_main_process:
        try:
            all_gt = np.stack(gt_preds, 0)
            if args.color_residual:
                for si in range(len(all_gt)):
                    all_gt[si,:,3:6] = np.clip(all_gt[si,:,3:6] + recon_means[si], 0, 1)
            gt_dir = Path(save_path)/f"ground_truth_gaussians{_vsfx}"
            save_reconstructed_gaussians(predictions=all_gt, output_dir=gt_dir,
                epoch=epoch, num_scenes=len(all_gt),
                max_sh_degree=args.recon_ply_max_sh, color_mode="1")
            print(f"  GROUND-TRUTH PLYs{(' ['+vis_tag+']') if vis_tag else ''} "
                  f"(saved once at epoch 0): {gt_dir}")
        except Exception as e: print(f"  GT PLY error: {e}")

    if do_pca and pca_input and accelerator.is_main_process:
        try:
            pca_dir = Path(save_path)/f"pca_visualisations{_vsfx}"/f"epoch_{epoch:03d}"
            pca_dir.mkdir(parents=True, exist_ok=True)
            for si in range(len(pca_input)):
                coords_in = pca_input[si][:,4:7]
                visualize_semantic_features(coords=coords_in, features=pca_input[si],
                    output_path=str(pca_dir/f"scene{si:02d}_input.ply"),
                    brightness=args.pca_brightness, verbose=False)
                visualize_semantic_features(coords=pca_recon[si][:,0:3], features=pca_recon[si],
                    output_path=str(pca_dir/f"scene{si:02d}_recon.ply"),
                    brightness=args.pca_brightness, verbose=False)
                if si < len(pca_sem_feat):
                    visualize_semantic_features(coords=coords_in, features=pca_sem_feat[si],
                        output_path=str(pca_dir/f"scene{si:02d}_semantic_infonce.ply"),
                        brightness=args.pca_brightness, verbose=False)
            print(f"  PCA PLYs: {pca_dir}")
        except Exception as e: print(f"  PCA error: {e}")

    if do_z_s_vis and z_s_proj_acc and accelerator.is_main_process:
        try:
            all_z_s = np.concatenate(z_s_proj_acc, 0)
            all_ld  = np.concatenate(label_dist_acc, 0)
            vis_dir = Path(save_path)/f"pca_visualisations{_vsfx}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out = visualize_z_s_space(all_z_s, all_ld,
                str(vis_dir/f"z_s_space_epoch_{epoch:03d}.ply"), verbose=True)
            if out: print(f"  z_s space PLY: {out}")
        except Exception as e: print(f"  z_s vis error: {e}")

    if do_zs_tok_vis and zs_tokens_acc and accelerator.is_main_process:
        try:
            all_toks = np.concatenate(zs_tokens_acc, axis=0)
            all_ld   = np.concatenate(label_dist_acc, axis=0)
            vis_dir  = Path(save_path) / f"pca_visualisations{_vsfx}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out_tok = visualize_zs_tokens(zs_tokens=all_toks, label_dists=all_ld,
                output_path=str(vis_dir / f"zs_tokens_epoch_{epoch:03d}.ply"), verbose=True)
            if out_tok: print(f"  z_s token PLY: {out_tok}")
        except Exception as e: print(f"  z_s token vis error: {e}")

    if do_zs_lay_vis and zs_layout_acc and accelerator.is_main_process:
        try:
            all_lay = np.concatenate(zs_layout_acc, axis=0)
            all_ld  = np.concatenate(label_dist_acc, axis=0) if label_dist_acc else None
            if all_ld is not None:
                vis_dir = Path(save_path) / f"pca_visualisations{_vsfx}"
                vis_dir.mkdir(parents=True, exist_ok=True)
                out_lay = visualize_zs_tokens(zs_tokens=all_lay, label_dists=all_ld,
                    output_path=str(vis_dir / f"zs_layout_epoch_{epoch:03d}.ply"), verbose=True)
                if out_lay: print(f"  z_layout PLY: {out_lay}")
        except Exception as e: print(f"  z_layout vis error: {e}")

    if do_zs_pool_vis and zs_pool_acc and accelerator.is_main_process:
        try:
            all_pool = np.concatenate(zs_pool_acc, axis=0)
            all_ld   = np.concatenate(label_dist_acc, axis=0) if label_dist_acc else None
            if all_ld is not None:
                vis_dir = Path(save_path) / f"pca_visualisations{_vsfx}"
                vis_dir.mkdir(parents=True, exist_ok=True)
                out_pool = visualize_z_s_space(z_s_proj=all_pool, label_dists=all_ld,
                    output_path=str(vis_dir / f'zs_pool_epoch_{epoch:03d}.ply'), verbose=True)
                if out_pool: print(f'  z_s pool PLY: {out_pool}')
        except Exception as e: print(f'  z_s pool vis error: {e}')

    model.train()
    n   = max(n_scenes, 1)      # for per-scene auxiliary means (MSE-based heads)
    _nb = max(n_batches, 1)     # for raw recon / per-attribute norms (matches training)
    return {
        # avg_l2_error and per-attribute losses: RAW norm per batch, averaged over
        # batches (NOT per scene) — same convention as the training-side printout.
        'avg_l2_error':       total_l2 / _nb,
        'cov_bures':          total_cov_bures / _nb,
        'cov_logeuclid':      total_cov_le / _nb,
        'aniso':              total_aniso / _nb,
        'avg_kl':             total_kl / n,
        'color_pred_loss':    total_color / n,
        'scene_semantic_kl':  total_scene_sem / n,
        'anchor_loss':        total_anchor / n,
        'layout_loss':        total_layout / n,
        'seg_pred_loss':      total_seg / n,
        'z_s_infonce_loss':   total_z_s_nce / n,
        'zs_tok_infonce_loss':  total_zs_tok_nce / n,
        'zs_lay_infonce_loss':  total_zs_lay_nce / n,
        'zs_pool_infonce_loss': 0.0,
        'pos_abs_range':      _pos_abs_max - _pos_abs_min,
        'pos_abs_min':        _pos_abs_min,
        'pos_abs_max':        _pos_abs_max,
        'pos_gt_range':       _pos_gt_range,
        **{f'{k}_loss': v/_nb for k, v in per_param.items()},
    }

# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n{'='*70}\nSTARTING TRAINING  (epoch {start_epoch} -> {args.num_epochs-1})\n{'='*70}\n")

_kl_anneal_active = (args.kl_anneal_steps > 0)
_kl_step_offset   = _bpe * start_epoch

if accelerator.is_main_process:
    print(f"  KL annealing : {'ENABLED' if _kl_anneal_active else 'DISABLED (fixed kl_weight)'}")
    if _kl_anneal_active:
        _ramp_epochs = args.kl_anneal_steps / max(_bpe, 1)
        print(f"  kl_anneal_steps={args.kl_anneal_steps}  "
              f"(~ {_ramp_epochs:.0f} epochs at {_bpe} steps/epoch)")
        print(f"  kl_weight ramps: 0.0 -> {args.kl_weight:.1e} over first {args.kl_anneal_steps} steps")
    else:
        print(f"  kl_weight fixed at {args.kl_weight:.1e} throughout")
    print()

global_step = _kl_step_offset

# DDP visibility hook: connect all side-head outputs to the model output graph
if raw_model.shape_model is not None:
    _SIDE_HEAD_ATTRS = [
        'last_mean_color_pred',
        'last_scene_semantic_pred',
        'last_scene_layout_pred',
        'last_z_s_infonce_proj',
        'last_zs_pool_proj',
        'last_zs_pool_hidden',
        'last_z_layout_proj',
        'last_z_layout_pool_proj',
        'last_z_layout_pool_hidden',
        'last_seg_pred',
    ]

    def _all_side_heads_ddp_visibility_hook(module, inp, output):
        uv = output[4]
        if uv is None:
            return output
        zero_sum = None
        for attr in _SIDE_HEAD_ATTRS:
            pred = getattr(module, attr, None)
            if pred is not None and isinstance(pred, torch.Tensor) and pred.requires_grad:
                term = pred.sum() * 0.0
                zero_sum = term if zero_sum is None else zero_sum + term
        if zero_sum is None:
            return output
        uv_modified = uv + zero_sum
        pf = output[5]
        pf_modified = (pf + zero_sum) if (pf is not None) else pf
        return (output[0], output[1], output[2], output[3], uv_modified, pf_modified)

    raw_model.shape_model.register_forward_hook(_all_side_heads_ddp_visibility_hook)
    if accelerator.is_main_process:
        print("  DDP visibility hook registered: ALL side-head outputs connected to "
              "model output graph via zero-gradient paths (fixes 'marked ready twice')")

for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Training",
                  disable=not accelerator.is_main_process):
    gs_autoencoder.train()

    e = {k: 0.0 for k in [
        'loss','recon','kl','sem','color_pred','scene_sem','anchor',
        'layout','cross_recon','ortho','seg_pred','scale_pen',
        'z_s_nce','z_s_npos',
        'zs_tok_nce','zs_tok_ncats',
        'zs_lay_nce','zs_lay_ncats',
        'zs_pool_nce','zs_pool_ncats',
        'pos','col','opa','scl','rot','shape','render','opos']}
    _ind_cache = {k: 0.0 for k in ('position','color','opacity','scale','rotation','shape')}

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)
        label_dist_v  = batch_data['label_dist'].float().to(device)
        B = UV_gs_batch.shape[0]

        _sem_valid = label_dist_v.sum(dim=1) > 1e-6   # [B] bool

        seg_labels = inst_labels = None
        if need_segment_labels:
            seg_labels  = batch_data['segment_labels'].long().to(device)
            if enable_semantic:
                inst_labels = batch_data['instance_labels'].long().to(device)

        sa_gpu  = (batch_data['scaffold_anchors'].float().to(device) if need_scaffold_data else None)
        sti_gpu = (batch_data['scaffold_token_ids'].long().to(device) if args.position_scaffold else None)

        optimizer.zero_grad()

        if _kl_anneal_active and global_step < args.kl_anneal_steps:
            _kl_current = args.kl_weight * (global_step / args.kl_anneal_steps)
        else:
            _kl_current = args.kl_weight

        with torch.autocast('cuda', dtype=_autocast_dtype, enabled=_use_autocast):
            (shape_embed, mu, log_var, z,
             UV_gs_recover, pg_features) = gs_autoencoder(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:,:,:3],
                scaffold_anchors=sa_gpu, scaffold_token_ids=sti_gpu)

        mcp   = raw_model.shape_model.last_mean_color_pred
        ssp   = raw_model.shape_model.last_scene_semantic_pred
        anch  = raw_model.shape_model.last_predicted_anchors_from_tokens
        slp   = raw_model.shape_model.last_scene_layout_pred
        sgp   = raw_model.shape_model.last_seg_pred
        zsp   = raw_model.shape_model.last_z_s_infonce_proj
        _mu_s = raw_model.shape_model._mu_s_cache
        _mu_g = raw_model.shape_model._mu_g_cache

        target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        if args.anchor_relative_decode:
            # decode() already produced ABSOLUTE positions (anchor + bounded
            # offset); train directly on absolute targets (no residual subtraction).
            target  = target_abs
            pred_3d = UV_gs_recover.reshape(B,-1,14)
        elif args.position_scaffold:
            pos_off = batch_data['position_offsets'].float().to(device)
            target  = target_abs.clone(); target[:,:,0:3] = pos_off
            pred_3d = UV_gs_recover.reshape(B,-1,14).clone()
            if anch is not None:
                idx_3d = sti_gpu.unsqueeze(-1).expand(-1,-1,3)
                pred_3d[:,:,0:3] -= torch.gather(anch, 1, idx_3d)
        elif args.position_layout_residual:
            pos_res = batch_data['position_residuals'].float().to(device)
            target  = target_abs.clone(); target[:,:,0:3] = pos_res
            pred_3d = UV_gs_recover.reshape(B,-1,14)
        else:
            target  = target_abs
            pred_3d = UV_gs_recover.reshape(B,-1,14)

        opacity_pos_loss = torch.tensor(0., device=device)
        if args.set_loss:
            _set_g = args.set_block_size if args.set_block_size > 0 else -(-args.num_gaussians // 512)
            if args.opacity_pos_weight > 0:
                recon_loss, opacity_pos_loss = compute_reconstruction_loss_set(
                    pred_3d, target, B, args.color_loss_weight, args.geom_shape_weight, _set_g,
                    args.set_pos_weight, args.set_opa_weight,
                    args.set_sinkhorn_eps, args.set_sinkhorn_iters, return_opos=True)
            else:
                recon_loss = compute_reconstruction_loss_set(
                    pred_3d, target, B, args.color_loss_weight, args.geom_shape_weight, _set_g,
                    args.set_pos_weight, args.set_opa_weight,
                    args.set_sinkhorn_eps, args.set_sinkhorn_iters)
        elif args.geom_loss == 'l2':
            if args.use_chamfer_loss:
                recon_loss = chamfer_reconstruction_loss(
                    pred_3d, target, B, args.color_loss_weight, args.chamfer_chunk)
            else:
                recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
        else:
            if args.use_chamfer_loss:
                recon_loss = chamfer_reconstruction_loss_geom(
                    pred_3d, target, B, args.color_loss_weight, args.geom_loss,
                    args.chamfer_chunk, args.geom_eps, args.geom_ns_iters, args.geom_shape_weight)
            else:
                recon_loss = compute_reconstruction_loss_geom(
                    pred_3d, target, B, args.color_loss_weight, args.geom_loss,
                    args.geom_eps, args.geom_ns_iters, args.geom_shape_weight)

        # Virtual-camera rendering loss (image-space crispness signal). Rendered in
        # fp32 outside autocast; the gradient flows back through the cast into the
        # (possibly bf16) model. Off unless --render_loss; gated by a warmup so it
        # can be switched on as a fine-tune phase after the set loss has converged.
        render_loss = torch.tensor(0., device=device)
        if args.render_loss and epoch >= args.render_warmup_epochs:
            if compute_render_loss is None:
                raise RuntimeError(
                    f"--render_loss set but render_loss could not be imported: "
                    f"{_RENDER_LOSS_IMPORT_ERROR}. Ensure render_loss.py is on the path "
                    f"and gsplat is installed.")
            with torch.amp.autocast('cuda', enabled=False):
                render_loss = compute_render_loss(
                    pred_3d.float(), target.float(), mean_color_gt.float(),
                    num_ring=args.render_views, img_size=args.render_img,
                    fov_deg=args.render_fov, ssim_weight=args.render_ssim_weight,
                    lpips_weight=args.render_lpips_weight,
                    max_scenes=args.render_max_scenes, up_axis=args.render_up_axis,
                    dist_mult=args.render_dist_mult, quat_order=args.render_quat_order)

        log_var_clamped = log_var.clamp(-10.0, 10.0)
        KL_loss     = -0.5*torch.sum(1+log_var_clamped-mu.pow(2)-log_var_clamped.exp(), dim=1).mean()

        color_pred_loss = torch.tensor(0., device=device)
        if mcp is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mcp.float(), mean_color_gt)

        scene_sem_loss = torch.tensor(0., device=device)
        if ssp is not None and args.scene_semantic_head:
            p_s = batch_data['label_dist'].float().to(device)
            scene_sem_loss = scene_semantic_kl_loss(ssp.float(), p_s)

        anchor_loss = torch.tensor(0., device=device)
        if anch is not None and args.position_scaffold and sa_gpu is not None:
            anchor_loss = F.mse_loss(anch.float(), sa_gpu)

        layout_loss = torch.tensor(0., device=device)
        if slp is not None and args.scene_layout_head:
            gt_c = batch_data['category_centroids'].float().to(device)
            gt_v = batch_data['category_valid'].float().to(device)
            layout_loss = compute_layout_loss(slp.float(), gt_c, gt_v)

        seg_pred_loss = torch.tensor(0., device=device)
        if args.predict_seg_labels and sgp is not None and seg_labels is not None:
            seg_pred_loss = compute_seg_pred_loss(sgp, seg_labels)

        semantic_loss    = torch.tensor(0., device=device)
        semantic_metrics = {}
        if enable_semantic and seg_labels is not None and pg_features is not None:
            if args.semantic_mode == 'dist':
                semantic_loss, semantic_metrics = compute_distribution_loss(
                    dist_logits=pg_features, segment_labels=seg_labels,
                    weight=args.segment_loss_weight)
            else:
                semantic_loss, semantic_metrics = compute_semantic_loss(
                    embeddings=pg_features, segment_labels=seg_labels,
                    instance_labels=inst_labels, batch_size=B,
                    segment_weight=args.segment_loss_weight,
                    instance_weight=args.instance_loss_weight,
                    temperature=args.semantic_temperature,
                    subsample=args.semantic_subsample,
                    sampling_strategy=args.sampling_strategy)

        z_s_nce_loss    = torch.tensor(0., device=device)
        z_s_nce_metrics = {'z_s_infonce_loss': 0., 'z_s_num_positives': 0., 'z_s_frac_anchors': 0.}
        if args.z_s_infonce_weight > 0 and zsp is not None:
            z_s_nce_loss, z_s_nce_metrics = compute_scene_infonce_loss(
                zsp, label_dist_v, args.z_s_infonce_temperature, args.z_s_infonce_delta)

        zs_tok_nce_loss    = torch.tensor(0., device=device)
        zs_tok_nce_metrics = {'zs_tok_infonce_loss': 0., 'zs_tok_num_categories': 0}
        if args.zs_token_infonce_weight > 0 and args.latent_disentangle:
            _ed_tok      = raw_model.shape_model.embed_dim
            _n_tok       = args.semantic_dims // _ed_tok
            _z_s_tok_all = z[:, :args.semantic_dims].reshape(B, _n_tok, _ed_tok)
            _n_sem_tok   = int(_sem_valid.sum().item())
            if _n_sem_tok >= 2:
                zs_tok_nce_loss, zs_tok_nce_metrics = compute_zs_token_infonce_loss(
                    _z_s_tok_all[_sem_valid], label_dist_v[_sem_valid],
                    args.zs_token_infonce_temperature)

        zs_lay_nce_loss    = torch.tensor(0., device=device)
        zs_lay_nce_metrics = {'zs_layout_infonce_loss': 0., 'zs_layout_num_cats': 0}
        z_lay_proj = raw_model.shape_model.last_z_layout_proj
        if args.zs_layout_infonce_weight > 0 and z_lay_proj is not None:
            zs_lay_nce_loss, zs_lay_nce_metrics = compute_zs_layout_infonce_loss(
                z_lay_proj, label_dist_v, args.zs_layout_infonce_temperature)

        zs_pool_nce_loss    = torch.tensor(0., device=device)
        zs_pool_nce_metrics = {'zs_pool_infonce_loss': 0., 'zs_pool_num_cats': 0}
        if args.zs_pool_infonce_weight > 0:
            _pool_emb = raw_model.shape_model.last_zs_pool_proj
            if _pool_emb is None:
                _pool_emb = getattr(raw_model.shape_model, 'last_z_layout_pool_proj', None)
            _n_sem_pool = int(_sem_valid.sum().item())
            if _pool_emb is not None and _n_sem_pool >= 2:
                _pe_v        = _pool_emb[_sem_valid]
                _ld_v        = label_dist_v[_sem_valid]
                _dom_cat     = _ld_v.float().argmax(dim=1)
                _pool_labels = _dom_cat.unsqueeze(1).expand(-1, _pe_v.shape[1]).long()
                zs_pool_nce_loss, _pool_metrics = compute_semantic_loss(
                    embeddings=_pe_v, segment_labels=_pool_labels,
                    instance_labels=None, batch_size=_n_sem_pool,
                    segment_weight=1.0, instance_weight=0.0,
                    temperature=args.zs_pool_infonce_temperature,
                    subsample=_pe_v.shape[1],
                    sampling_strategy=args.sampling_strategy)
                zs_pool_nce_metrics = {
                    'zs_pool_infonce_loss': _pool_metrics.get('segment_loss', 0.),
                    'zs_pool_num_cats':     _pool_metrics.get('num_categories_in_batch', 0)}

        cross_recon_loss = torch.tensor(0., device=device)
        if (args.latent_disentangle and args.cross_recon_weight > 0
                and _mu_s is not None and _mu_g is not None and B > 1):
            D_s = args.semantic_dims
            mu_s_shifted = torch.roll(_mu_s, shifts=1, dims=0)
            lv_s_shifted = torch.roll(log_var[:, :D_s], shifts=1, dims=0)
            z_s_swapped  = mu_s_shifted + torch.exp(0.5*lv_s_shifted) * torch.randn_like(mu_s_shifted)
            z_g_current  = _mu_g + torch.exp(0.5*log_var[:, D_s:]) * torch.randn_like(_mu_g)
            z_cross      = torch.cat([z_s_swapped, z_g_current], dim=-1)
            lat_cross    = z_cross.reshape(B, 512, -1)

            if (raw_model.shape_model.scene_layout_module is not None and
                    args.semantic_token_heads):
                with torch.no_grad():
                    _ed = raw_model.shape_model.embed_dim
                    _sd = args.semantic_dims
                    if args.structured_layout_tokens:
                        _n_s   = raw_model.shape_model._n_sem_tokens
                        _start = _ed + _n_s * _ed
                        z_lay_B = z_s_swapped[:, _start:_sd]
                        raw_model.shape_model.last_scene_layout_pred = \
                            raw_model.shape_model.scene_layout_module(z_lay_B)
                    else:
                        z_sem_B = z_s_swapped[:, _ed:_sd]
                        raw_model.shape_model.last_scene_layout_pred = \
                            raw_model.shape_model.scene_layout_module(z_sem_B)

            se_shifted = torch.roll(raw_model.shape_model._shape_embed_cache, shifts=1, dims=0)
            _z_layout_shifted = None
            _any_B_train = args.decoder_layout_cross_attn or args.decoder_layout_additive
            if _any_B_train and raw_model.shape_model.last_z_layout is not None:
                _z_layout_shifted = torch.roll(
                    raw_model.shape_model.last_z_layout, shifts=1, dims=0)

            _saved_slm = raw_model.shape_model.scene_layout_module
            _saved_slp = raw_model.shape_model.last_scene_layout_pred
            raw_model.shape_model.scene_layout_module = None
            if _saved_slp is not None:
                raw_model.shape_model.last_scene_layout_pred = _saved_slp.detach()

            with torch.autocast('cuda', dtype=_autocast_dtype, enabled=_use_autocast):
                UV_cross, _ = raw_model.shape_model.decode(
                    lat_cross, volume_queries=None,
                    return_semantic_features=False, shape_embed=se_shifted,
                    scaffold_anchors=sa_gpu, scaffold_token_ids=sti_gpu,
                    z_layout=_z_layout_shifted)

            raw_model.shape_model.scene_layout_module = _saved_slm
            raw_model.shape_model.last_scene_layout_pred = _saved_slp
            pred_cross_3d = UV_cross.reshape(B, -1, 14)

            if args.position_scaffold:
                cross_anch = raw_model.shape_model.last_predicted_anchors_from_tokens
                if cross_anch is not None and sti_gpu is not None:
                    idx_cr = sti_gpu.unsqueeze(-1).expand(-1,-1,3)
                    cross_dc = torch.gather(cross_anch, 1, idx_cr).detach()
                    pred_cross_for_loss = pred_cross_3d.clone()
                    pred_cross_for_loss[:,:,0:3] -= cross_dc
                else:
                    pred_cross_for_loss = pred_cross_3d
            else:
                pred_cross_for_loss = pred_cross_3d

            cross_recon_loss = compute_cross_recon_loss(pred_cross_for_loss, target, B)

            if (raw_model.shape_model.scene_layout_module is not None and
                    args.semantic_token_heads):
                raw_model.shape_model.last_scene_layout_pred = slp

        ortho_loss = torch.tensor(0., device=device)
        if (args.latent_disentangle and args.ortho_weight > 0
                and _mu_s is not None and _mu_g is not None and B > 1):
            ortho_loss = compute_orthogonality_loss(_mu_s, _mu_g)

        scale_pen = torch.tensor(0., device=device)
        if args.scale_penalty_weight > 0:
            scale_pen = compute_scale_penalty(UV_gs_recover.reshape(B,-1,14),
                                              threshold=args.scale_penalty_threshold)

        total_loss = (args.param_loss_weight  * recon_loss
                      + args.render_loss_weight    * render_loss
                      + args.opacity_pos_weight    * opacity_pos_loss
                      + _kl_current                * KL_loss
                      + args.mean_color_weight       * color_pred_loss
                      + args.scene_semantic_weight   * scene_sem_loss
                      + args.anchor_loss_weight      * anchor_loss
                      + args.layout_loss_weight      * layout_loss
                      + args.cross_recon_weight      * cross_recon_loss
                      + args.ortho_weight            * ortho_loss
                      + args.seg_pred_weight         * seg_pred_loss
                      + args.scale_penalty_weight    * scale_pen
                      + args.z_s_infonce_weight      * z_s_nce_loss
                      + args.zs_token_infonce_weight * zs_tok_nce_loss
                      + args.zs_layout_infonce_weight * zs_lay_nce_loss
                      + args.zs_pool_infonce_weight   * zs_pool_nce_loss
                      + semantic_loss)

        accelerator.backward(total_loss)
        accelerator.clip_grad_norm_(gs_autoencoder.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        # Per-component readout: matched-by-position under Chamfer (the set is
        # permutation-free), index-matched for element-wise (slot order is enforced).
        if args.set_loss:
            # Set matching needs a Sinkhorn; compute the readout on a stride and carry it
            # forward (the loss still matches every step). Removes a redundant 2nd match/step.
            if i_batch == 0 or (global_step % max(args.set_diag_every, 1) == 0):
                _set_g = args.set_block_size if args.set_block_size > 0 else -(-args.num_gaussians // 512)
                _ind_cache = set_matched_individual_losses(
                    pred_3d, target, _set_g, args.color_loss_weight, args.geom_shape_weight,
                    args.set_pos_weight, args.set_opa_weight, args.set_sinkhorn_eps, args.set_sinkhorn_iters)
            ind = _ind_cache
        elif args.use_chamfer_loss:
            ind = compute_individual_losses_matched(pred_3d, target, args.chamfer_chunk)
        else:
            ind = compute_individual_losses(pred_3d, target)
        e['loss']       += total_loss.item()
        e['recon']      += recon_loss.item()
        e['render']     += float(render_loss)
        e['opos']       += float(opacity_pos_loss)
        e['kl']         += KL_loss.item()
        e['sem']        += semantic_loss.item()
        e['color_pred'] += color_pred_loss.item()
        e['scene_sem']  += scene_sem_loss.item()
        e['anchor']     += anchor_loss.item()
        e['layout']     += layout_loss.item()
        e['cross_recon'] += cross_recon_loss.item()
        e['ortho']      += ortho_loss.item()
        e['seg_pred']   += seg_pred_loss.item()
        e['scale_pen']  += scale_pen.item()
        e['z_s_nce']    += z_s_nce_loss.item()
        e['z_s_npos']   += z_s_nce_metrics.get('z_s_num_positives', 0.)
        e['zs_tok_nce']   += zs_tok_nce_loss.item()
        e['zs_tok_ncats'] += zs_tok_nce_metrics.get('zs_tok_num_categories', 0)
        e['zs_lay_nce']    += zs_lay_nce_loss.item()
        e['zs_lay_ncats']  += zs_lay_nce_metrics.get('zs_layout_num_cats', 0)
        e['zs_pool_nce']   += zs_pool_nce_loss.item()
        e['zs_pool_ncats'] += zs_pool_nce_metrics.get('zs_pool_num_cats', 0)
        e['pos'] += ind['position']; e['col'] += ind['color']
        e['opa'] += ind['opacity'];  e['scl'] += ind['scale']
        e['rot'] += ind['rotation']
        # Covariance shape term backed out of Recon so it is visible in the log.
        # Recon*B = pos + color_weight*col + opa + shape, and Pos/Col/Opa here are the
        # same norms the loss used (exact for element-wise; approximate under Chamfer,
        # whose matched pairs differ from the index-matched diagnostics).
        if args.set_loss:
            e['shape'] += ind['shape']          # set-matched covariance (meaningful)
        elif args.geom_loss != 'l2':
            e['shape'] += (recon_loss.item() * B
                           - ind['position'] - args.color_loss_weight * ind['color']
                           - ind['opacity'])

        if epoch == start_epoch and i_batch == 0 and accelerator.is_main_process:
            print(f"\nEPOCH {epoch} BATCH 0 DIAGNOSTIC:")
            print(f"  recon={recon_loss.item():.4f} | KL={KL_loss.item():.4f} | "
                  f"kl_weight={_kl_current:.2e} | KL_contrib={_kl_current*KL_loss.item():.4f} | "
                  f"mu=[{mu.min().item():.3f},{mu.max().item():.3f}]")
            print(f"  recon objective: {'CHAMFER' if args.use_chamfer_loss else 'element-wise'} | "
                  f"morton_order={args.morton_order}")
            if _kl_anneal_active:
                _pct = min(100.0, 100.0 * global_step / args.kl_anneal_steps)
                print(f"  KL annealing: step {global_step}/{args.kl_anneal_steps} "
                      f"({_pct:.1f}% of ramp complete)")
            if _mu_s is not None:
                print(f"  mu_s=[{_mu_s.min().item():.3f},{_mu_s.max().item():.3f}]  "
                      f"mu_g=[{_mu_g.min().item():.3f},{_mu_g.max().item():.3f}]")

        global_step += 1

    nb = len(trainDataLoader)
    lr_now = scheduler.get_last_lr()[0]
    _kl_log = _kl_current
    if accelerator.is_main_process:
        print(f"\nEpoch {epoch:04d} | "
              f"Loss={e['loss']/nb:.4f} | "
              f"Recon={e['recon']/nb:.4f} | "
              f"KL={e['kl']/nb:.4f} | "
              f"KLw={_kl_log:.2e} | "
              f"ColorPred={e['color_pred']/nb:.6f} | "
              f"SceneSem={e['scene_sem']/nb:.4f} | "
              f"Layout={e['layout']/nb:.4f} | "
              f"CrossRecon={e['cross_recon']/nb:.4f} | "
              f"Ortho={e['ortho']/nb:.6f} | "
              f"Anchor={e['anchor']/nb:.4f} | "
              f"SegPred={e['seg_pred']/nb:.4f} | "
              f"ScalePen={e['scale_pen']/nb:.6f} | "
              f"Z_sNCE={e['z_s_nce']/nb:.4f} | "
              f"Z_sNPos={e['z_s_npos']/nb:.1f} | "
              f"ZsTokNCE={e['zs_tok_nce']/nb:.4f} | "
              f"ZsTokNCats={e['zs_tok_ncats']/nb:.1f} | "
              f"LayNCE={e['zs_lay_nce']/nb:.4f} | "
              f"LayNCats={e['zs_lay_ncats']/nb:.1f} | "
              f"PoolNCE={e['zs_pool_nce']/nb:.4f} | "
              f"PoolNCats={e['zs_pool_ncats']/nb:.1f} | "
              f"PgNCE={e['sem']/nb:.4f} | "
              f"LR={lr_now:.2e}")
        print(f"  Pos={e['pos']/nb:.3f} | Col={e['col']/nb:.3f} | "
              f"Opa={e['opa']/nb:.3f} | Scl={e['scl']/nb:.3f} | Rot={e['rot']/nb:.3f}"
              + (f" | Shape={e['shape']/nb:.3f}" if args.geom_loss != 'l2' else "")
              + (f" | Render={e['render']/nb:.4f}" if args.render_loss else "")
              + (f" | OPos={e['opos']/nb:.2f}" if args.opacity_pos_weight > 0 else ""))

    # EVALUATION
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:

        val_metrics = evaluate_model(gs_autoencoder, raw_model, valDataLoader,
                                     device, accelerator, epoch=epoch, do_vis=True)

        if accelerator.is_main_process:
            print(f"\n--- Val FULL SCENES epoch {epoch} ---")
            print(f"  L2={val_metrics['avg_l2_error']:.4f}  "
                  f"Pos={val_metrics['position_loss']:.4f}  "
                  f"Col={val_metrics['color_loss']:.4f}  "
                  f"Opa={val_metrics['opacity_loss']:.4f}  "
                  f"Scl={val_metrics['scale_loss']:.4f}  "
                  f"Rot={val_metrics['rotation_loss']:.4f}")
            if args.cov_metric:
                print(f"  CovBures={val_metrics['cov_bures']:.5f}  "
                      f"CovLogE={val_metrics['cov_logeuclid']:.5f}  "
                      f"Aniso={val_metrics['aniso']:.2f}  (gauge-invariant; raw Rot is gauge-blind)")
            if args.color_residual:
                print(f"  ColorPredMSE={val_metrics['color_pred_loss']:.6f}")
            if args.scene_semantic_head:
                print(f"  SceneSemKL={val_metrics['scene_semantic_kl']:.4f}")
            if args.scene_layout_head:
                print(f"  LayoutMSE={val_metrics['layout_loss']:.4f}")
            if args.z_s_infonce_weight > 0:
                print(f"  Z_sNCE={val_metrics['z_s_infonce_loss']:.4f}")
            if args.zs_token_infonce_weight > 0:
                print(f"  ZsTokNCE={val_metrics['zs_tok_infonce_loss']:.4f}")
            if args.zs_layout_infonce_weight > 0:
                print(f"  LayNCE={val_metrics['zs_lay_infonce_loss']:.4f}")

        chunk_metrics = None
        if _has_chunk_val:
            # do_vis=True + vis_tag="chunk": save held-out-chunk reconstructions to
            # reconstructed_gaussians_chunk/ (separate from the full-scene PLYs).
            chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                           device, accelerator, epoch=epoch,
                                           do_vis=True, vis_tag="chunk")
            if accelerator.is_main_process:
                print(f"\n--- Val HELD-OUT CHUNKS epoch {epoch} "
                      f"(skip={_n_train_chunks}, n={len(gs_dataset_val_chunk)}) ---")
                print(f"  L2={chunk_metrics['avg_l2_error']:.4f}  "
                      f"Pos={chunk_metrics['position_loss']:.4f}  "
                      f"Col={chunk_metrics['color_loss']:.4f}  "
                      f"Opa={chunk_metrics['opacity_loss']:.4f}  "
                      f"Scl={chunk_metrics['scale_loss']:.4f}  "
                      f"Rot={chunk_metrics['rotation_loss']:.4f}")
                if args.cov_metric:
                    print(f"  CovBures={chunk_metrics['cov_bures']:.5f}  "
                          f"CovLogE={chunk_metrics['cov_logeuclid']:.5f}  "
                          f"Aniso={chunk_metrics['aniso']:.2f}  (gauge-invariant)")
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    _gap = val_metrics['avg_l2_error'] / chunk_metrics['avg_l2_error']
                    print(f"  DISTRIBUTION GAP  full_L2 / chunk_L2 = {_gap:.2f}x  "
                          f"({'negligible' if _gap < 1.3 else 'moderate' if _gap < 2.0 else 'large — chunks much easier'})")

        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch    = epoch
            if accelerator.is_main_process:
                ckpt_dict = {
                    'epoch':            epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'val_l2_error':     val_metrics['avg_l2_error'],
                    **_ckpt_meta,
                }
                if chunk_metrics is not None:
                    ckpt_dict['chunk_val_l2_error'] = chunk_metrics['avg_l2_error']
                torch.save(ckpt_dict, os.path.join(save_path, "best_model.pth"))
                print(f"  [NEW BEST] full_L2={best_val_loss:.4f} saved")

        if accelerator.is_main_process and wandb_enabled:
            log_dict = {
                'epoch': epoch,
                'val_full_l2': val_metrics['avg_l2_error'],
                'val_full_pos': val_metrics['position_loss'],
                'val_full_col': val_metrics['color_loss'],
            }
            if args.cov_metric:
                log_dict['val_full_cov_bures']     = val_metrics['cov_bures']
                log_dict['val_full_cov_logeuclid'] = val_metrics['cov_logeuclid']
                log_dict['val_full_aniso']         = val_metrics['aniso']
            if chunk_metrics is not None:
                log_dict['val_chunk_l2']  = chunk_metrics['avg_l2_error']
                log_dict['val_chunk_pos'] = chunk_metrics['position_loss']
                if args.cov_metric:
                    log_dict['val_chunk_cov_bures']     = chunk_metrics['cov_bures']
                    log_dict['val_chunk_cov_logeuclid'] = chunk_metrics['cov_logeuclid']
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    log_dict['val_dist_gap'] = (val_metrics['avg_l2_error']
                                                / chunk_metrics['avg_l2_error'])
            wandb_run.log(log_dict)

# ============================================================================
# FINAL SAVE
# ============================================================================
accelerator.wait_for_everyone()
final_metrics = evaluate_model(gs_autoencoder, raw_model, valDataLoader, device,
                               accelerator, epoch=args.num_epochs-1, do_vis=True)
final_chunk_metrics = None
if _has_chunk_val:
    final_chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                         device, accelerator, epoch=args.num_epochs-1,
                                         do_vis=True, vis_tag="chunk")

if accelerator.is_main_process:
    print(f"\nFinal full_L2 : {final_metrics['avg_l2_error']:.4f}")
    if final_chunk_metrics is not None:
        print(f"Final chunk_L2: {final_chunk_metrics['avg_l2_error']:.4f}")
        if final_chunk_metrics['avg_l2_error'] > 1e-6:
            print(f"Final gap     : {final_metrics['avg_l2_error']/final_chunk_metrics['avg_l2_error']:.2f}x")
    print(f"Best full_L2  : {best_val_loss:.4f}  (epoch {best_epoch})")

    final_dict = {
        'epoch':            args.num_epochs - 1,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_l2':     final_metrics['avg_l2_error'],
        'best_val_l2':      best_val_loss,
        'best_epoch':       best_epoch,
        **_ckpt_meta,
        'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
    }
    if final_chunk_metrics is not None:
        final_dict['final_chunk_val_l2'] = final_chunk_metrics['avg_l2_error']
    torch.save(final_dict, os.path.join(save_path, "final.pth"))
    print(f"Saved: {save_path}final.pth")

if wandb_enabled and accelerator.is_main_process:
    summary = {"final_val_l2": final_metrics['avg_l2_error'],
               "best_val_l2": best_val_loss, "best_epoch": best_epoch}
    if final_chunk_metrics is not None:
        summary["final_chunk_val_l2"] = final_chunk_metrics['avg_l2_error']
    wandb_run.summary.update(summary)
    wandb_run.finish()

if accelerator.is_main_process: print("Done.")