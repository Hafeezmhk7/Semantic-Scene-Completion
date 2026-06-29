#!/usr/bin/env python3
"""
apply_sigma_loss_patch.py
=========================

Adds a gauge-invariant Σ-loss option to gs_can3tok_2.py for testing whether
the small-isotropic-Gaussian failure mode in element-wise L2 on 3DGS
parameters is caused by gauge ambiguity in the (s, q) parameterisation.

THE DIAGNOSIS THIS TEST FALSIFIES
---------------------------------
The 3DGS factorisation Σ = R(q)·diag(s²)·R(q)ᵀ is many-to-one in (s, q):
axis-permutation pairs (σ·s, q·r_σ), the ±q double cover, and S¹/S³ fibres
for degenerate scales all give the same Σ. Training data records ONE (s, q)
representative per scene per slot, chosen essentially at random by the
upstream 3DGS optimiser. An element-wise L2 loss on (s, q) is then asked to
regress on a gauge-randomised target; the Bayes-optimal predictor is the
orbit centroid, which for typical anisotropic Σ is isotropic in s and
degenerate in q — i.e. small, spherical Gaussians, the artefact you see.

If the diagnosis is right, replacing L2-on-(s,q) with L2-on-Σ (which only
sees the gauge-INVARIANT quantity) should produce anisotropic Gaussians
within a few hundred epochs without any architectural change. If the
diagnosis is wrong, nothing changes and the failure is elsewhere.

WHAT THIS PATCH DOES
--------------------
1. Adds `_build_sigma_from_sq(s, q)` helper that constructs the covariance
   Σ = R(q)·diag(s²)·R(q)ᵀ in fp32 (even under bf16 autocast).
2. Extends `compute_reconstruction_loss` with `use_sigma_loss` and
   `sigma_weight` arguments. When the flag is on, the element-wise L2 terms
   on scale (7:10) and rotation (10:14) are REPLACED with a single Frobenius
   norm on Σ. Position (0:3), color (3:6), opacity (6:7) stay as element-wise
   L2 since they carry no gauge.
3. Adds CLI flags `--sigma_loss` (bool) and `--sigma_weight` (float, default
   1.0).
4. Threads the new flags through the training-loop call site and the
   `evaluate_model` call site.
5. Records the flags in checkpoint metadata so resumed runs are reproducible.
6. Adds a startup-summary notification line when the flag is active.

HOW TO READ THE RESULTS
-----------------------
THE PLY RENDER IS THE REAL SIGNAL. Under --sigma_loss the printed `Scl=...`
and `Rot=...` numbers in the training log are computed via
`_masked_individual_losses` on the raw decoder outputs; since Σ-loss is
gauge-invariant, the model is free to pick ANY orbit representative, so
those per-attribute L2 magnitudes can grow or oscillate even as the rendered
Gaussians become correctly anisotropic. Open the recon PLYs at epochs
500 / 1000 / etc. and look at the SHAPES of the Gaussians.

USAGE
-----
    python apply_sigma_loss_patch.py             # default training-script path
    python apply_sigma_loss_patch.py PATH        # custom path

Idempotent: re-running on a patched file is a no-op (a sentinel comment is
appended). A backup of the original is written to PATH + '.bak.sigma' before
the patched content is written.
"""
from __future__ import annotations
import os, sys, shutil

SENTINEL = "# === SIGMA_LOSS_PATCH_APPLIED ==="
DEFAULT_PATH = "/home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion/gs_can3tok_2.py"

# ---------------------------------------------------------------------------
# Patch blocks. Each block is applied in order via str.replace; the order
# matters because some blocks insert content that later blocks anchor on.
# Each `old` must appear EXACTLY ONCE in the file (verified before any
# changes are written), so the patch fails loudly on any mismatch.
# ---------------------------------------------------------------------------

BLOCK_1_HELPER_OLD = """# ============================================================================
# LOSS HELPERS
# ============================================================================
def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0,
"""

BLOCK_1_HELPER_NEW = """# ============================================================================
# LOSS HELPERS
# ============================================================================
def _build_sigma_from_sq(s, q, eps=1e-8):
    \"\"\"Build the 3D Gaussian covariance Σ = R(q)·diag(s²)·R(q)ᵀ from (scale, quat).

    The 3DGS factorisation (s, q) -> Σ is many-to-one (gauge-redundant): axis
    permutations (σ·s, q·r_σ), ±q sign flips, and S¹/S³ fibres for degenerate
    scales all give the same Σ. Element-wise L2 on (s, q) averages over that
    orbit and converges to its centroid -- which for typical anisotropic Σ is
    isotropic in scale and degenerate in rotation, producing the small-circular-
    Gaussian artefact. A loss in Σ-space sees only the gauge-INVARIANT quantity,
    so any orbit representative is admissible and the artefact goes away.

    s : [..., 3]  scale (positive; raw decoder output, assumed softplus'd)
    q : [..., 4]  quaternion in 3DGS (w, x, y, z) convention, L2-normalised here
    eps : numerical floor for the quaternion norm

    Returns : [..., 3, 3]  symmetric positive-(semi)definite covariance, fp32.
    \"\"\"
    # fp32 inside even under bf16 autocast: the per-Gaussian matmul is small but
    # the L2 distance accumulates over B*N matrices, so stability matters.
    s32 = s.float()
    q32 = q.float()
    q32 = q32 / q32.norm(dim=-1, keepdim=True).clamp_min(eps)
    w, x, y, z = q32.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],     dim=-1),
        torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],     dim=-1),
        torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=-1),
    ], dim=-2)                                                # [..., 3, 3]
    # Σ = R · diag(s²) · Rᵀ computed as (R * s²[None, :]) @ Rᵀ
    s2  = s32 * s32                                           # [..., 3]
    Rs2 = R * s2.unsqueeze(-2)                                # R[..., :, j] *= s²[..., j]
    return Rs2 @ R.transpose(-1, -2)                          # [..., 3, 3]


def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0,
"""

BLOCK_2_RECON_OLD = '''def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0,
                                valid_mask=None, exclude_scale_rotation=False):
    """Element-wise L2 recon loss. valid_mask: optional [B,N] tensor; when given
    (canonical_voxel padding case), padding slots (mask==0) are zeroed in BOTH
    prediction and target before the norm so only real Gaussians are graded. None =
    original behaviour (every slot counted).

    exclude_scale_rotation : DIAGNOSTIC flag. When True, the scale (7:10) and
    rotation (10:14) attributes are dropped from the loss entirely, so their
    decoder heads receive ZERO gradient from this term. The loss becomes
    Pos + color_weight * Col + Opa only. Used to test whether the local
    encoder / token-local decoder can fit the smooth-across-slots attributes
    when the slot-arbitrary ones (Scl, Rot) are removed. The Scl=... Rot=...
    numbers in the training log are still computed via _masked_individual_losses
    but are LOGGING only -- they do not drive training. Pair with --kl_weight 0
    and a small --train_scenes to run a clean overfit diagnostic.
    """
    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).to(prediction.dtype)   # [B,N,1] broadcast over 14 attrs
        prediction = prediction * m
        target     = target * m
    if exclude_scale_rotation:
'''

BLOCK_2_RECON_NEW = '''def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0,
                                valid_mask=None, exclude_scale_rotation=False,
                                use_sigma_loss=False, sigma_weight=1.0):
    """Element-wise L2 recon loss. valid_mask: optional [B,N] tensor; when given
    (canonical_voxel padding case), padding slots (mask==0) are zeroed in BOTH
    prediction and target before the norm so only real Gaussians are graded. None =
    original behaviour (every slot counted).

    exclude_scale_rotation : DIAGNOSTIC flag. When True, the scale (7:10) and
    rotation (10:14) attributes are dropped from the loss entirely, so their
    decoder heads receive ZERO gradient from this term. The loss becomes
    Pos + color_weight * Col + Opa only. Used to test whether the local
    encoder / token-local decoder can fit the smooth-across-slots attributes
    when the slot-arbitrary ones (Scl, Rot) are removed. The Scl=... Rot=...
    numbers in the training log are still computed via _masked_individual_losses
    but are LOGGING only -- they do not drive training. Pair with --kl_weight 0
    and a small --train_scenes to run a clean overfit diagnostic.

    use_sigma_loss : DIAGNOSTIC flag (gauge-removal test). When True, the
    element-wise L2 terms on scale (7:10) and rotation (10:14) are REPLACED with
    a single Frobenius norm on the covariance Σ = R(q)·diag(s²)·R(q)ᵀ, which is
    GAUGE-INVARIANT in (s, q): any orbit representative produces the same Σ, so
    the L2-on-orbit averaging that drives predictions to isotropic blobs no
    longer applies. Position (0:3), color (3:6), opacity (6:7) stay as element-
    wise L2 since they carry no gauge group. The Scl=... Rot=... numbers in the
    log are still computed on raw (s, q) but become uninformative -- the model
    is free to pick any orbit representative, so those per-attribute L2
    magnitudes can grow or oscillate even as the rendered Gaussians become
    correctly anisotropic. THE PLY RENDER IS THE REAL SIGNAL. Takes precedence
    over exclude_scale_rotation if both are set.
    sigma_weight scales the Σ Frobenius term; default 1.0 is the raw Frobenius
    distance summed over all (B*N) Gaussians.
    """
    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).to(prediction.dtype)   # [B,N,1] broadcast over 14 attrs
        prediction = prediction * m
        target     = target * m
    if use_sigma_loss:
        # Gauge-invariant path: replace L2 on (s, q) with Frobenius on Σ.
        # Position, color, opacity carry no gauge -> element-wise L2 unchanged.
        pos_loss   = torch.norm(prediction[:,:,0:3] - target[:,:,0:3], p=2)
        col_loss   = torch.norm(prediction[:,:,3:6] - target[:,:,3:6], p=2) * color_weight
        opa_loss   = torch.norm(prediction[:,:,6:7] - target[:,:,6:7], p=2)
        sigma_pred = _build_sigma_from_sq(prediction[:,:,7:10], prediction[:,:,10:14])
        sigma_targ = _build_sigma_from_sq(target[:,:,7:10],     target[:,:,10:14])
        diff       = sigma_pred - sigma_targ                                # [B, N, 3, 3]
        if valid_mask is not None:
            diff = diff * valid_mask.unsqueeze(-1).unsqueeze(-1).to(diff.dtype)
        sigma_loss = torch.norm(diff, p=2) * float(sigma_weight)
        return (pos_loss + col_loss + opa_loss + sigma_loss) / batch_size
    if exclude_scale_rotation:
'''

BLOCK_3_CLI_OLD = """parser.add_argument('--no_scale_rotation_loss', action='store_true', default=False,
    help='DIAGNOSTIC: drop scale (7:10) and rotation (10:14) from the reconstruction '
         'loss so their decoder heads receive ZERO gradient from this term. The model '
         'still PRODUCES scale/rotation outputs (the Scl=... Rot=... numbers in the log '
         'are still computed for monitoring) but they do not drive training. The Scl/Rot '
         'numbers will be uninformative under this flag and should be ignored when '
         'reading the run. Pairs with --kl_weight 0 and a small --train_scenes to test '
         'whether the local encoder / token-local decoder can fit the smooth-across-slots '
         'attributes (Pos, Col) when the slot-arbitrary ones (Scl, Rot) are removed. '
         'Reference target on 4 chunks with the flat decoder + reordering, KL=0: '
         'Pos~7, Col~3, Opa~0.07.')
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)
"""

BLOCK_3_CLI_NEW = """parser.add_argument('--no_scale_rotation_loss', action='store_true', default=False,
    help='DIAGNOSTIC: drop scale (7:10) and rotation (10:14) from the reconstruction '
         'loss so their decoder heads receive ZERO gradient from this term. The model '
         'still PRODUCES scale/rotation outputs (the Scl=... Rot=... numbers in the log '
         'are still computed for monitoring) but they do not drive training. The Scl/Rot '
         'numbers will be uninformative under this flag and should be ignored when '
         'reading the run. Pairs with --kl_weight 0 and a small --train_scenes to test '
         'whether the local encoder / token-local decoder can fit the smooth-across-slots '
         'attributes (Pos, Col) when the slot-arbitrary ones (Scl, Rot) are removed. '
         'Reference target on 4 chunks with the flat decoder + reordering, KL=0: '
         'Pos~7, Col~3, Opa~0.07.')
parser.add_argument('--sigma_loss', action='store_true', default=False,
    help='DIAGNOSTIC (gauge-removal): replace element-wise L2 on scale (7:10) and '
         'rotation (10:14) with a gauge-invariant Frobenius norm on the covariance '
         'Sigma = R(q)*diag(s^2)*R(q)^T. Tests whether the small-isotropic-Gaussian '
         'failure mode is caused by gauge ambiguity in the (s, q) parameterisation. '
         'Position (0:3), color (3:6), opacity (6:7) stay as element-wise L2 since '
         'they carry no gauge group. Pair with the same setup as a no_scale_rotation '
         'overfit baseline so the runs are directly comparable. The Scl=.../Rot=... '
         'numbers in the log become uninformative under this flag -- inspect the '
         'reconstructed PLYs to confirm Gaussians become anisotropic. Takes precedence '
         'over --no_scale_rotation_loss if both are set.')
parser.add_argument('--sigma_weight', type=float, default=1.0,
    help='Multiplier on the Sigma Frobenius term when --sigma_loss is on. 1.0 = raw '
         'Frobenius distance summed over all (B*N) Gaussians. Increase if the early '
         'Pos/Col loss dominates and Sigma stays unfit; decrease if Sigma explodes and '
         'Pos/Col stop converging. Default 1.0 matches the magnitude of the prior L2 '
         'scale+rotation term roughly for typical normalised splat sizes.')
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)
"""

# Training-loop call site (8-space indent inside `for i_batch, batch_data in enumerate`).
BLOCK_4A_TRAIN_OLD = """        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                 valid_mask=vmask_gpu,
                                                 exclude_scale_rotation=args.no_scale_rotation_loss)
"""

BLOCK_4A_TRAIN_NEW = """        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                 valid_mask=vmask_gpu,
                                                 exclude_scale_rotation=args.no_scale_rotation_loss,
                                                 use_sigma_loss=args.sigma_loss,
                                                 sigma_weight=args.sigma_weight)
"""

# evaluate_model call site (12-space indent inside `with torch.no_grad(): for batch_data ...`).
BLOCK_4B_EVAL_OLD = """            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                     valid_mask=vmask_gpu,
                                                     exclude_scale_rotation=args.no_scale_rotation_loss)
"""

BLOCK_4B_EVAL_NEW = """            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                     valid_mask=vmask_gpu,
                                                     exclude_scale_rotation=args.no_scale_rotation_loss,
                                                     use_sigma_loss=args.sigma_loss,
                                                     sigma_weight=args.sigma_weight)
"""

BLOCK_5_CKPT_OLD = """    'canonical_voxel':            args.canonical_voxel,
    'voxel_res':                  args.voxel_res,
    'voxel_snap':                 args.voxel_snap,
}
"""

BLOCK_5_CKPT_NEW = """    'canonical_voxel':            args.canonical_voxel,
    'voxel_res':                  args.voxel_res,
    'voxel_snap':                 args.voxel_snap,
    'sigma_loss':                 args.sigma_loss,
    'sigma_weight':               args.sigma_weight,
}
"""

BLOCK_6_SUMMARY_OLD = """    if args.canonical_voxel:
        print(f"  CANONICAL VOXEL : ON  R={args.voxel_res}  snap={args.voxel_snap}  "
              f"(gauge-removal: one rep Gaussian/occupied voxel, padding masked in loss)")
    print(f"  cross_recon={args.cross_recon_weight} ortho={args.ortho_weight}")
"""

BLOCK_6_SUMMARY_NEW = """    if args.canonical_voxel:
        print(f"  CANONICAL VOXEL : ON  R={args.voxel_res}  snap={args.voxel_snap}  "
              f"(gauge-removal: one rep Gaussian/occupied voxel, padding masked in loss)")
    if args.sigma_loss:
        print(f"  SIGMA LOSS      : ON  weight={args.sigma_weight}  "
              f"(gauge-INVARIANT Frobenius on Sigma=R(q)*diag(s^2)*R(q)^T; "
              f"replaces element-wise L2 on (s, q))")
    print(f"  cross_recon={args.cross_recon_weight} ortho={args.ortho_weight}")
"""

# Sentinel: append after the existing no_scale_rotation sentinel so re-runs are no-ops.
BLOCK_7_SENTINEL_OLD = "# === NO_SCALE_ROTATION_PATCH_APPLIED ==="
BLOCK_7_SENTINEL_NEW = "# === NO_SCALE_ROTATION_PATCH_APPLIED ===\n" + SENTINEL


BLOCKS = [
    ("1: insert _build_sigma_from_sq helper",           BLOCK_1_HELPER_OLD,    BLOCK_1_HELPER_NEW),
    ("2: extend compute_reconstruction_loss",            BLOCK_2_RECON_OLD,     BLOCK_2_RECON_NEW),
    ("3: add --sigma_loss / --sigma_weight CLI flags",   BLOCK_3_CLI_OLD,       BLOCK_3_CLI_NEW),
    ("4a: training-loop call site",                      BLOCK_4A_TRAIN_OLD,    BLOCK_4A_TRAIN_NEW),
    ("4b: evaluate_model call site",                     BLOCK_4B_EVAL_OLD,     BLOCK_4B_EVAL_NEW),
    ("5: checkpoint metadata",                           BLOCK_5_CKPT_OLD,      BLOCK_5_CKPT_NEW),
    ("6: startup summary notification",                  BLOCK_6_SUMMARY_OLD,   BLOCK_6_SUMMARY_NEW),
    ("7: append sentinel",                               BLOCK_7_SENTINEL_OLD,  BLOCK_7_SENTINEL_NEW),
]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"Target file: {path}")
    if not os.path.exists(path):
        print(f"ERROR: target file not found: {path}", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if SENTINEL in content:
        print(f"Sentinel '{SENTINEL}' already present -- patch is a no-op. Exiting.")
        return 0

    # Pre-flight: verify every old_str appears EXACTLY once. Bail on any mismatch
    # before touching the file, so a partial patch is impossible.
    print("\nPre-flight: verifying anchor strings...")
    errors = []
    for name, old, _new in BLOCKS:
        n = content.count(old)
        if n != 1:
            errors.append(f"  Block {name}: anchor found {n} times (expected 1)")
        else:
            print(f"  Block {name}: anchor OK")
    if errors:
        print("\nERROR: anchor mismatch -- the file does not match the patch's expected"
              " state.\nThis usually means another patch has already modified the same"
              " region,\nor the file is not the one this patch was written against.", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 2

    # Apply blocks in order. Each replacement narrows the next anchor's location
    # but does not break it (we never overlap regions).
    print("\nApplying blocks...")
    patched = content
    for name, old, new in BLOCKS:
        before = len(patched)
        patched = patched.replace(old, new, 1)
        delta = len(patched) - before
        print(f"  Block {name}: applied (+{delta} chars)")

    if SENTINEL not in patched:
        print(f"ERROR: sentinel '{SENTINEL}' missing from patched content -- aborting"
              " without writing.", file=sys.stderr)
        return 3

    # Backup the original next to it (separate suffix from the no_scale_rotation backup).
    backup = path + ".bak.sigma"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"\nBackup written: {backup}")
    else:
        print(f"\nBackup already exists, not overwriting: {backup}")

    # Atomic write: tmp file then rename.
    tmp = path + ".tmp.sigma"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(patched)
    os.replace(tmp, path)
    print(f"Patched file written: {path}")
    print("\nDone. Verify with:")
    print(f"  grep -n 'sigma_loss\\|_build_sigma_from_sq' {path} | head -20")
    print("\nRun the test with --sigma_loss --sigma_weight 1.0 added to your usual command.")
    print("The reference job is can3tok_overfit_sigma_loss.job (small overfit; ~1h on 1 H100).")
    return 0


if __name__ == "__main__":
    sys.exit(main())