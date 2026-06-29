"""
apply_no_scale_rotation_patch.py
================================
Patch gs_can3tok_2.py in place to add --no_scale_rotation_loss flag.

WHAT THIS PATCH DOES:
  Adds a --no_scale_rotation_loss CLI flag that DROPS the scale (7:10) and
  rotation (10:14) terms from the reconstruction loss. The model still
  PRODUCES scale and rotation outputs, but they receive ZERO gradient signal
  from the recon loss, so they drift based only on initialization and any
  indirect gradient leaking through shared encoder/backbone layers from the
  Pos/Col/Opa terms. The Scl=... Rot=... numbers in the log are still
  computed (via _masked_individual_losses) but they are LOGGING ONLY -- they
  no longer drive training.

DIAGNOSTIC PURPOSE:
  Tests whether the local encoder + token-local decoder can fit Position and
  Color (the smooth-across-slots attributes) when the non-smooth attributes
  (Scale, Rotation) are removed from the loss.

  Hypothesis: the shared MLP in the token-local decoder cannot represent
  the slot-arbitrary Scl/Rot pattern (78 unrelated values per token from a
  32-dim code), but it IS able to represent the smooth Pos/Col pattern. If
  this is the actual mechanism, dropping Scl/Rot from the loss should let
  Pos and Col fall to roughly flat-decoder territory:

    Reference (flat decoder + reordering, 4 scenes, KL=0, 4000 epochs):
      Pos = 7.2   Col = 2.8   Opa = 0.07

  Reading the result of THIS run (local pair + reordering + the new flag):

    Pos near 7 and Col near 3
      -> local architecture is fine for the smooth subset; the floor was
         specifically due to per-slot arbitrariness in Scl/Rot meeting a
         shared, smoothness-biased decoder. Conclusion: architecture-loss
         mismatch on the non-smooth attributes only.

    Pos still well above 7 (e.g. >15) even with Scl/Rot out of the loss
      -> the per-token decoder also has a capacity ceiling on the SMOOTH
         attributes. Smoothness mismatch is part of the story, not the
         whole story. A capacity widening test (embed_dim=64) would then
         be the next probe.

HOW TO RUN:
  python3 apply_no_scale_rotation_patch.py path/to/gs_can3tok_2.py

  Backs up to gs_can3tok_2.py.bak first. Idempotent (running again is a
  no-op). Pre-flight verifies all 4 anchors match exactly once before
  writing anything; aborts with exit 2 on any mismatch.

PAIRS WITH:
  can3tok_overfit_local_no_scl_rot.job
  -- the diagnostic job script that enables --local_encoder --local_window 1
     --token_local_decoder --morton_order on 4 chunks, KL=0, plus the new
     --no_scale_rotation_loss flag.
"""

import sys
import os
import shutil
from pathlib import Path


PATCH_MARKER = "# === NO_SCALE_ROTATION_PATCH_APPLIED ==="


# =============================================================================
# PATCH BLOCKS
# =============================================================================
# Each entry is (description, find_text, replacement_text). find_text must
# occur EXACTLY ONCE in the file. The script verifies every match in a
# pre-flight pass before applying any edit, so a failed match aborts cleanly
# without partial-patching.

PATCHES = []


# -- Patch 1: rewrite compute_reconstruction_loss to add exclude_scale_rotation
PATCHES.append((
    "1/4: Add exclude_scale_rotation parameter to compute_reconstruction_loss",
    '''def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0, valid_mask=None):
    """Element-wise L2 recon loss. valid_mask: optional [B,N] tensor; when given
    (canonical_voxel padding case), padding slots (mask==0) are zeroed in BOTH
    prediction and target before the norm so only real Gaussians are graded. None =
    original behaviour (every slot counted)."""
    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).to(prediction.dtype)   # [B,N,1] broadcast over 14 attrs
        prediction = prediction * m
        target     = target * m
    if color_weight == 1.0:
        return torch.norm(prediction - target, p=2) / batch_size
    return (torch.norm(prediction[:,:,0:3] - target[:,:,0:3], p=2)
          + torch.norm(prediction[:,:,3:6] - target[:,:,3:6], p=2) * color_weight
          + torch.norm(prediction[:,:,6:]  - target[:,:,6:],  p=2)) / batch_size''',
    '''def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0,
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
        # Diagnostic path: split the loss so the third term covers ONLY opacity
        # (6:7). Scale (7:10) and rotation (10:14) get no gradient. Always go
        # through this split even when color_weight==1.0 (i.e. skip the
        # whole-tensor norm shortcut below).
        return (torch.norm(prediction[:,:,0:3] - target[:,:,0:3], p=2)
              + torch.norm(prediction[:,:,3:6] - target[:,:,3:6], p=2) * color_weight
              + torch.norm(prediction[:,:,6:7] - target[:,:,6:7], p=2)
              ) / batch_size
    if color_weight == 1.0:
        return torch.norm(prediction - target, p=2) / batch_size
    return (torch.norm(prediction[:,:,0:3] - target[:,:,0:3], p=2)
          + torch.norm(prediction[:,:,3:6] - target[:,:,3:6], p=2) * color_weight
          + torch.norm(prediction[:,:,6:]  - target[:,:,6:],  p=2)) / batch_size''',
))


# -- Patch 2: add the --no_scale_rotation_loss CLI flag --------------------
PATCHES.append((
    "2/4: Add --no_scale_rotation_loss CLI flag",
    '''parser.add_argument('--color_loss_weight',    type=float, default=1.0)
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)''',
    '''parser.add_argument('--color_loss_weight',    type=float, default=1.0)
parser.add_argument('--no_scale_rotation_loss', action='store_true', default=False,
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
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)''',
))


# -- Patch 3: pass the flag at the training-loop call site ------------------
PATCHES.append((
    "3/4: Pass exclude_scale_rotation in the training loop",
    '''        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                 valid_mask=vmask_gpu)''',
    '''        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                 valid_mask=vmask_gpu,
                                                 exclude_scale_rotation=args.no_scale_rotation_loss)''',
))


# -- Patch 4: pass the flag at the evaluate_model call site -----------------
# Same call but with the deeper indentation it has inside evaluate_model.
PATCHES.append((
    "4/4: Pass exclude_scale_rotation in evaluate_model",
    '''            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                     valid_mask=vmask_gpu)''',
    '''            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                     valid_mask=vmask_gpu,
                                                     exclude_scale_rotation=args.no_scale_rotation_loss)''',
))


# =============================================================================
# APPLICATION DRIVER
# =============================================================================

def apply_patch(target_path: Path) -> int:
    if not target_path.exists():
        print(f"ERROR: {target_path} does not exist", file=sys.stderr)
        return 1

    source = target_path.read_text(encoding='utf-8')

    # Idempotency: detect that the patch was already applied.
    if "--no_scale_rotation_loss" in source and "exclude_scale_rotation" in source:
        print(f"[OK] Patch already applied to {target_path}. No changes made.")
        print(f"     (detected '--no_scale_rotation_loss' and 'exclude_scale_rotation' "
              f"in the file)")
        return 0
    if PATCH_MARKER in source:
        print(f"[OK] Patch marker {PATCH_MARKER} found. No changes made.")
        return 0

    # Pre-flight: verify every find_text matches EXACTLY ONCE before any write.
    print(f"Pre-flight: verifying all {len(PATCHES)} patch blocks match in {target_path}")
    print(f"           (each find_text must occur exactly once)")
    print()

    failures = []
    for i, (desc, find, _repl) in enumerate(PATCHES, start=1):
        count = source.count(find)
        status = "OK"   if count == 1 else f"FAIL ({count} matches)"
        print(f"  [{status:>13s}] {desc}")
        if count != 1:
            failures.append((i, desc, count))

    if failures:
        print()
        print("PRE-FLIGHT FAILED: one or more patch blocks did not match exactly once.")
        print("This usually means your gs_can3tok_2.py is a different version than the")
        print("one this patch was written for. Check the failing blocks above and either")
        print("update the patch to match your file, or apply the changes manually.")
        for i, desc, count in failures:
            print(f"  - Block {i}: {desc} ({count} matches; expected 1)")
        return 2

    # Backup before writing.
    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    shutil.copy2(target_path, backup_path)
    print()
    print(f"Backup saved to {backup_path}")

    # Apply patches sequentially. Each operates on the running buffer.
    print()
    print(f"Applying {len(PATCHES)} patches...")
    patched = source
    for i, (desc, find, repl) in enumerate(PATCHES, start=1):
        patched = patched.replace(find, repl, 1)
        print(f"  [APPLIED] {desc}")

    # Add a footer marker so re-runs short-circuit even if individual block
    # detection were to fail in the future.
    patched += f"\n\n{PATCH_MARKER}\n"

    # Write atomically (write to temp file, then rename).
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(patched, encoding='utf-8')
    os.replace(tmp_path, target_path)

    print()
    print(f"[DONE] Patched {target_path} successfully.")
    print(f"       Original at  {backup_path}")
    print(f"       Lines added: {patched.count(chr(10)) - source.count(chr(10))}")
    print()
    print("Next steps:")
    print(f"  1) Sanity-check with:  python3 -c \"import ast; ast.parse(open('{target_path}').read())\"")
    print("  2) Submit the diagnostic job:  sbatch can3tok_overfit_local_no_scl_rot.job")
    print("  3) After the run, read the LAST epoch line and check:")
    print("       Pos near 7 and Col near 3  -> local arch is fine for smooth attrs,")
    print("                                     the floor was specifically the Scl/Rot")
    print("                                     non-smoothness vs shared decoder mismatch.")
    print("       Pos still well above 7     -> capacity ceiling on the smooth attrs too;")
    print("                                     not purely a smoothness story.")
    print("       (Ignore the Scl/Rot numbers in the log -- under this flag they get no")
    print("        gradient and drift, so they are uninformative.)")
    return 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 apply_no_scale_rotation_patch.py <path-to-gs_can3tok_2.py>",
              file=sys.stderr)
        print()
        print("Example:")
        print("  python3 apply_no_scale_rotation_patch.py /home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion/gs_can3tok_2.py")
        return 1
    return apply_patch(Path(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())