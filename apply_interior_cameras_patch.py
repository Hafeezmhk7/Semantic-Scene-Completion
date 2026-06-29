#!/usr/bin/env python3
"""
apply_interior_cameras_patch.py
===============================

In-place patch for render_loss_helpers.py that switches the synthesised cameras
used by the render-loss diagnostic from EXTERIOR (1.5 x scene_radius, look-at-
centroid) to INTERIOR (0.1 x scene_radius, look-outward-random-direction).

WHY
---
SceneSplat-7K is indoor scenes; the original 3DGS fits used cameras INSIDE the
rooms looking outward at walls. The first render_loss_helpers.py shipped with
exterior-looking-inward cameras as a defensible default for arbitrary 3DGS
data. The epoch-0 sanity PNGs from the first run showed the exterior-camera
GT renders are fragmented surface-back textures against black -- the rasteriser
IS wired correctly (you can clearly see coherent panel/wall structure in the
GT renders), but most of each frame is empty space behind the scene and the
render loss has weaker gradient than it should.

Interior cameras give renders where each frame is filled with wall/floor/
furniture structure, matching how the original 3DGS captures were taken and
giving the render loss a much stronger shape-supervision signal.

WHAT IT CHANGES
---------------
Only the sample_cameras function. Everything else in render_loss_helpers.py
(rasteriser wrapper, SSIM, RenderLossModule, sanity-render helper) is
untouched. The function signature is unchanged so the rest of the code keeps
working.

USAGE
-----
    python apply_interior_cameras_patch.py

Idempotent: a sentinel comment is appended; re-running is a no-op.
A backup is written to render_loss_helpers.py.bak.interior_cameras.

NEXT STEPS AFTER PATCHING
-------------------------
1. scancel the currently-running render-loss job (if it's still running).
2. Resubmit can3tok_overfit_render_loss.job.
3. The new epoch-0 sanity PNGs in render_sanity/epoch_0000/ should now show
   GT renders that look like room interiors (walls, floor, furniture) filling
   most of the frame, rather than fragmented surface backs against black.
"""
from __future__ import annotations
import os, sys, shutil

SENTINEL = "# === INTERIOR_CAMERAS_PATCH_APPLIED ==="
DEFAULT_PATH = "/home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion/render_loss_helpers.py"

# The original sample_cameras function as embedded in the patch's HELPERS_CONTENT.
# Anchored on the docstring + cam_radius line so we catch it regardless of which
# whitespace style the helpers file has.
OLD_FUNC = '''def sample_cameras(centroid, radius, num_cameras, image_size, fov_deg, device, dtype):
    """Sample num_cameras around a scene centroid, all looking inward."""
    cam_radius = 1.5 * float(radius)
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    viewmats = []
    for _ in range(int(num_cameras)):
        theta = float(torch.rand((), device='cpu').item()) * 2.0 * math.pi
        phi = (float(torch.rand((), device='cpu').item()) - 0.5) * (2.0 / 3.0 * math.pi)
        x = cam_radius * math.cos(phi) * math.cos(theta)
        y = cam_radius * math.sin(phi)
        z = cam_radius * math.cos(phi) * math.sin(theta)
        eye = centroid + torch.tensor([x, y, z], device=device, dtype=dtype)
        target = centroid + (torch.rand(3, device=device, dtype=dtype) - 0.5) * (0.1 * radius)
        viewmats.append(_look_at_view_matrix(eye, target, up))
    viewmats = torch.stack(viewmats, dim=0)
    K = _intrinsic_matrix(fov_deg, image_size, device, dtype)
    Ks = K.unsqueeze(0).expand(int(num_cameras), -1, -1).contiguous()
    return viewmats, Ks'''

NEW_FUNC = '''def sample_cameras(centroid, radius, num_cameras, image_size, fov_deg, device, dtype):
    """Sample num_cameras INSIDE a scene with random outward-looking directions.

    Cameras are placed near the centroid (jittered by 0.1 * radius along a random
    direction) and look outward along a random azimuth with bounded elevation
    (+/- 30 deg, to avoid every camera pointing straight at floor or ceiling).

    Designed for INTERIOR 3DGS scenes (e.g. SceneSplat-7K rooms) where the
    original captures were taken from inside the room looking at walls/floor/
    ceiling/furniture. Renders fill the frame with coherent surface structure
    rather than fragmented surface backs against black.

    Up axis is +Z to match the SceneSplat scene-gravity convention (the default
    of --aug_yaw_axis is z).
    """
    cam_offset = 0.1 * float(radius)
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    viewmats = []
    for _ in range(int(num_cameras)):
        # Camera position: small jitter around centroid in a random direction.
        offset_dir = torch.randn(3, device=device, dtype=dtype)
        offset_dir = offset_dir / offset_dir.norm().clamp_min(1e-8)
        eye = centroid + offset_dir * cam_offset
        # Look direction: random azimuth, bounded elevation. Convert to a unit
        # vector in the scene frame (X-forward / Y-forward / Z-up convention).
        theta = float(torch.rand((), device='cpu').item()) * 2.0 * math.pi
        phi = (float(torch.rand((), device='cpu').item()) - 0.5) * (math.pi / 3.0)
        look_dir = torch.tensor([
            math.cos(phi) * math.cos(theta),
            math.cos(phi) * math.sin(theta),
            math.sin(phi),
        ], device=device, dtype=dtype)
        # Target far along the look direction so the view ray hits the far wall.
        target = eye + look_dir * radius
        viewmats.append(_look_at_view_matrix(eye, target, up))
    viewmats = torch.stack(viewmats, dim=0)
    K = _intrinsic_matrix(fov_deg, image_size, device, dtype)
    Ks = K.unsqueeze(0).expand(int(num_cameras), -1, -1).contiguous()
    return viewmats, Ks'''


def main() -> int:
    path = DEFAULT_PATH
    # Allow override but ignore --force-* style flags that are not paths
    for a in sys.argv[1:]:
        if not a.startswith('-') and os.path.exists(a):
            path = a
            break

    print(f"Target file: {path}")
    if not os.path.exists(path):
        print(f"ERROR: target file not found: {path}", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if SENTINEL in content:
        print(f"Sentinel '{SENTINEL}' already present -- patch is a no-op. Exiting.")
        return 0

    n = content.count(OLD_FUNC)
    if n == 0:
        print("ERROR: original sample_cameras function not found.\n"
              "Possible causes:\n"
              "  1. The helpers file has already been modified (manually or by another patch).\n"
              "  2. The helpers file shipped with the standalone render_loss_helpers.py\n"
              "     instead of the patch's embedded copy (very slightly different whitespace).\n"
              "Try removing render_loss_helpers.py and re-running apply_render_loss_patch.py\n"
              "with --force-helpers to install the embedded copy, then re-run this patch.",
              file=sys.stderr)
        return 2
    if n > 1:
        print(f"ERROR: original sample_cameras function found {n} times (expected 1).",
              file=sys.stderr)
        return 2

    patched = content.replace(OLD_FUNC, NEW_FUNC, 1)
    if not patched.rstrip().endswith(SENTINEL):
        patched = patched.rstrip() + "\n\n" + SENTINEL + "\n"

    backup = path + ".bak.interior_cameras"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"Backup written: {backup}")
    else:
        print(f"Backup already exists, not overwriting: {backup}")

    tmp = path + ".tmp.interior_cameras"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(patched)
    os.replace(tmp, path)
    print(f"Patched: {path}")

    print("\nDone. The render-loss helpers now sample INTERIOR cameras.")
    print("Verify with:")
    print(f"  grep -n 'INTERIOR_CAMERAS_PATCH_APPLIED' {path}")
    print(f"  grep -A2 'def sample_cameras' {path} | head -5")
    print("\nIf the current SLURM job is still running, scancel it now.")
    print("Then resubmit:  sbatch can3tok_overfit_render_loss.job")
    print("\nThe new epoch-0 sanity PNGs should show GT renders that look like room")
    print("interiors -- walls, floors, furniture filling most of the frame -- rather")
    print("than fragmented surface backs against black.")
    return 0


if __name__ == "__main__":
    sys.exit(main())