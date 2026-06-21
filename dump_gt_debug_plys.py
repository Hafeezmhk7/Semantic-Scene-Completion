"""
Isolate the GT-rendering pipeline from training.

Run from the repo root in the can3tok env:
    python dump_gt_debug_plys.py            # chunk index 0
    python dump_gt_debug_plys.py 5          # chunk index 5

Writes two PLYs into gt_debug_plys/ for ONE chunk, with NO model involved:
  full_raw.ply        every Gaussian, no sampling, no merge  (the cleanest possible GT)
  canonical_voxel.ply the res-64 merge -- exactly what training rendered as GT

Open both in https://supersplat.at and compare.
  * full_raw clean, canonical_voxel a cloud  -> canonical_voxel is degrading the GT
                                                (hypotheses 1/2: downsample / rep-bias)
  * full_raw ALSO a cloud                    -> rendering / normalization / units issue
                                                affecting everything (the bigger finding)

It also PRINTS the raw scale/opacity/color ranges and the valid-rep count M, which
answer the units and over-downsample questions on their own.
"""
import os, sys
import numpy as np

# repo-local imports (run from the repo root)
from gs_dataset_scenesplat import normalize_with_norm_factor, canonical_voxel_merge
from gs_ply_reconstructor import reconstruct_single_scene

CHUNK_ROOT = ("/home/yli7/scratch/datasets/gaussian_world/preprocessed/"
              "interior_gs/train_grid1.0cm_chunk8x8_stride6x6")
IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = "gt_debug_plys"
os.makedirs(OUT, exist_ok=True)

scene_dirs = sorted(os.path.join(CHUNK_ROOT, d) for d in os.listdir(CHUNK_ROOT)
                    if os.path.isdir(os.path.join(CHUNK_ROOT, d)))
sd = scene_dirs[IDX]
print(f"\nchunk[{IDX}]: {sd}\n")

coord   = np.load(os.path.join(sd, 'coord.npy'))
color   = np.load(os.path.join(sd, 'color.npy')).astype(np.float32)
scale   = np.load(os.path.join(sd, 'scale.npy'))
quat    = np.load(os.path.join(sd, 'quat.npy'))
opacity = np.load(os.path.join(sd, 'opacity.npy'))
try:
    segment  = np.load(os.path.join(sd, 'segment.npy'))
    instance = np.load(os.path.join(sd, 'instance.npy'))
except FileNotFoundError:
    segment  = np.full(len(coord), -1, np.int16)
    instance = np.full(len(coord), -1, np.int32)

print(f"raw N = {len(coord)}")
print(f"  raw scale  : min={scale.min():.4f} max={scale.max():.4f} mean={scale.mean():.4f}"
      f"   <- if many are NEGATIVE this is LOG-scale, not linear metres (units bug)")
print(f"  raw opacity: min={opacity.min():.4f} max={opacity.max():.4f} mean={opacity.mean():.4f}"
      f"   <- expect ~[0,1]")
print(f"  raw color  : min={color.min():.1f} max={color.max():.1f}   <- expect 0..255")

coord_n, scale_n = normalize_with_norm_factor(
    coord, scale, scene_dir=sd, target_radius=10.0, scale_norm_mode='linear')
ext = coord_n.max(0) - coord_n.min(0)
print(f"\n  normed coord range: [{coord_n.min():.2f}, {coord_n.max():.2f}]")
print(f"  normed per-axis extent: [{ext[0]:.2f}, {ext[1]:.2f}, {ext[2]:.2f}]"
      f"   <- a chunk filling only a small part of the 20-wide frame is hypothesis 1")
print(f"  normed scale: min={scale_n.min():.4f} max={scale_n.max():.4f} "
      f"mean={scale_n.mean():.4f} median={np.median(scale_n):.4f}"
      f"   <- compare to the extent above; >~1 means blooming splats")

col01 = np.clip(color / 255.0, 0.0, 1.0)


def dump(name, c, col, op, sc, q):
    pred = np.concatenate([c, col, op.reshape(-1, 1), sc, q], axis=1).astype(np.float32)
    reconstruct_single_scene(pred, os.path.join(OUT, name), max_sh_degree=3, verbose=True)


# 1) FULL raw chunk: every Gaussian, no sampling, no merge.
dump("full_raw.ply", coord_n, col01, opacity, scale_n, quat)

# 2) Canonical-voxel merge (res 64): exactly what training used as the GT.
(cc, ccol, cscale, cquat, copa, cseg, cinst, valid) = canonical_voxel_merge(
    coord_n, col01, scale_n, quat, opacity, segment, instance,
    voxel_res=64, frame_radius=10.0, target_points=40000, snap_to_center=False)
M = int(valid.sum())
print(f"\n  canonical_voxel: valid reps M = {M} / 40000  "
      f"({100.0 * M / 40000:.1f}% real, rest is invisible padding)"
      f"   <- small M is hypothesis 1 confirmed")
keep = valid > 0
dump("canonical_voxel.ply", cc[keep], ccol[keep], copa[keep], cscale[keep], cquat[keep])

print(f"\nWrote PLYs to {OUT}/  -- open full_raw.ply and canonical_voxel.ply in SuperSplat.\n")