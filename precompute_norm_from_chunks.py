"""
precompute_norm_from_chunks.py
================================
Computes global normalization parameters for each scene by combining
ALL chunks that belong to it — exactly like the original Can3Tok approach
which uses the full scene's SfM point cloud.

KEY IDEA (from the screenshot):
  0201_840151_0  ← scene 0201_840151, chunk 0
  0201_840151_1  ← scene 0201_840151, chunk 1
  ...
  0201_840151_7  ← scene 0201_840151, chunk 7

  All 8 chunks combined = full scene coordinate coverage.
  norm_factor computed from union of all chunk coords = global scene frame.
  Save norm_factor.npy to every chunk so they share one coordinate system.

WHY THIS FIXES POSITION CONVERGENCE:
  Before: each chunk is independently normalised into its own 10m sphere.
    Position [2, 0, 1] means different things in different chunks.
    Decoder cannot learn any generalizable position rule.

  After: all chunks from 0201_840151 share the same coordinate frame.
    Position [2, 0, 1] always refers to the same physical location.
    Decoder can learn: "scenes with this z tend to have furniture here."

norm_factor.npy format: [cx, cy, cz, scale_factor]
  Apply as: coord_norm = (coord - [cx, cy, cz]) * scale_factor

USAGE:
  # Dry run first to check naming:
  python precompute_norm_from_chunks.py \
    --chunks_dir /home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6 \
    --dry_run

  # Full run:
  python precompute_norm_from_chunks.py \
    --chunks_dir /home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6 \
    --workers 16

  # Verify after running:
  python precompute_norm_from_chunks.py \
    --chunks_dir ... \
    --verify_only
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ── Parse scene ID from chunk directory name ──────────────────────────────────

def get_scene_id(chunk_dir_name):
    """
    Extract scene ID by removing the last _N suffix.
    
    0201_840151_0  → 0201_840151
    0201_840151_12 → 0201_840151
    scene0001_3    → scene0001
    
    Works for any naming where chunk index is the last underscore-separated token.
    """
    return chunk_dir_name.rsplit('_', 1)[0]


# ── Compute norm_factor from all chunks of one scene ─────────────────────────

def compute_scene_norm_factor(scene_id, chunk_dirs, target_radius=10.0):
    """
    Combine coordinates from ALL chunks of a scene, then compute
    center and scale from the union — equivalent to using the full scene.

    Returns norm_factor = [cx, cy, cz, scale_factor]
    """
    all_coords = []

    for chunk_dir in chunk_dirs:
        coord_path = os.path.join(chunk_dir, 'coord.npy')
        if not os.path.exists(coord_path):
            continue
        try:
            coord = np.load(coord_path)
            # Subsample to max 5000 per chunk for speed (still representative)
            if len(coord) > 5000:
                idx   = np.random.choice(len(coord), 5000, replace=False)
                coord = coord[idx]
            all_coords.append(coord)
        except Exception as e:
            continue

    if not all_coords:
        return None, f"No coord.npy found in any chunk of {scene_id}"

    # Combine all chunk coordinates = full scene point cloud
    combined = np.concatenate(all_coords, axis=0)   # [N_total, 3]

    # Compute global center and scale (identical formula to normalize_to_canonical_sphere)
    center      = combined.mean(axis=0)
    coord_c     = combined - center
    max_dist    = np.linalg.norm(coord_c, axis=1).max()
    if max_dist < 1e-6:
        max_dist = 1.0
    scale_factor = target_radius / (max_dist * 1.1)

    norm_factor = np.array([center[0], center[1], center[2], scale_factor],
                           dtype=np.float32)
    return norm_factor, None


# ── Worker: process one scene ─────────────────────────────────────────────────

def process_one_scene(args):
    scene_id, chunk_dirs, target_radius, overwrite = args

    # Check if already done (all chunks have norm_factor.npy)
    if not overwrite:
        all_done = all(
            os.path.exists(os.path.join(cd, 'norm_factor.npy'))
            for cd in chunk_dirs
        )
        if all_done:
            return scene_id, len(chunk_dirs), 0, None   # already done

    norm_factor, err = compute_scene_norm_factor(scene_id, chunk_dirs, target_radius)
    if err:
        return scene_id, 0, 0, err

    # Save to every chunk
    saved = 0
    for chunk_dir in chunk_dirs:
        out_path = os.path.join(chunk_dir, 'norm_factor.npy')
        np.save(out_path, norm_factor)
        saved += 1

    return scene_id, saved, saved, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Precompute global normalization from combined chunks')
    p.add_argument('--chunks_dir',    required=True,
                   help='Path to train_grid1.0cm_chunk8x8_stride6x6/')
    p.add_argument('--target_radius', type=float, default=10.0)
    p.add_argument('--workers',       type=int,   default=8)
    p.add_argument('--overwrite',     action='store_true')
    p.add_argument('--dry_run',       action='store_true',
                   help='Show grouping without writing anything')
    p.add_argument('--verify_only',   action='store_true',
                   help='Check consistency of saved norm_factor.npy files')
    args = p.parse_args()

    chunks_dir = Path(args.chunks_dir)
    np.random.seed(42)

    # ── Group chunk directories by scene ID ───────────────────────────────────
    all_chunk_dirs = sorted([d for d in chunks_dir.iterdir() if d.is_dir()])
    scene_to_chunks = defaultdict(list)
    for d in all_chunk_dirs:
        sid = get_scene_id(d.name)
        scene_to_chunks[sid].append(str(d))

    total_chunks  = len(all_chunk_dirs)
    total_scenes  = len(scene_to_chunks)
    chunk_counts  = [len(v) for v in scene_to_chunks.values()]

    print(f"\n{'='*60}")
    print(f"  CHUNK NORMALIZATION PRECOMPUTATION")
    print(f"{'='*60}")
    print(f"  Chunks dir:    {chunks_dir}")
    print(f"  Total chunks:  {total_chunks}")
    print(f"  Unique scenes: {total_scenes}")
    print(f"  Chunks/scene:  min={min(chunk_counts)}  "
          f"mean={np.mean(chunk_counts):.1f}  max={max(chunk_counts)}")

    # Show a few examples
    print(f"\n  NAMING CONVENTION CHECK (first 3 scenes):")
    for sid, dirs in list(scene_to_chunks.items())[:3]:
        names = [Path(d).name for d in dirs[:4]]
        print(f"    Scene '{sid}': {len(dirs)} chunks — {names}{'...' if len(dirs)>4 else ''}")

    if args.dry_run:
        print(f"\n  DRY RUN — no files written.")
        print(f"  If scene IDs and chunks look correct, remove --dry_run to proceed.")
        return

    # ── Verify mode ───────────────────────────────────────────────────────────
    if args.verify_only:
        print(f"\n  VERIFICATION:")
        n_ok = n_missing = n_inconsistent = n_checked = 0

        for sid, dirs in tqdm(list(scene_to_chunks.items()),
                               desc="  Verifying"):
            nf_list = []
            has_missing = False
            for d in dirs:
                nf_path = os.path.join(d, 'norm_factor.npy')
                if os.path.exists(nf_path):
                    nf_list.append(np.load(nf_path))
                else:
                    has_missing = True
                    n_missing += 1

            if has_missing:
                continue

            n_checked += 1
            if len(nf_list) >= 2:
                all_same = all(np.allclose(nf_list[0], nf, atol=1e-5)
                               for nf in nf_list)
                if all_same:
                    n_ok += 1
                else:
                    n_inconsistent += 1
                    print(f"  INCONSISTENT: {sid}")
                    for i, nf in enumerate(nf_list[:3]):
                        print(f"    chunk {i}: center=({nf[0]:.3f},{nf[1]:.3f},{nf[2]:.3f})  scale={nf[3]:.4f}")

        print(f"\n  Scenes fully checked:           {n_checked}/{total_scenes}")
        print(f"  Consistent (all chunks match):  {n_ok}")
        print(f"  Missing norm_factor.npy:        {n_missing} chunks")
        print(f"  Inconsistent:                   {n_inconsistent}")

        if n_missing == 0 and n_inconsistent == 0:
            print(f"\n  ✓ ALL GOOD — normalization is consistent across all chunks")
            print(f"  Next step: update gs_dataset_scenesplat.py to use norm_factor.npy")
        elif n_missing > 0:
            print(f"\n  ✗ {n_missing} chunks missing norm_factor.npy")
            print(f"  Run without --verify_only to compute them")
        return

    # ── Compute and save ──────────────────────────────────────────────────────
    print(f"\n  Computing norm_factor from combined chunks...")
    print(f"  Each scene: load all chunk coords → combine → compute center+scale → save")
    print(f"  Workers: {args.workers}  |  Overwrite: {args.overwrite}")
    print()

    tasks = [
        (sid, dirs, args.target_radius, args.overwrite)
        for sid, dirs in scene_to_chunks.items()
    ]

    n_already_done = n_newly_saved = n_errors = 0
    error_list = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_scene, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks),
                        desc="  Scenes"):
            sid, total_saved, newly_saved, err = fut.result()
            if err:
                n_errors += 1
                error_list.append((sid, err))
            else:
                n_already_done += (total_saved - newly_saved)
                n_newly_saved  += newly_saved

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Already done (skipped):  {n_already_done} chunks")
    print(f"  Newly saved:             {n_newly_saved} chunks")
    print(f"  Errors:                  {n_errors} scenes")
    if error_list:
        print(f"  First errors:")
        for sid, err in error_list[:3]:
            print(f"    {sid}: {err}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print(f"\n  SANITY CHECK:")
    example_sid  = list(scene_to_chunks.keys())[0]
    example_dirs = scene_to_chunks[example_sid]
    print(f"  Scene: {example_sid}  ({len(example_dirs)} chunks)")
    nf_values = []
    for d in example_dirs[:4]:
        nf_path = os.path.join(d, 'norm_factor.npy')
        if os.path.exists(nf_path):
            nf = np.load(nf_path)
            nf_values.append(nf)
            print(f"    {Path(d).name}: "
                  f"center=({nf[0]:.3f},{nf[1]:.3f},{nf[2]:.3f})  "
                  f"scale={nf[3]:.4f}")
    if len(nf_values) > 1:
        all_same = all(np.allclose(nf_values[0], nf, atol=1e-5)
                       for nf in nf_values)
        print(f"\n  All chunks share same norm_factor: {all_same}  ← must be True")
        if all_same:
            print(f"  ✓ PRECOMPUTATION SUCCESSFUL")
        else:
            print(f"  ✗ Something went wrong — values differ across chunks")

    print(f"\n{'='*60}")
    print(f"  NEXT STEPS")
    print(f"{'='*60}")
    print(f"  1. Verify: python precompute_norm_from_chunks.py \\")
    print(f"       --chunks_dir {chunks_dir} --verify_only")
    print(f"")
    print(f"  2. Update gs_dataset_scenesplat.py:")
    print(f"     In __getitem__, replace:")
    print(f"       coord, scale = normalize_to_canonical_sphere(coord, scale, ...)")
    print(f"     With:")
    print(f"       coord, scale = normalize_with_norm_factor(coord, scale, scene_dir, ...)")
    print(f"")
    print(f"  3. Re-run diagnostic to confirm pos_intra > pos_inter")
    print(f"     (was 0.277 vs 0.284 before fix)")
    print(f"")
    print(f"  4. Train on 3800 chunks — position should now converge")


if __name__ == '__main__':
    main()