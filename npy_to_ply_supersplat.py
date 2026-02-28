"""
npy_to_ply_supersplat.py
=========================
Convert InteriorGS scene directories → 3DGS .ply for SuperSplat visualization.

ALL Gaussians are written as-is. No subsampling. No normalization.
The only transforms applied are the PLY format requirements:
  - color   : uint8 [0,255] → f_dc SH coefficients  (PLY stores SH, not RGB)
  - opacity : probability [0,1] → logit              (PLY stores pre-sigmoid)
  - scale   : metres → log(metres)                   (PLY stores log-scale)
  - quat    : normalize to unit length               (safety, usually already unit)

DATASET STRUCTURE:
    train/
        0210_840153/
            coord.npy       [N, 3]  float  — Gaussian centres, metres
            color.npy       [N, 3]  uint8  — RGB [0,255]
            scale.npy       [N, 3]  float  — Gaussian radii, metres
            quat.npy        [N, 4]  float  — rotation quaternion (w,x,y,z)
            opacity.npy     [N]     float  — opacity [0,1]
        0832_840249/
            ...

USAGE:
    # Default: first 100 scenes
    python npy_to_ply_supersplat.py \\
        --dataset-dir /path/to/train \\
        --output-dir  ./ply_for_labeling

    # Quick test — 5 scenes only:
    python npy_to_ply_supersplat.py \\
        --dataset-dir /path/to/train \\
        --num-scenes 5

    # Dry run — list scenes without writing:
    python npy_to_ply_supersplat.py \\
        --dataset-dir /path/to/train \\
        --dry-run

OUTPUT:
    ply_for_labeling/
        scene_0001_0210_840153.ply
        scene_0002_0832_840249.ply
        ...
        scene_index.csv     ← fill in 'category' column after viewing in SuperSplat

WORKFLOW:
    1. sbatch convert_ply_slurm.sh
    2. tar -czf ply_for_labeling.tar.gz ply_for_labeling/
       scp user@snellius:/path/ply_for_labeling.tar.gz .
       tar -xzf ply_for_labeling.tar.gz
    3. https://supersplat.at → drag each .ply → fill scene_index.csv
    4. python scene_index_to_config.py --index ply_for_labeling/scene_index.csv
    5. sbatch run_tsne_analysis.sh
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# ── Constants ─────────────────────────────────────────────────────────────────
C0  = 0.28209479177387814   # SH degree-0 coefficient
EPS = 1e-7

REQUIRED_FILES = {'coord.npy', 'color.npy', 'scale.npy', 'quat.npy', 'opacity.npy'}

# Supervisor-provided labels — pre-filled in scene_index.csv
KNOWN_CATEGORIES = {
    '0210_840153': ('concert_hall',      'supervisor-provided'),
    '0832_840249': ('apartment',         'supervisor-provided'),
    '0839_841757': ('apartment',         'supervisor-provided'),
    '0001_839920': ('restaurant',        'supervisor-provided'),
    '0032_839877': ('restaurant',        'supervisor-provided'),
    '0011_840866': ('apartment',         'supervisor-provided'),
    '0008_840170': ('cinema',            'supervisor-provided'),
    '0009_840175': ('convenience_store', 'supervisor-provided'),
    '0026_839976': ('library',           'supervisor-provided'),
    '0027_839914': ('library',           'supervisor-provided'),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def discover_scenes(dataset_dir: Path) -> list:
    """Return sorted list of scene directories that contain all 5 .npy files."""
    candidates = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    if not candidates:
        print(f"\n[ERROR] No subdirectories in: {dataset_dir}")
        print("Expected: train/0001_839920/coord.npy  (etc.)")
        sys.exit(1)

    valid, skipped = [], []
    for d in candidates:
        present = {f.name for f in d.iterdir() if f.suffix == '.npy'}
        missing = REQUIRED_FILES - present
        (valid if not missing else skipped).append(
            d if not missing else (d.name, missing)
        )

    if skipped:
        print(f"  Note: {len(skipped)} dirs skipped (missing npy files)")
        for name, miss in skipped[:3]:
            print(f"    {name}: missing {miss}")
        if len(skipped) > 3:
            print(f"    ... and {len(skipped)-3} more")

    if not valid:
        print(f"\n[ERROR] No valid scene directories in: {dataset_dir}")
        sys.exit(1)

    return valid


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT CONVERSIONS  (PLY spec requirements only — no processing)
# ═══════════════════════════════════════════════════════════════════════════════

def color_to_f_dc(color_uint8):
    """uint8 RGB [0,255] → degree-0 SH coefficients (what PLY stores)."""
    rgb = np.clip(color_uint8.astype(np.float32) / 255.0, 0.0, 1.0)
    return ((rgb - 0.5) / C0).astype(np.float32)


def opacity_to_logit(opacity):
    """Probability [0,1] → logit  (PLY stores pre-sigmoid opacity)."""
    p = np.clip(opacity.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).astype(np.float32)


def scale_to_log(scale_metres):
    """Metres → log(metres)  (PLY stores log-scale)."""
    return np.log(np.maximum(scale_metres.astype(np.float32), EPS))


def normalize_quats(quat):
    """Unit-normalise quaternions (safety — should already be unit)."""
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    return (quat / np.where(norms > 0, norms, 1.0)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PLY BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_vertex_array(coord, f_dc, log_opacity, log_scale, quat,
                       max_sh_degree=3):
    """Assemble numpy structured array for plyfile."""
    N          = coord.shape[0]
    num_f_rest = 3 * ((max_sh_degree + 1) ** 2 - 1)   # 45 for degree 3

    dtype = (
        [("x","f4"), ("y","f4"), ("z","f4"),
         ("nx","f4"), ("ny","f4"), ("nz","f4")]
      + [(f"f_dc_{i}",   "f4") for i in range(3)]
      + [(f"f_rest_{i}", "f4") for i in range(num_f_rest)]
      + [("opacity", "f4")]
      + [(f"scale_{i}", "f4") for i in range(3)]
      + [(f"rot_{i}",   "f4") for i in range(4)]
    )

    v = np.zeros(N, dtype=dtype)
    v["x"], v["y"], v["z"] = coord[:, 0], coord[:, 1], coord[:, 2]
    # normals left as zero (standard for 3DGS)
    for i in range(3):
        v[f"f_dc_{i}"]   = f_dc[:, i]
        v[f"scale_{i}"]  = log_scale[:, i]
    # f_rest stays zero (no higher-order SH)
    v["opacity"] = log_opacity.ravel()
    for i in range(4):
        v[f"rot_{i}"] = quat[:, i]
    return v


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERT ONE SCENE DIRECTORY → PLY
# ═══════════════════════════════════════════════════════════════════════════════

def convert_scene(scene_dir: Path, output_path: Path, verbose: bool = False) -> bool:
    """Load all Gaussians → apply format conversions only → write PLY."""
    try:
        coord   = np.load(scene_dir / 'coord.npy')    # [N, 3] float, metres
        color   = np.load(scene_dir / 'color.npy')    # [N, 3] uint8
        scale   = np.load(scene_dir / 'scale.npy')    # [N, 3] float, metres
        quat    = np.load(scene_dir / 'quat.npy')     # [N, 4] float
        opacity = np.load(scene_dir / 'opacity.npy')  # [N]    float [0,1]

        N = len(coord)
        if verbose:
            print(f"    {N:,} Gaussians")

        vertex = build_vertex_array(
            coord        = coord.astype(np.float32),
            f_dc         = color_to_f_dc(color),
            log_opacity  = opacity_to_logit(opacity),
            log_scale    = scale_to_log(scale),
            quat         = normalize_quats(quat),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(
            str(output_path)
        )
        return True

    except FileNotFoundError as e:
        print(f"\n  [WARN] {scene_dir.name}: missing file — {e}")
        return False
    except Exception as e:
        print(f"\n  [WARN] {scene_dir.name}: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE INDEX CSV
# ═══════════════════════════════════════════════════════════════════════════════

def write_scene_index(output_dir: Path, records: list) -> Path:
    """Write annotation worksheet — open in Excel, fill 'category' column."""
    csv_path = output_dir / 'scene_index.csv'
    with open(str(csv_path), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene_number', 'scene_id', 'n_gaussians',
                         'ply_file', 'category', 'notes'])
        writer.writerow([
            '# Categories:',
            'restaurant | apartment | cinema | library | concert_hall | '
            'convenience_store | go_kart | office | gym | hotel | museum | '
            'school | hospital | wedding_hall | unknown',
            '', '', '', '',
        ])
        for r in records:
            writer.writerow([r['scene_number'], r['scene_id'],
                             r.get('n_gaussians', ''), r.get('ply_file', ''),
                             r.get('category', ''),    r.get('notes', '')])
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description='InteriorGS .npy scenes → 3DGS PLY for SuperSplat (all Gaussians)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--dataset-dir',   type=Path, required=True,
                    help='Train dir — each subdir is a scene with coord.npy etc.')
    ap.add_argument('--output-dir',    type=Path, default=Path('./ply_for_labeling'))
    ap.add_argument('--num-scenes',    type=int,  default=100,
                    help='How many scenes to convert (default: 100)')
    ap.add_argument('--dry-run',       action='store_true',
                    help='Print plan without writing any files')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip scenes whose PLY already exists')
    ap.add_argument('--verbose',       action='store_true',
                    help='Print per-scene Gaussian counts')
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"[ERROR] Not found: {args.dataset_dir}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"NPY → PLY  (all Gaussians, no subsampling, no normalization)")
    print(f"{'='*65}")
    print(f"  Dataset:    {args.dataset_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Dry run:    {args.dry_run}\n")

    print("Scanning scene directories...")
    all_scenes = discover_scenes(args.dataset_dir)
    n_total    = len(all_scenes)
    print(f"  Found {n_total} valid scenes\n")

    n_scenes = min(args.num_scenes, n_total)
    scenes   = all_scenes[:n_scenes]

    print(f"{'─'*65}")
    print(f"  Converting:   {n_scenes} of {n_total} scenes")
    print(f"  Gaussians:    ALL (no subsampling)")
    print(f"  Size/scene:   depends on scene — typically 100 MB–1 GB")
    print(f"{'─'*65}\n")

    if args.dry_run:
        print("[DRY RUN] — no files written\n")
        for i, d in enumerate(scenes[:20], 1):
            cat = KNOWN_CATEGORIES.get(d.name, ('',))[0]
            print(f"  [{i:3d}/{n_scenes}]  {d.name}  →  scene_{i:04d}_{d.name}.ply"
                  + (f"  [{cat}]" if cat else ""))
        if n_scenes > 20:
            print(f"  ... and {n_scenes - 20} more")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records    = []
    n_ok       = 0
    n_failed   = 0
    total_mb   = 0.0

    for idx, scene_dir in enumerate(scenes, start=1):
        scene_id = scene_dir.name
        ply_name = f"scene_{idx:04d}_{scene_id}.ply"
        out_ply  = args.output_dir / ply_name

        if args.skip_existing and out_ply.exists():
            cat, note = KNOWN_CATEGORIES.get(scene_id, ('', ''))
            print(f"  [{idx:3d}/{n_scenes}]  {scene_id:<20}  SKIP (exists)")
            records.append({'scene_number': idx, 'scene_id': scene_id,
                            'ply_file': ply_name, 'category': cat, 'notes': note})
            n_ok += 1
            continue

        # Quick peek at Gaussian count for the CSV (memory-mapped, no full load)
        try:
            n_gs = np.load(scene_dir / 'coord.npy', mmap_mode='r').shape[0]
        except Exception:
            n_gs = -1

        ok = convert_scene(scene_dir, out_ply, verbose=args.verbose)

        if ok:
            n_ok    += 1
            mb       = out_ply.stat().st_size / 1024 / 1024
            total_mb += mb
            status   = f"✓  {mb:.0f} MB  ({n_gs:,} Gaussians)"
        else:
            n_failed += 1
            status   = "✗  FAILED"

        cat, note = KNOWN_CATEGORIES.get(scene_id, ('', ''))
        print(f"  [{idx:3d}/{n_scenes}]  {scene_id:<20}  {status}"
              + (f"  [{cat}]" if cat else ""))

        records.append({'scene_number': idx, 'scene_id': scene_id,
                        'n_gaussians':  n_gs,
                        'ply_file':     ply_name if ok else '',
                        'category':     cat,
                        'notes':        note})

    csv_path = write_scene_index(args.output_dir, records)

    print(f"\n{'='*65}")
    print(f"DONE")
    print(f"  PLY files:  {n_ok} written" + (f", {n_failed} failed" if n_failed else ""))
    print(f"  Disk used:  {total_mb:.0f} MB  ({total_mb/1024:.1f} GB)")
    print(f"  Index:      {csv_path}")
    print(f"\nNext steps:")
    print(f"  tar -czf ply_for_labeling.tar.gz {args.output_dir}/")
    print(f"  scp user@snellius:/path/ply_for_labeling.tar.gz .")
    print(f"  # open supersplat.at, drag each .ply, fill scene_index.csv")
    print(f"  python scene_index_to_config.py \\")
    print(f"      --index {args.output_dir}/scene_index.csv \\")
    print(f"      --output scene_config.json")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()