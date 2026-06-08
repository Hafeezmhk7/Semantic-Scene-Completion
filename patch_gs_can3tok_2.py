"""
Applies the two crop_percentile changes to gs_can3tok_2.py.
Run from the Can3Tok directory:
    python patch_training_script.py
"""
import re, sys

path = "gs_can3tok_2.py"
with open(path) as f:
    src = f.read()

# ── PATCH 1: add --crop_percentile argument after --random_subset_seed ────────
OLD1 = """parser.add_argument('--random_subset_seed', type=int, default=None,
    help='Random seed for selecting a subset of scenes. None = sorted first-N '
         '(default). Set to any int (e.g. 42) to randomly select train_scenes '
         'from the full directory. Only affects training data, not validation.')"""

NEW1 = OLD1 + """
# ── Spatial crop ──────────────────────────────────────────────────────────────
# When crop_percentile < 100, the dataset keeps only the inner crop_percentile%
# of Gaussians by distance from the scene centroid (in normalized space) BEFORE
# opacity sampling. This tightens the positional prediction range and improves
# convergence on full scenes.
#
# Why it helps: opacity-based sampling from a full room picks mostly wall/floor
# Gaussians spread across the full [-1,1]³ normalised volume. Cropping to the
# central 50% retains furniture and objects near the room interior, spanning
# roughly [-0.5,0.5]³ — an 8× reduction in effective volume for position targets.
#
# Recommended value: 50.0 (keep inner 50%) for full-scene runs.
# For chunk runs this is less critical (chunks are already spatially bounded).
# 100.0 = disabled (default, backward compatible).
parser.add_argument('--crop_percentile', type=float, default=100.0,
    help='Spatial crop: keep inner crop_percentile%% of Gaussians by distance '
         'from centroid before opacity sampling. Applied after normalization. '
         '100.0 = disabled (default). 50.0 = keep inner half of each scene. '
         'Reduces positional prediction range → improves position convergence '
         'on full-scene training.')"""

# ── PATCH 2: add crop_percentile to _ds_kwargs ────────────────────────────────
OLD2 = """_ds_kwargs = dict(
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
    scene_layout_head=args.scene_layout_head,
    jepa_idea1=args.jepa_idea1,
    position_layout_residual=args.position_layout_residual,
)"""

NEW2 = """_ds_kwargs = dict(
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
    scene_layout_head=args.scene_layout_head,
    jepa_idea1=args.jepa_idea1,
    position_layout_residual=args.position_layout_residual,
    crop_percentile=args.crop_percentile,   # spatial crop before opacity sampling
)"""

if OLD1 not in src:
    print("ERROR: PATCH 1 anchor not found — has gs_can3tok_2.py been modified?")
    sys.exit(1)
if OLD2 not in src:
    print("ERROR: PATCH 2 anchor not found — has _ds_kwargs been modified?")
    sys.exit(1)

src = src.replace(OLD1, NEW1)
src = src.replace(OLD2, NEW2)

with open(path, 'w') as f:
    f.write(src)

print("gs_can3tok_2.py patched successfully.")
print("  PATCH 1: --crop_percentile argument added after --random_subset_seed")
print("  PATCH 2: crop_percentile added to _ds_kwargs")