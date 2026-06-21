import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import glob

# ── CONFIG ───────────────────────────────────────────────────────────────────
# We auto-select the largest scene in the val folder for a good visualisation.
# Override SCENE_DIR manually if you want a specific room.
VAL_ROOT   = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/test/"
SCENE_DIR  = None          # None = auto-pick largest scene; or set e.g. "0101_839973"
TOP_K      = 80000
OUT_PATH   = "hilbert_ordering.png"
FRAME      = 10.0          # canonical radius — must match training config
GRID_BITS  = 10            # 2^10 = 1024 grid — must match training config

# ── IMPORT FROM YOUR PROJECT ──────────────────────────────────────────────────
sys.path.insert(0, "/gpfs/work1/0/prjs2084/Hafeez_thesis/Semantic-Scene-Completion")
from gs_dataset_scenesplat import space_filling_sort_indices

# ── AUTO-SELECT LARGEST SCENE ─────────────────────────────────────────────────
if SCENE_DIR is None:
    scene_dirs = sorted(glob.glob(os.path.join(VAL_ROOT, "*/")))
    if not scene_dirs:
        raise FileNotFoundError(f"No scene dirs found under {VAL_ROOT}")
    # Pick the scene whose coord.npy has the most Gaussians (widest spatial extent)
    best_dir, best_n = None, 0
    for d in scene_dirs[:30]:   # check first 30 to avoid long scan
        cp = os.path.join(d, "coord.npy")
        if not os.path.exists(cp):
            continue
        c = np.load(cp)
        # Prefer scenes with large spatial footprint (max extent across x-z plane)
        extent = float(c[:, 0].max() - c[:, 0].min() +
                       c[:, 2].max() - c[:, 2].min())
        if len(c) > best_n and extent > 3.0:
            best_n   = len(c)
            best_dir = d
    if best_dir is None:
        best_dir = scene_dirs[0]
    SCENE_DIR = best_dir
    print(f"Auto-selected scene: {os.path.basename(SCENE_DIR.rstrip('/'))}")
else:
    SCENE_DIR = os.path.join(VAL_ROOT, SCENE_DIR.rstrip("/"), "")
    print(f"Using scene: {os.path.basename(SCENE_DIR.rstrip('/'))}")

# ── LOAD & NORMALISE ──────────────────────────────────────────────────────────
coord   = np.load(os.path.join(SCENE_DIR, "coord.npy"))
opacity = np.load(os.path.join(SCENE_DIR, "opacity.npy")).squeeze()

print(f"  Total Gaussians : {len(coord):,}")
print(f"  Spatial extent  : "
      f"x=[{coord[:,0].min():.2f}, {coord[:,0].max():.2f}]  "
      f"z=[{coord[:,2].min():.2f}, {coord[:,2].max():.2f}]")

# Normalise using norm_factor if available (same as training pipeline)
nf_path = os.path.join(SCENE_DIR, "norm_factor.npy")
if os.path.exists(nf_path):
    nf           = np.load(nf_path)
    center       = nf[:3]
    scale_factor = float(nf[3])
    coord        = (coord - center) * scale_factor
    print(f"  Normalised with norm_factor  (scale={scale_factor:.4f})")
else:
    center       = coord.mean(axis=0)
    max_dist     = np.linalg.norm(coord - center, axis=1).max()
    scale_factor = FRAME / (max_dist * 1.1)
    coord        = (coord - center) * scale_factor
    print(f"  Normalised per-scene  (scale={scale_factor:.4f})")

print(f"  After norm extent: "
      f"x=[{coord[:,0].min():.2f}, {coord[:,0].max():.2f}]  "
      f"z=[{coord[:,2].min():.2f}, {coord[:,2].max():.2f}]")

# ── TOP-K BY OPACITY ──────────────────────────────────────────────────────────
top_idx    = np.argsort(-opacity)[:TOP_K]
coord_topk = coord[top_idx]
print(f"  Selected top-{TOP_K:,} by opacity")

# ── HILBERT SORT ──────────────────────────────────────────────────────────────
sort_idx = space_filling_sort_indices(
    coord_topk,
    curve="hilbert",
    bits=GRID_BITS,
    frame_radius=FRAME
)
hilbert_order = sort_idx

# ── MEAN CONSECUTIVE DISTANCE ─────────────────────────────────────────────────
def mean_consec_dist(c):
    return float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1)))

d_opacity = mean_consec_dist(coord_topk)
d_hilbert = mean_consec_dist(coord_topk[hilbert_order])

print(f"\n  Mean consecutive-slot distance:")
print(f"    Opacity order : {d_opacity:.4f} m")
print(f"    Hilbert order : {d_hilbert:.4f} m")
print(f"    Locality gain : {d_opacity / max(d_hilbert, 1e-8):.2f}x  (expect > 1.0)")

# ── COLOUR MAPPING ────────────────────────────────────────────────────────────
colours_opacity = cm.plasma(np.linspace(0, 1, TOP_K))

slot_of                  = np.empty(TOP_K, dtype=np.int64)
slot_of[hilbert_order]   = np.arange(TOP_K)
colours_hilbert          = cm.plasma(slot_of / (TOP_K - 1))

# ── PLOT ──────────────────────────────────────────────────────────────────────
# Use constrained_layout instead of tight_layout for better colorbar spacing
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5.5),
    constrained_layout=True
)
fig.patch.set_facecolor("white")
s = 4   # slightly larger dots for better visibility

# Determine common axis limits so both panels are directly comparable
xmin = coord_topk[:, 0].min() - 0.3
xmax = coord_topk[:, 0].max() + 0.3
zmin = coord_topk[:, 2].min() - 0.3
zmax = coord_topk[:, 2].max() + 0.3

# ── Left panel: opacity order ─────────────────────────────────────────────────
ax1.scatter(coord_topk[:, 0], coord_topk[:, 2],
            c=colours_opacity, s=s, rasterized=True)
ax1.set_title("Opacity-ranked order", fontsize=13, fontweight="bold", pad=8)
ax1.set_xlabel("x  (m)", fontsize=11)
ax1.set_ylabel("z  (m)", fontsize=11)
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(zmin, zmax)
ax1.set_aspect("equal", adjustable="box")
ax1.annotate(
    f"mean consec. dist = {d_opacity:.2f} m",
    xy=(0.04, 0.05), xycoords="axes fraction",
    fontsize=9.5, color="#b91c1c",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dc2626",
              lw=1.3, alpha=0.92)
)

# ── Right panel: Hilbert order ────────────────────────────────────────────────
ax2.scatter(coord_topk[:, 0], coord_topk[:, 2],
            c=colours_hilbert, s=s, rasterized=True)
ax2.set_title(f"Hilbert-ordered  (fixed ±{FRAME:.0f} m frame)",
              fontsize=13, fontweight="bold", pad=8)
ax2.set_xlabel("x  (m)", fontsize=11)
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(zmin, zmax)
ax2.set_aspect("equal", adjustable="box")
ax2.annotate(
    f"mean consec. dist = {d_hilbert:.2f} m",
    xy=(0.04, 0.05), xycoords="axes fraction",
    fontsize=9.5, color="#15803d",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#16a34a",
              lw=1.3, alpha=0.92)
)

# ── Colorbar — placed to the right of ax2 with generous padding ───────────────
sm = plt.cm.ScalarMappable(
    cmap="plasma", norm=plt.Normalize(vmin=0, vmax=TOP_K - 1))
sm.set_array([])
# shrink + pad give the colorbar room so it does not clip the right panel title
cbar = fig.colorbar(sm, ax=ax2, orientation="vertical",
                    shrink=0.85, pad=0.06, aspect=30)
cbar.set_label("Slot index\n(0 = purple,  N−1 = yellow)",
               fontsize=9.5, labelpad=6)
cbar.ax.tick_params(labelsize=8)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"\n  Saved → {OUT_PATH}")
print(f"\n  What to look for:")
print(f"    Left  : colours should be spatially RANDOM (mixed purple/yellow)")
print(f"    Right : colours should form a SMOOTH GRADIENT across the room")
print(f"    The bigger the locality gain, the clearer the contrast.")