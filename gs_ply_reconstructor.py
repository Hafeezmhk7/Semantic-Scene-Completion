"""
3DGS PLY Reconstructor — adapted from supervisor's script
==========================================================

Supervisor's instruction:
  "coord, f_dc, opacity_act, scale_act, quat are directly after the VAE
   predictions. The '_act' means values are after activation function
   (sigmoid or exp). When writing back to PLY, for opacity and scale the
   script will convert them back — that is according to the 3DGS PLY
   saving conventions."

So the pipeline is exactly the supervisor's to_raw_fields():
    opacity_act  (post-sigmoid, ~[0,1])  →  logit(opacity_act)  →  PLY
    scale_act    (post-exp,     ~>0)     →  log(scale_act)      →  PLY

The PLY stores raw (pre-activation) values.
The 3DGS viewer applies sigmoid/exp when rendering.

PIPELINE MATCHES INPUT SCENE (visualize_input_scenes.py):
    raw_scales = np.log(np.maximum(scale, EPS))   # no upper cap
    raw_scales = np.log(np.maximum(scale, EPS))   # same here

No scale capping is applied. Both input and reconstruction go through
the identical pipeline so any difference seen in supersplat.at is purely
from the model, not from the PLY writing code.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from plyfile import PlyData, PlyElement

# ── constants ────────────────────────────────────────────────────────────────

C0  = 0.28209479177387814   # SH DC constant: 1 / (2√π)
EPS = 1e-7

# ── parameter slices in decoder output [N, 14] ───────────────────────────────

COORD_SLICE   = slice(0,  3)   # xyz positions
COLOR_SLICE   = slice(3,  6)   # rgb colours  (normalised [0,1] in your dataset)
OPACITY_SLICE = slice(6,  7)   # opacity_act  (post-sigmoid, target ~[0,1])
SCALE_SLICE   = slice(7,  10)  # scale_act    (post-exp,     target >0)
QUAT_SLICE    = slice(10, 14)  # quaternion   (qw, qx, qy, qz)


# ── activation inversions (supervisor's to_raw_fields) ───────────────────────

def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Inverse of sigmoid.  PLY opacity field = logit(opacity_act)."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def safe_log(s: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Inverse of exp.  PLY scale field = log(scale_act).

    Matches visualize_input_scenes.py exactly:
        raw_scales = np.log(np.maximum(scale, EPS))

    Only prevents log(0) or log(negative) from invalid values.
    No upper cap — same as input scene pipeline.
    """
    return np.log(np.maximum(s, eps)).astype(np.float32)


# ── colour conversion ─────────────────────────────────────────────────────────

def rgb_to_f_dc(rgb: np.ndarray, color_mode: str = "1") -> np.ndarray:
    """
    Convert RGB values → SH DC coefficients (f_dc).

    Dataset convention:  color stored as f_pc * 255,  where f_pc = f_dc * C0 + 0.5
    Inversion:           f_dc = (f_pc - 0.5) / C0

    color_mode:
        '1'    → already in [0,1]      ← your dataset
        '255'  → divide by 255 first
        'auto' → auto-detect
    """
    rgb = rgb.astype(np.float32, copy=False)

    if color_mode == "1":
        scale = 1.0
    elif color_mode == "255":
        scale = 255.0
    elif color_mode == "auto":
        scale = 255.0 if (rgb.dtype == np.uint8 or float(np.nanmax(rgb)) > 1.5) else 1.0
    else:
        raise ValueError(f"Unknown color_mode: {color_mode!r}")

    f_pc = np.clip(rgb / scale, 0.0, 1.0)
    return ((f_pc - 0.5) / C0).astype(np.float32)


# ── quaternion normalisation ──────────────────────────────────────────────────

def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    return (quat / norm).astype(np.float32)


# ── PLY vertex struct builder ─────────────────────────────────────────────────

def build_vertex_struct(
    coord:        np.ndarray,           # (N, 3)  xyz
    f_dc:         np.ndarray,           # (N, 3)  SH DC coefficients
    raw_opacity:  np.ndarray,           # (N,)    logit(opacity_act)
    raw_scales:   np.ndarray,           # (N, 3)  log(scale_act)
    quat:         np.ndarray,           # (N, 4)  normalised quaternion
    max_sh_degree: int = 3,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the numpy structured array for the PLY vertex element."""
    N = coord.shape[0]

    quat    = normalize_quaternion(quat.reshape(N, 4))
    normals = normals.astype(np.float32) if normals is not None \
              else np.zeros((N, 3), dtype=np.float32)

    raw_opacity = raw_opacity.reshape(N).astype(np.float32)
    raw_scales  = raw_scales.reshape(N, -1).astype(np.float32)
    f_dc        = f_dc.reshape(N, -1).astype(np.float32)

    num_f_dc   = f_dc.shape[1]        # 3
    num_scale  = raw_scales.shape[1]  # 3
    num_f_rest = 3 * ((max_sh_degree + 1) ** 2 - 1)

    dtype_list = (
        [("x","f4"), ("y","f4"), ("z","f4")]
      + [("nx","f4"), ("ny","f4"), ("nz","f4")]
      + [(f"f_dc_{i}",   "f4") for i in range(num_f_dc)]
      + [(f"f_rest_{i}", "f4") for i in range(num_f_rest)]
      + [("opacity", "f4")]
      + [(f"scale_{i}", "f4") for i in range(num_scale)]
      + [(f"rot_{i}",   "f4") for i in range(4)]
    )

    vert = np.empty(N, dtype=dtype_list)

    vert["x"],  vert["y"],  vert["z"]  = coord[:,0],   coord[:,1],   coord[:,2]
    vert["nx"], vert["ny"], vert["nz"] = normals[:,0], normals[:,1], normals[:,2]

    for i in range(num_f_dc):
        vert[f"f_dc_{i}"] = f_dc[:, i]

    # Higher-order SH = 0  (supervisor's instruction)
    for i in range(num_f_rest):
        vert[f"f_rest_{i}"] = 0.0

    vert["opacity"] = raw_opacity

    for i in range(num_scale):
        vert[f"scale_{i}"] = raw_scales[:, i]

    for i in range(4):
        vert[f"rot_{i}"] = quat[:, i]

    return vert


# ── single-scene reconstruction ───────────────────────────────────────────────

def reconstruct_single_scene(
    prediction:    np.ndarray,   # (N, 14)  one scene from decoder
    output_path:   Path,
    max_sh_degree: int = 3,
    color_mode:    str = "1",
    verbose:       bool = True,
) -> Optional[str]:
    """
    Reconstruct one 3DGS PLY from a single decoder output scene.

    Follows supervisor's script and matches visualize_input_scenes.py exactly:
        coord       → written as-is
        rgb [0,1]   → f_dc via C0 transform
        opacity_act → logit(opacity_act)              (undo sigmoid for PLY)
        scale_act   → log(maximum(scale_act, EPS))    (undo exp, no upper cap)
        quat        → normalised, written as-is
    """
    try:
        N = prediction.shape[0]

        coord       = prediction[:, COORD_SLICE  ].astype(np.float32)
        rgb         = prediction[:, COLOR_SLICE  ].astype(np.float32)
        opacity_act = prediction[:, OPACITY_SLICE].astype(np.float32)
        scale_act   = prediction[:, SCALE_SLICE  ].astype(np.float32)
        quat        = prediction[:, QUAT_SLICE   ].astype(np.float32)

        # ── convert to raw values for PLY ────────────────────────────────────
        raw_opacity = logit(opacity_act)    # undo sigmoid
        raw_scales  = safe_log(scale_act)   # undo exp, no upper cap

        # ── colour → SH DC ───────────────────────────────────────────────────
        f_dc = rgb_to_f_dc(rgb, color_mode=color_mode)

        vertex = build_vertex_struct(
            coord=coord,
            f_dc=f_dc,
            raw_opacity=raw_opacity,
            raw_scales=raw_scales,
            quat=quat,
            max_sh_degree=max_sh_degree,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output_path))

        return str(output_path)

    except Exception as e:
        print(f"  ⚠  Error: {e}")
        import traceback; traceback.print_exc()
        return None


# ── batch reconstruction (called from training loop) ─────────────────────────

def save_reconstructed_gaussians(
    predictions:   np.ndarray,   # (B, N, 14)
    output_dir:    Path,
    epoch:         int,
    num_scenes:    int = 5,
    max_sh_degree: int = 3,
    color_mode:    str = "1",
    prefix:        str = "scene",
) -> dict:
    """
    Save the first `num_scenes` decoder outputs as 3DGS PLY files
    viewable on https://supersplat.at.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_save = min(num_scenes, predictions.shape[0])
    saved  = {}

    for i in range(n_save):
        out_path = output_dir / f"{prefix}_{i:03d}_epoch_{epoch:03d}.ply"
        path = reconstruct_single_scene(
            prediction=predictions[i],
            output_path=out_path,
            max_sh_degree=max_sh_degree,
            color_mode=color_mode,
            verbose=True,
        )
        if path:
            saved[f"scene_{i:03d}"] = path

    return saved


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rng  = np.random.default_rng(42)
    B, N = 2, 1000

    # --- Test case 1: epoch 0 (random, unconstrained) ---
    pred_ep0 = np.zeros((B, N, 14), np.float32)
    pred_ep0[:, :, 0:3]   = rng.uniform(-2, 2,    (B, N, 3))
    pred_ep0[:, :, 3:6]   = rng.uniform(0,  1,    (B, N, 3))
    pred_ep0[:, :, 6]     = rng.uniform(0.1, 0.9, (B, N))
    pred_ep0[:, :, 7:10]  = rng.uniform(0.01, 1.5,(B, N, 3))
    q = rng.standard_normal((B, N, 4))
    pred_ep0[:, :, 10:14] = q / np.linalg.norm(q, axis=2, keepdims=True)

    # --- Test case 2: epoch 50 (converging) ---
    pred_ep50 = pred_ep0.copy()
    pred_ep50[:, :, 0:3]  = rng.uniform(-5, 5,    (B, N, 3))
    pred_ep50[:, :, 6]    = rng.uniform(0.86, 1.12,(B, N))
    pred_ep50[:, :, 7:10] = rng.uniform(0.01, 0.13,(B, N, 3))

    all_ok = True
    for label, preds, ep in [("Epoch 0  (random init)", pred_ep0,  0),
                              ("Epoch 50 (converging)",  pred_ep50, 50)]:
        print(f"\n{'─'*60}")
        print(f"Test: {label}")
        print(f"{'─'*60}")
        paths = save_reconstructed_gaussians(
            predictions=preds,
            output_dir=Path(f"test_output/epoch_{ep:03d}"),
            epoch=ep, num_scenes=1,
        )
        if not paths:
            print(f"❌ FAILED to generate PLY"); all_ok = False; continue

        ply   = PlyData.read(list(paths.values())[0])
        verts = ply["vertex"]
        sc_min = verts["scale_0"].min()
        sc_max = verts["scale_0"].max()
        print(f"  raw_scale range in PLY: [{sc_min:.4f}, {sc_max:.4f}]")
        print(f"  splat size range:       [{np.exp(sc_min):.4f}, {np.exp(sc_max):.4f}]m")
        print(f"  coord range x:          [{verts['x'].min():.4f}, {verts['x'].max():.4f}]m")
        print(f"  ✅ PLY written successfully (no capping applied)")

    import shutil
    shutil.rmtree("test_output", ignore_errors=True)

    print(f"\n{'='*60}")
    print("✅ ALL TESTS PASSED" if all_ok else "❌ SOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)