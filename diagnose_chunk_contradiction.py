"""
diagnose_chunk_contradiction.py
================================
Diagnose whether grid chunks cause contradictory supervision for the decoder.

HYPOTHESIS:
  Adjacent chunks from the same scene produce similar encoder latents z
  but require different position outputs from the decoder.
  This creates contradictory gradients that prevent position convergence.

THREE MEASUREMENTS:
  1. Intra-scene z similarity vs inter-scene z similarity
     If hypothesis correct: intra > inter (chunks from same room are more similar)

  2. Intra-scene position target similarity vs inter-scene
     If hypothesis correct: intra < inter (same-room chunks need DIFFERENT positions)

  3. Contradiction score = z_sim_intra / pos_sim_intra
     HIGH score = model receives similar inputs but needs different outputs
     This is the core contradiction metric.

USAGE:
  # Run without training — uses frozen random encoder as proxy:
  python diagnose_chunk_contradiction.py --data_path /path/to/train_grid... --mode random

  # Run with actual Stage 1 checkpoint:
  python diagnose_chunk_contradiction.py \
    --data_path /path/to/train_grid... \
    --checkpoint /path/to/best_model.pth \
    --config ./model/configs/aligned_shape_latents/shapevae-256.yaml \
    --mode checkpoint

  # Run on full scenes for comparison baseline:
  python diagnose_chunk_contradiction.py --data_path /path/to/train --mode random

EXPECTED OUTPUT (if hypothesis correct):
  ┌─────────────────────────────────────────────────────────────┐
  │  CHUNK CONTRADICTION ANALYSIS                               │
  │  Intra-scene z cosine sim:   0.82  (chunks from same room)  │
  │  Inter-scene z cosine sim:   0.31  (chunks from diff rooms) │
  │  Intra-scene pos similarity: 0.12  (DIFFERENT target pos)   │
  │  Inter-scene pos similarity: 0.18                           │
  │  Contradiction score:        6.83  ← HIGH = hypothesis TRUE │
  └─────────────────────────────────────────────────────────────┘

  A contradiction score >> 1 confirms the hypothesis.
  A score near 1 refutes it.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_scene(scene_dir, target_radius=10.0, max_gaussians=40000):
    """Load and normalise one chunk/scene."""
    try:
        coord   = np.load(os.path.join(scene_dir, 'coord.npy'))
        opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))
        color   = np.load(os.path.join(scene_dir, 'color.npy')) / 255.0
        scale   = np.load(os.path.join(scene_dir, 'scale.npy'))
        quat    = np.load(os.path.join(scene_dir, 'quat.npy'))
    except FileNotFoundError as e:
        return None, None

    # Top-N opacity sampling (same as training)
    N = len(coord)
    T = min(max_gaussians, N)
    selected = np.argsort(opacity)[-T:]
    if N < max_gaussians:
        extra    = np.full(max_gaussians - N, selected[-1])
        selected = np.concatenate([selected, extra])

    coord   = coord  [selected[:max_gaussians]]
    opacity = opacity[selected[:max_gaussians]]
    color   = color  [selected[:max_gaussians]]
    scale   = scale  [selected[:max_gaussians]]
    quat    = quat   [selected[:max_gaussians]]

    # Per-chunk normalisation (exactly as in training)
    center      = coord.mean(axis=0)
    coord_c     = coord - center
    max_dist    = np.linalg.norm(coord_c, axis=1).max()
    if max_dist < 1e-6:
        max_dist = 1.0
    scale_factor = target_radius / (max_dist * 1.1)
    coord_norm   = coord_c * scale_factor

    # Return normalised positions as targets, and raw features for encoder
    mean_color = color.mean(axis=0)
    color_res  = color - mean_color

    voxel_size   = 16.0 / 40
    origin_offset = np.array([19.5] * 3) * voxel_size
    shifted       = coord_norm + origin_offset
    voxel_idx     = np.floor(shifted / voxel_size).clip(0, 39)
    voxel_centers = (voxel_idx - 19.5) * voxel_size

    features = np.concatenate([
        voxel_centers,                      # cols 0:3
        np.zeros((max_gaussians, 1)),       # col  3  point_uniq_idx placeholder
        coord_norm,                          # cols 4:7  xyz
        color_res,                           # cols 7:10 rgb residual
        opacity[:, np.newaxis],              # col  10
        scale * scale_factor,               # cols 11:14
        quat,                                # cols 14:18
    ], axis=1).astype(np.float32)

    return (torch.from_numpy(features).unsqueeze(0),   # [1, 40000, 18]
            torch.from_numpy(coord_norm).unsqueeze(0)) # [1, 40000, 3]  target positions


def cosine_sim(a, b):
    """Cosine similarity between two flattened vectors."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def position_similarity(pos_a, pos_b):
    """
    How similar are two position fields?
    Use negative normalised L2 distance mapped to [0,1].
    0 = completely different, 1 = identical.
    """
    diff = (pos_a - pos_b).reshape(-1).float()
    dist = diff.norm().item()
    scale = pos_a.reshape(-1).float().norm().item() + pos_b.reshape(-1).float().norm().item()
    if scale < 1e-8:
        return 1.0
    # Normalised: 0=identical, 1=orthogonal/opposite
    # We return similarity = 1 - normalised_dist
    return max(0.0, 1.0 - dist / (scale + 1e-8))


# ── Group chunks by parent scene ─────────────────────────────────────────────

def group_by_scene(data_root):
    """
    Grid chunk directories look like:
      scene0001_chunk_r0_c0/
      scene0001_chunk_r0_c1/
      scene0001_chunk_r1_c0/
      ...
    Group by the scene prefix before '_chunk'.

    Falls back to grouping by first N characters if '_chunk' not in name.
    """
    all_dirs = sorted([
        d for d in Path(data_root).iterdir() if d.is_dir()
    ])

    groups = defaultdict(list)
    for d in all_dirs:
        name = d.name
        if '_chunk' in name:
            scene_id = name.split('_chunk')[0]
        elif '_grid' in name:
            scene_id = name.split('_grid')[0]
        else:
            # Try splitting on last underscore-number pattern
            # e.g. scene0001_0_0 → scene0001
            parts = name.rsplit('_', 2)
            scene_id = parts[0] if len(parts) >= 3 else name
        groups[scene_id].append(d)

    # Only keep scenes with multiple chunks
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"  Found {len(all_dirs)} total directories")
    print(f"  Grouped into {len(groups)} unique scenes")
    print(f"  Scenes with ≥2 chunks: {len(multi)}")
    if multi:
        sizes = [len(v) for v in multi.values()]
        print(f"  Chunks per scene: min={min(sizes)} mean={np.mean(sizes):.1f} max={max(sizes)}")
    return multi


# ── Encoder ──────────────────────────────────────────────────────────────────

def build_encoder(mode, checkpoint_path, config_path, device):
    if mode == 'random':
        print("  Using RANDOM encoder (no checkpoint needed)")
        print("  This tests structural similarity in feature space, not learned z")
        return None

    print(f"  Loading Stage 1 checkpoint: {checkpoint_path}")
    from model.michelangelo.utils import instantiate_from_config
    from model.michelangelo.utils.misc import get_config_from_file

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = get_config_from_file(config_path).model
    p = model_config.params.shape_module_cfg.params

    for key in ['latent_disentangle','semantic_dims','color_residual',
                'decoder_fourier_pe','scene_layout_head','scene_semantic_head',
                'semantic_token_heads','decoder_zs_cross_attn',
                'decoder_layout_cross_attn','structured_layout_tokens']:
        setattr(p, key, ckpt.get(key, False))
    p.semantic_mode      = 'none'
    p.predict_seg_labels = False
    p.position_scaffold  = False
    p.jepa_idea1         = False
    p.token_cond         = False
    p.decoder_pos_enc    = False
    p.decoder_layout_additive = False

    from model.michelangelo.utils import instantiate_from_config
    stage1 = instantiate_from_config(model_config)
    stage1.load_state_dict(ckpt['model_state_dict'], strict=False)
    stage1.shape_model.to(device).eval()
    for param in stage1.shape_model.parameters():
        param.requires_grad_(False)
    print(f"  Checkpoint loaded OK")
    return stage1.shape_model


@torch.no_grad()
def get_z(encoder, features, device):
    """Get latent z from encoder, or raw feature mean if mode=random."""
    if encoder is None:
        # Proxy: use the mean of the input features as a simple embedding
        return features.to(device).mean(dim=1)   # [1, 18]
    feats = features.to(device)
    _, mu, _, _, _ = encoder.encode(pc=feats, feats=feats, sample_posterior=False)
    return mu   # [1, D]


# ── Main diagnostic ──────────────────────────────────────────────────────────

def run_diagnostic(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"  CHUNK CONTRADICTION DIAGNOSTIC")
    print(f"  Data: {args.data_path}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    # Group chunks by parent scene
    scene_groups = group_by_scene(args.data_path)
    if not scene_groups:
        print("ERROR: No multi-chunk scenes found.")
        print("Try checking the directory naming convention.")
        print("Listing first 5 directories:")
        for d in sorted(Path(args.data_path).iterdir())[:5]:
            print(f"  {d.name}")
        return

    # Build encoder
    encoder = build_encoder(args.mode, args.checkpoint, args.config, device)

    # Sample pairs
    scene_ids    = list(scene_groups.keys())[:args.max_scenes]
    np.random.seed(42)

    intra_z_sims   = []   # z similarity: chunks from SAME scene
    intra_pos_sims = []   # pos similarity: chunks from SAME scene
    inter_z_sims   = []   # z similarity: chunks from DIFFERENT scenes
    inter_pos_sims = []   # pos similarity: chunks from DIFFERENT scenes

    grad_dot_products = []  # dot product of position gradients from same-scene chunks

    print(f"\nAnalysing {len(scene_ids)} scenes × {args.pairs_per_scene} pairs each...")

    for scene_id in tqdm(scene_ids, desc="Scenes"):
        chunks = scene_groups[scene_id]
        if len(chunks) < 2:
            continue

        # Sample intra-scene pairs (same scene, different chunk)
        for _ in range(args.pairs_per_scene):
            idx_a, idx_b = np.random.choice(len(chunks), 2, replace=False)
            feats_a, pos_a = load_scene(str(chunks[idx_a]))
            feats_b, pos_b = load_scene(str(chunks[idx_b]))
            if feats_a is None or feats_b is None:
                continue

            z_a = get_z(encoder, feats_a, device).cpu()
            z_b = get_z(encoder, feats_b, device).cpu()

            intra_z_sims.append(cosine_sim(z_a, z_b))
            intra_pos_sims.append(position_similarity(pos_a, pos_b))

            # Gradient dot product diagnostic:
            # If ∇L_A · ∇L_B < 0, gradients point in opposite directions
            # For linear output: gradient ∝ (target - prediction)
            # Use target positions as proxy for gradient direction
            g_a = pos_a.reshape(-1).float()
            g_b = pos_b.reshape(-1).float()
            g_a = g_a - g_a.mean();  g_b = g_b - g_b.mean()
            dot = F.cosine_similarity(g_a.unsqueeze(0), g_b.unsqueeze(0)).item()
            grad_dot_products.append(dot)

    # Sample inter-scene pairs (different scenes)
    n_inter = len(intra_z_sims)
    scene_pairs = []
    while len(scene_pairs) < n_inter:
        i, j = np.random.choice(len(scene_ids), 2, replace=False)
        if i != j:
            scene_pairs.append((i, j))

    print(f"Sampling {n_inter} inter-scene pairs for comparison...")
    for i, j in tqdm(scene_pairs[:n_inter], desc="Inter-scene"):
        chunks_i = scene_groups[scene_ids[i]]
        chunks_j = scene_groups[scene_ids[j]]
        chunk_i  = chunks_i[np.random.randint(len(chunks_i))]
        chunk_j  = chunks_j[np.random.randint(len(chunks_j))]
        feats_i, pos_i = load_scene(str(chunk_i))
        feats_j, pos_j = load_scene(str(chunk_j))
        if feats_i is None or feats_j is None:
            continue
        z_i = get_z(encoder, feats_i, device).cpu()
        z_j = get_z(encoder, feats_j, device).cpu()
        inter_z_sims.append(cosine_sim(z_i, z_j))
        inter_pos_sims.append(position_similarity(pos_i, pos_j))

    # ── Results ──────────────────────────────────────────────────────────────
    def stats(arr, name):
        a = np.array(arr)
        print(f"  {name:45s}  mean={a.mean():.4f}  std={a.std():.4f}  "
              f"min={a.min():.4f}  max={a.max():.4f}  n={len(a)}")
        return a.mean()

    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"\n  LATENT z SIMILARITY (cosine):")
    intra_z  = stats(intra_z_sims,   "  intra-scene (same room, diff chunk)")
    inter_z  = stats(inter_z_sims,   "  inter-scene (diff rooms)")

    print(f"\n  POSITION TARGET SIMILARITY:")
    intra_p  = stats(intra_pos_sims,  "  intra-scene (same room, diff chunk)")
    inter_p  = stats(inter_pos_sims,  "  inter-scene (diff rooms)")

    print(f"\n  GRADIENT DIRECTION (position target cosine similarity):")
    grad_dot = stats(grad_dot_products, "  intra-scene grad dot product")
    print(f"    (Negative = gradients CANCEL  |  Positive = gradients ALIGN)")

    print(f"\n{'='*65}")
    print(f"  CONTRADICTION DIAGNOSIS")
    print(f"{'='*65}")

    # Key ratio: how much more similar are same-scene z vs same-scene targets?
    # If z_intra >> z_inter AND pos_intra << pos_inter:
    #   → encoder conflates chunks from same room
    #   → decoder must produce different outputs for similar inputs
    #   → CONTRADICTION

    z_ratio   = intra_z / (inter_z + 1e-8)
    pos_ratio = intra_p / (inter_p + 1e-8)

    print(f"\n  Intra/inter z similarity ratio:       {z_ratio:.3f}")
    print(f"  (>1 means encoder conflates same-room chunks)")
    print(f"\n  Intra/inter position similarity ratio: {pos_ratio:.3f}")
    print(f"  (<1 means same-room chunks need DIFFERENT decoder outputs)")
    print(f"\n  Contradiction score = z_ratio / pos_ratio: "
          f"{z_ratio / (pos_ratio + 1e-8):.3f}")
    print(f"  (>>1 confirms hypothesis  |  ~1 refutes it)")

    print(f"\n  Mean gradient dot product: {np.mean(grad_dot_products):.4f}")
    frac_neg = np.mean(np.array(grad_dot_products) < 0)
    print(f"  Fraction of intra-scene gradient pairs that CANCEL: {frac_neg:.1%}")

    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")
    score = z_ratio / (pos_ratio + 1e-8)
    if score > 3.0:
        print(f"\n  ✗ HYPOTHESIS CONFIRMED (score={score:.2f})")
        print(f"    Chunks from same room produce similar encoder outputs")
        print(f"    but require contradictory decoder position outputs.")
        print(f"    This explains why position loss cannot converge at 3800 chunks.")
        print(f"    FIX: train on full scenes OR normalise chunks in parent frame.")
    elif score > 1.5:
        print(f"\n  ~ HYPOTHESIS PARTIALLY SUPPORTED (score={score:.2f})")
        print(f"    Some contradiction exists but may not be the dominant cause.")
    else:
        print(f"\n  ✓ HYPOTHESIS REFUTED (score={score:.2f})")
        print(f"    Same-room chunks are NOT more similar to each other than")
        print(f"    cross-room chunks. Look elsewhere for the convergence failure.")

    print(f"{'='*65}\n")

    # Save results for plotting
    results = {
        'intra_z_sims':      np.array(intra_z_sims),
        'inter_z_sims':      np.array(inter_z_sims),
        'intra_pos_sims':    np.array(intra_pos_sims),
        'inter_pos_sims':    np.array(inter_pos_sims),
        'grad_dot_products': np.array(grad_dot_products),
        'z_ratio':           z_ratio,
        'pos_ratio':         pos_ratio,
        'contradiction_score': score,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), **results)
    print(f"  Results saved: {out}")
    return results


# ── Diagnostic 2: Gradient cancellation during actual training ───────────────

GRAD_HOOK_RESULTS = {}

def install_position_gradient_hook(gs_autoencoder_module):
    """
    Hook into the GS_decoder's output_linear to measure position gradient norms
    and direction similarity across batches.

    Call this ONCE after model creation. Results accumulate in GRAD_HOOK_RESULTS.
    Print them every N batches.
    """
    decoder = gs_autoencoder_module.shape_model.GS_decoder
    last_grad = [None]

    def hook(grad):
        # grad shape: [B*40000*14] or similar — just measure position slice
        # output_linear outputs [B, 40000*14], position is first 3 of 14
        g = grad.detach().cpu().float()
        # Reshape to [B, 40000, 14], take position grad [B, 40000, 3]
        try:
            B = g.shape[0] if g.dim() > 1 else 1
            g_pos = g.reshape(B, 40000, 14)[:, :, 0:3].reshape(B, -1)
            # Mean gradient magnitude for position
            GRAD_HOOK_RESULTS['pos_grad_norm'] = g_pos.norm(dim=1).mean().item()
            # Cosine similarity between first and second item in batch
            if B >= 2:
                sim = F.cosine_similarity(
                    g_pos[0:1], g_pos[1:2]).item()
                GRAD_HOOK_RESULTS['pos_grad_cosine_01'] = sim
            if last_grad[0] is not None:
                # Cosine between consecutive batch gradients
                cur  = g_pos.mean(0)
                prev = last_grad[0]
                sim2 = F.cosine_similarity(cur.unsqueeze(0), prev.unsqueeze(0)).item()
                GRAD_HOOK_RESULTS['pos_grad_cosine_consecutive'] = sim2
            last_grad[0] = g_pos.mean(0).clone()
        except Exception:
            pass

    # Register on the output_linear weight
    decoder.output_linear.weight.register_hook(hook)
    print("[HOOK] Position gradient hook installed on GS_decoder.output_linear")


def print_gradient_diagnosis():
    """Call this every eval_every epochs from the training loop."""
    if not GRAD_HOOK_RESULTS:
        return
    print(f"\n  [GRAD DIAG]")
    print(f"    pos_grad_norm           = "
          f"{GRAD_HOOK_RESULTS.get('pos_grad_norm', 'N/A'):.4f}")
    print(f"    pos_grad_cosine (b0,b1) = "
          f"{GRAD_HOOK_RESULTS.get('pos_grad_cosine_01', 'N/A'):.4f}  "
          f"(negative = gradients CANCEL within batch)")
    print(f"    pos_grad_cosine (steps) = "
          f"{GRAD_HOOK_RESULTS.get('pos_grad_cosine_consecutive', 'N/A'):.4f}  "
          f"(negative = gradients CANCEL across batches)")


# ── Diagnostic 3: per-epoch gradient variance analysis ───────────────────────

def compute_gradient_snr(model, loss, accelerator):
    """
    Compute Signal-to-Noise Ratio of position gradients.

    SNR = ||E[g]||² / Var[g]

    Low SNR (< 0.1) means gradient noise dominates — gradients from different
    scenes cancel each other, giving the model no consistent learning signal.

    Call AFTER accelerator.backward(loss) but BEFORE optimizer.step().
    """
    decoder = accelerator.unwrap_model(model).shape_model.GS_decoder
    w = decoder.output_linear.weight

    if w.grad is None:
        return {}

    g = w.grad.detach().float()
    # output_linear: [40000*14, 1024]
    # Position outputs are rows 0:3*40000 = rows 0:120000
    g_pos = g[:120000, :]   # [120000, 1024]

    grad_mean = g_pos.mean()
    grad_var  = g_pos.var()
    snr       = (grad_mean ** 2) / (grad_var + 1e-10)

    return {
        'pos_grad_mean':     grad_mean.item(),
        'pos_grad_std':      g_pos.std().item(),
        'pos_grad_snr':      snr.item(),
        'pos_grad_norm':     g_pos.norm().item(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chunk contradiction diagnostic')
    parser.add_argument('--data_path',       type=str, required=True,
                        help='Path to grid chunk dataset directory')
    parser.add_argument('--mode',            type=str, default='random',
                        choices=['random', 'checkpoint'],
                        help='random=feature proxy, checkpoint=actual encoder z')
    parser.add_argument('--checkpoint',      type=str, default=None,
                        help='Stage 1 checkpoint (required for mode=checkpoint)')
    parser.add_argument('--config',          type=str,
                        default='./model/configs/aligned_shape_latents/shapevae-256.yaml')
    parser.add_argument('--max_scenes',      type=int, default=100,
                        help='Number of scenes to analyse')
    parser.add_argument('--pairs_per_scene', type=int, default=3,
                        help='Intra-scene pairs to sample per scene')
    parser.add_argument('--output',          type=str,
                        default='./diagnostics/chunk_contradiction.npz')
    args = parser.parse_args()

    if args.mode == 'checkpoint' and args.checkpoint is None:
        parser.error('--checkpoint required when mode=checkpoint')

    run_diagnostic(args)