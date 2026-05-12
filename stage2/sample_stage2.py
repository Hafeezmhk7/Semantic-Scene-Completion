"""
Can3Tok Stage 2 — Generation and Completion Inference
======================================================
Generates full 3DGS scenes from noise (Strategy A / D) or completes
partial scans (Strategy A / D / B1).

Usage
-----
# Unconditional generation — Strategy A
python sample_stage2.py \
    --strategy A \
    --layout_checkpoint   /path/to/layout_best.pth \
    --geometry_checkpoint /path/to/geometry_best.pth \
    --stage1_checkpoint   /path/to/stage1_best.pth \
    --num_samples 4 --num_steps 50 \
    --output_dir ./generated_scenes/

# Scene completion — Strategy B1
python sample_stage2.py \
    --strategy B1 \
    --completion_checkpoint /path/to/completion_best.pth \
    --stage1_checkpoint     /path/to/stage1_best.pth \
    --partial_scan          /path/to/partial_scene_dir/ \
    --coverage              0.4 \
    --output_dir ./completed_scenes/
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_ply_reconstructor import save_reconstructed_gaussians

from stage2.external.transport import create_transport
from stage2.models.layout_dit     import LayoutDiT_models
from stage2.models.geometry_dit   import GeometryDiT_models, GeometryDiT_adaLN_models
from stage2.models.completion_dit import CompletionDiT_models, sample_voxel_mask


# ============================================================================
# Simple Euler ODE sampler (no torchdiffeq dependency)
# ============================================================================

@torch.no_grad()
def euler_sample(model, x_init: torch.Tensor, num_steps: int = 50, **model_kwargs) -> torch.Tensor:
    """
    Euler ODE sampler for flow matching.

    Integrates the learned velocity field from t=0 (noise) to t=1 (data)
    using fixed-step Euler method.

    x_init     : pure Gaussian noise, same shape as the target latent
    num_steps  : number of Euler steps (50 is sufficient for most cases;
                 more steps → higher quality but slower)
    model_kwargs: forwarded to model(x, t, **model_kwargs) at each step
    """
    model.eval()
    x  = x_init
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_scalar = i / num_steps
        t        = torch.full((x.shape[0],), t_scalar, device=x.device, dtype=x.dtype)
        v        = model(x, t, **model_kwargs)
        x        = x + v * dt
    return x


# ============================================================================
# Stage 1 loader (shared with train_stage2.py)
# ============================================================================

def load_stage1(checkpoint_path: str, config_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    s1   = {
        "latent_disentangle":        ckpt.get("latent_disentangle",        False),
        "semantic_dims":             ckpt.get("semantic_dims",             512),
        "color_residual":            ckpt.get("color_residual",            False),
        "decoder_fourier_pe":        ckpt.get("decoder_fourier_pe",        False),
        "decoder_layout_cross_attn": ckpt.get("decoder_layout_cross_attn", False),
        "decoder_zs_cross_attn":     ckpt.get("decoder_zs_cross_attn",     False),
        "structured_layout_tokens":  ckpt.get("structured_layout_tokens",  False),
        "scene_layout_head":         ckpt.get("scene_layout_head",         False),
        "scene_semantic_head":       ckpt.get("scene_semantic_head",       False),
        "semantic_token_heads":      ckpt.get("semantic_token_heads",      False),
    }

    model_config = get_config_from_file(config_path).model
    p = model_config.params.shape_module_cfg.params
    for k, v in s1.items():
        setattr(p, k, v)
    p.semantic_mode         = "none"
    p.predict_seg_labels    = False
    p.position_scaffold     = False
    p.jepa_idea1            = False
    p.token_cond            = False
    p.decoder_pos_enc       = False
    p.decoder_layout_additive = False

    stage1 = instantiate_from_config(model_config)
    stage1.load_state_dict(ckpt["model_state_dict"], strict=False)
    shape_model = stage1.shape_model
    shape_model.to(device).eval()
    for param in shape_model.parameters():
        param.requires_grad_(False)

    return shape_model, s1


def load_stage2_model(checkpoint_path: str, device: torch.device):
    """Load a Stage 2 model from a checkpoint, auto-detecting model type."""
    ckpt            = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    strategy        = ckpt["strategy"]
    stage           = ckpt["stage"]
    size            = ckpt["model_size"]
    zs_conditioning = ckpt.get("zs_conditioning", "cross_attn")   # default for old checkpoints

    if stage == "layout":
        model = LayoutDiT_models[f"LayoutDiT-{size}"]()
    elif stage == "geometry":
        if zs_conditioning == "adaLN":
            model = GeometryDiT_adaLN_models[f"GeometryDiT_adaLN-{size}"]()
        else:
            suffix = "A" if strategy == "A" else "D"
            model  = GeometryDiT_models[f"GeometryDiT{suffix}-{size}"]()
    elif stage == "completion":
        model = CompletionDiT_models[f"CompletionDiT-{size}"]()
    else:
        raise ValueError(f"Unknown stage '{stage}' in checkpoint.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, strategy, stage


# ============================================================================
# Decode Z with frozen Stage 1 decoder
# ============================================================================

@torch.no_grad()
def decode_with_stage1(shape_model, Z: torch.Tensor, color_residual: bool, mean_color=None):
    """
    Decode a latent Z [B, 512, 32] using the frozen Stage 1 VAE decoder.
    Returns reconstructed Gaussians [B, 40000, 14].
    """
    latents = Z.reshape(Z.shape[0], 512, 32)
    recon, _ = shape_model.decode(
        latents, volume_queries=None,
        return_semantic_features=False,
        shape_embed=None,
    )
    recon_3d = recon.reshape(Z.shape[0], 40000, 14)
    if color_residual and mean_color is not None:
        recon_3d[:, :, 3:6] = recon_3d[:, :, 3:6] + mean_color.unsqueeze(1)
    return recon_3d


# ============================================================================
# Generation — Strategy A and D
# ============================================================================

@torch.no_grad()
def generate_scenes(
    layout_model,
    geometry_model,
    shape_model,
    strategy:     str,
    num_samples:  int,
    num_steps:    int,
    device:       torch.device,
    color_residual: bool,
):
    """
    Generate num_samples scenes unconditionally.

    Stage 2a: sample z_s from noise using LayoutDiT
    Stage 2b: sample z_g from noise conditioned on z_s using GeometryDiTA/D
    Decode:   z_s + z_g → frozen Stage 1 decoder → 40000 Gaussians
    """
    print(f"Generating {num_samples} scenes  (strategy={strategy}, steps={num_steps})")

    # Stage 2a — generate layout tokens
    z_s_noise = torch.randn(num_samples, 16, 32, device=device)
    z_s_gen   = euler_sample(layout_model, z_s_noise, num_steps=num_steps)
    print(f"  z_s generated  [{z_s_gen.shape}]  range=[{z_s_gen.min():.2f}, {z_s_gen.max():.2f}]")

    # Stage 2b — generate geometry tokens
    z_g_noise = torch.randn(num_samples, 496, 32, device=device)
    z_g_gen   = euler_sample(geometry_model, z_g_noise, num_steps=num_steps, z_s_clean=z_s_gen)
    print(f"  z_g generated  [{z_g_gen.shape}]  range=[{z_g_gen.min():.2f}, {z_g_gen.max():.2f}]")

    # Assemble full Z and decode
    Z_gen    = torch.cat([z_s_gen, z_g_gen], dim=1)   # [B, 512, 32]
    scenes   = decode_with_stage1(shape_model, Z_gen, color_residual)
    return scenes.cpu().numpy()


# ============================================================================
# Completion — Strategy B1
# ============================================================================

@torch.no_grad()
def complete_scene(
    completion_model,
    shape_model,
    features:      torch.Tensor,
    coverage:      float,
    num_steps:     int,
    device:        torch.device,
    color_residual: bool,
):
    """
    Complete a partial scene using Strategy B1.

    1. Encode the (potentially partial) input scene → Z_encoder, z_layout
    2. Build a mask at the requested coverage level
    3. Denoise unobserved tokens conditioned on z_layout
    4. Decode the completed Z
    """
    features = features.to(device)
    B        = features.shape[0]

    with torch.no_grad():
        shape_embed, mu, _, _, _ = shape_model.encode(
            pc=features, feats=features, sample_posterior=False
        )
        z_encoder = mu.reshape(B, 512, 32)
        z_layout  = shape_model.layout_projector(shape_embed)   # [B, 16, 32]

    # Build mask
    obs_mask = sample_voxel_mask(B, 512, device=device, coverage_range=(coverage, coverage))

    # Initial noisy tokens: observed = from encoder, unobserved = noise
    z_noise   = torch.randn_like(z_encoder)
    mask_exp  = obs_mask.unsqueeze(-1)
    z_init    = z_encoder * mask_exp + z_noise * (1.0 - mask_exp)

    # Closure: fix observed tokens throughout sampling
    def masked_model(x, t, **kw):
        v = completion_model(x, t, z_layout, obs_mask)
        # Zero velocity for observed tokens — they don't move
        return v * (1.0 - mask_exp)

    z_completed = euler_sample(masked_model, z_init, num_steps=num_steps)
    # Restore observed tokens exactly
    z_completed = z_completed * (1.0 - mask_exp) + z_encoder * mask_exp

    scenes = decode_with_stage1(shape_model, z_completed, color_residual)
    return scenes.cpu().numpy()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Can3Tok Stage 2 Sampling")
    p.add_argument("--strategy",              type=str,   required=True, choices=["A", "D", "B1"])
    p.add_argument("--stage1_checkpoint",     type=str,   required=True)
    p.add_argument("--stage1_config",         type=str,
                   default="./model/configs/aligned_shape_latents/shapevae-256.yaml")
    # Generation checkpoints (A / D)
    p.add_argument("--layout_checkpoint",     type=str,   default=None)
    p.add_argument("--geometry_checkpoint",   type=str,   default=None)
    # Completion checkpoint (B1)
    p.add_argument("--completion_checkpoint", type=str,   default=None)
    # Sampling
    p.add_argument("--num_samples",           type=int,   default=4)
    p.add_argument("--num_steps",             type=int,   default=50)
    p.add_argument("--output_dir",            type=str,   default="./stage2_samples")
    # Completion-specific
    p.add_argument("--partial_scan",          type=str,   default=None)
    p.add_argument("--coverage",              type=float, default=0.4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load Stage 1 ─────────────────────────────────────────────────────────
    print("Loading Stage 1 model...")
    shape_model, s1_meta = load_stage1(args.stage1_checkpoint, args.stage1_config, device)
    color_residual = s1_meta["color_residual"]

    # ── Generation (Strategy A / D) ──────────────────────────────────────────
    if args.strategy in ("A", "D"):
        assert args.layout_checkpoint   is not None, "--layout_checkpoint required"
        assert args.geometry_checkpoint is not None, "--geometry_checkpoint required"

        print("Loading Stage 2 models...")
        layout_model,   _, _ = load_stage2_model(args.layout_checkpoint,   device)
        geometry_model, _, _ = load_stage2_model(args.geometry_checkpoint, device)

        scenes = generate_scenes(
            layout_model, geometry_model, shape_model,
            strategy=args.strategy, num_samples=args.num_samples,
            num_steps=args.num_steps, device=device, color_residual=color_residual,
        )

        print(f"Saving {len(scenes)} PLYs to {out} ...")
        save_reconstructed_gaussians(
            predictions=scenes, output_dir=out, epoch=0,
            num_scenes=len(scenes), max_sh_degree=3, color_mode="1",
        )

    # ── Completion (Strategy B1) ──────────────────────────────────────────────
    elif args.strategy == "B1":
        assert args.completion_checkpoint is not None, "--completion_checkpoint required"
        assert args.partial_scan          is not None, "--partial_scan required"

        print("Loading Stage 2 completion model...")
        completion_model, _, _ = load_stage2_model(args.completion_checkpoint, device)

        # Load partial scan features
        # Expects the same .npy structure as gs_dataset_scenesplat
        import numpy as np
        from gs_dataset_scenesplat import gs_dataset, normalize_to_canonical_sphere

        scan_dir  = Path(args.partial_scan)
        coord     = np.load(scan_dir / "coord.npy")
        color     = np.load(scan_dir / "color.npy") / 255.0
        scale     = np.load(scan_dir / "scale.npy")
        quat      = np.load(scan_dir / "quat.npy")
        opacity   = np.load(scan_dir / "opacity.npy")
        coord, scale = normalize_to_canonical_sphere(coord, scale, target_radius=10.0)

        # Pack into feature tensor [1, 40000, 14]
        N = min(len(coord), 40000)
        idx  = np.argsort(opacity)[-N:]
        feat = np.concatenate([coord[idx], color[idx], opacity[idx:idx+1] if opacity.ndim > 1 else opacity[idx, None],
                               scale[idx], quat[idx]], axis=1)   # [N, 14]
        # Pad to 40000 if needed
        if N < 40000:
            feat = np.concatenate([feat, np.tile(feat[[-1]], (40000-N, 1))], axis=0)

        # Build the full [1, 40000, 18] feature tensor (voxel_centers + point_uniq_idx + params)
        # For simplicity, place zeros for voxel_centers and voxel_id
        zeros = np.zeros((40000, 4), dtype=np.float32)
        feat_full = np.concatenate([zeros, feat], axis=1)  # [40000, 18]
        features  = torch.tensor(feat_full, dtype=torch.float32).unsqueeze(0)  # [1, 40000, 18]

        scenes = complete_scene(
            completion_model, shape_model, features,
            coverage=args.coverage, num_steps=args.num_steps,
            device=device, color_residual=color_residual,
        )

        print(f"Saving completed scene to {out} ...")
        save_reconstructed_gaussians(
            predictions=scenes, output_dir=out, epoch=0,
            num_scenes=1, max_sh_degree=3, color_mode="1",
        )

    print("Done.")


if __name__ == "__main__":
    main()