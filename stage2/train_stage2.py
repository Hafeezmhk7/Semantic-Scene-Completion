"""
Can3Tok Stage 2 Training
========================
Unified training script for all three Stage 2 models.

Changes vs original version:
  • --vis_freq flag: periodically generate scenes and save PLY during training
  • --flow_diag_freq flag: print per-timestep flow matching diagnostics
  • Richer per-epoch stats: t_mean, v_target_std, v_pred_std, per-t-bin loss
  • B1 loss scale note printed at startup

Usage
-----
# Strategy A — Layout DiT
accelerate launch --config_file job_scripts/accelerate_config.yaml train_stage2.py \\
    --strategy A --stage layout \\
    --stage1_checkpoint /path/to/best_model.pth --model_size B

# Strategy A — Geometry DiT (cross-attention, default)
accelerate launch ... train_stage2.py \\
    --strategy A --stage geometry \\
    --stage1_checkpoint /path/to/best_model.pth \\
    --zs_conditioning cross_attn --vis_freq 50

# Strategy A — Geometry DiT (adaLN ablation)
accelerate launch ... train_stage2.py \\
    --strategy A --stage geometry \\
    --stage1_checkpoint /path/to/best_model.pth \\
    --zs_conditioning adaLN

# Strategy D — Geometry DiT
accelerate launch ... train_stage2.py \\
    --strategy D --stage geometry \\
    --stage1_checkpoint /path/to/best_model.pth \\
    --zs_conditioning cross_attn

# Strategy B1 — Completion DiT
accelerate launch ... train_stage2.py \\
    --strategy B1 --stage completion \\
    --stage1_checkpoint /path/to/best_model.pth

Flow matching diagnostics (--flow_diag_freq N)
----------------------------------------------
Every N epochs, prints:
  t_mean/std       : verify Uniform(0,1) sampling
  xt_norm_by_t     : at t~0 x_t should look like noise; at t~1 like clean data
  vtarget_mean/std : should be stable (z_clean − z_noise statistics)
  vpred_mean/std   : should converge toward vtarget statistics
  loss_by_t_bin    : loss split into t ∈ [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]
                     a healthy model decreases loss in all bins but early t-bins
                     (where x_t is noisy) are usually hardest

PLY visualization (--vis_freq N)
---------------------------------
Every N epochs, runs euler_sample on a small batch and saves PLY files.
  layout  stage : z_s generated from noise → decoded with zero z_g (colour only visible)
  geometry stage: z_g generated from real z_s (from val encoder) + euler_sample
                  → full scene decoded. Best qualitative signal.
  completion     : completion on val scenes at 40% coverage → PLY
"""

import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data as Data
from accelerate import Accelerator, DistributedDataParallelKwargs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gs_dataset_scenesplat import gs_dataset
from gs_ply_reconstructor import save_reconstructed_gaussians
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

from stage2.external.transport import create_transport
from stage2.models.layout_dit    import LayoutDiT_models
from stage2.models.geometry_dit  import (
    GeometryDiT_models, GeometryDiT_adaLN_models,
)
from stage2.models.completion_dit import (
    CompletionDiT_models, completion_training_step, sample_voxel_mask,
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ============================================================================
# Stage 1 loader
# ============================================================================

def load_stage1(checkpoint_path: str, config_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    s1 = {
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
    p.latent_disentangle        = s1["latent_disentangle"]
    p.semantic_dims             = s1["semantic_dims"]
    p.color_residual            = s1["color_residual"]
    p.decoder_fourier_pe        = s1["decoder_fourier_pe"]
    p.decoder_layout_cross_attn = s1["decoder_layout_cross_attn"]
    p.decoder_zs_cross_attn     = s1["decoder_zs_cross_attn"]
    p.structured_layout_tokens  = s1["structured_layout_tokens"]
    p.scene_layout_head         = s1["scene_layout_head"]
    p.scene_semantic_head       = s1["scene_semantic_head"]
    p.semantic_token_heads      = s1["semantic_token_heads"]
    p.semantic_mode             = "none"
    p.predict_seg_labels        = False
    p.position_scaffold         = False
    p.jepa_idea1                = False
    p.token_cond                = False
    p.decoder_pos_enc           = False
    p.decoder_layout_additive   = False

    stage1 = instantiate_from_config(model_config)
    missing, unexpected = stage1.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [Stage 1] {len(missing)} missing keys (expected)")
    if unexpected:
        print(f"  [Stage 1] {len(unexpected)} unexpected keys")

    shape_model = stage1.shape_model
    shape_model.to(device).eval()
    for param in shape_model.parameters():
        param.requires_grad_(False)

    print(f"  Stage 1 loaded: {checkpoint_path}")
    print(f"  latent_disentangle={s1['latent_disentangle']}  "
          f"semantic_dims={s1['semantic_dims']}  "
          f"color_residual={s1['color_residual']}  "
          f"decoder_zs_cross_attn={s1['decoder_zs_cross_attn']}")
    return shape_model, s1


# ============================================================================
# Encode batch  (frozen Stage 1)
# ============================================================================

@torch.no_grad()
def encode_batch(shape_model, features, strategy, s1_meta):
    B = features.shape[0]
    shape_embed, mu, log_var, z, _ = shape_model.encode(
        pc=features, feats=features, sample_posterior=False
    )
    z_clean   = mu.reshape(B, 512, 32)
    z_s_clean = z_clean[:, :16, :]
    z_g_clean = z_clean[:, 16:, :]

    z_layout = None
    if strategy == "B1" and hasattr(shape_model, "layout_projector") and \
            shape_model.layout_projector is not None:
        z_layout = shape_model.layout_projector(shape_embed)

    return z_s_clean, z_g_clean, z_clean, z_layout


# ============================================================================
# Euler sampler  (no torchdiffeq)
# ============================================================================

@torch.no_grad()
def euler_sample(model, x_init: torch.Tensor, num_steps: int = 50, **kw) -> torch.Tensor:
    model.eval()
    x, dt = x_init, 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i / num_steps, device=x.device, dtype=x.dtype)
        x = x + model(x, t, **kw) * dt
    model.train()
    return x


# ============================================================================
# PLY generation  (Stage 2 → PLY files on disk)
# ============================================================================

@torch.no_grad()
def generate_and_save_ply(
    raw_model,
    shape_model,
    val_loader,
    strategy:      str,
    stage:         str,
    zs_conditioning: str,
    save_dir:      Path,
    epoch:         int,
    device:        torch.device,
    color_residual: bool,
    num_samples:   int = 4,
    num_steps:     int = 50,
):
    """
    Generate scenes with the current DiT weights and save as PLY.

    Layout stage:
        Generate z_s from noise. Decode with z_g = zeros (only colour
        and scene-type token influences are visible). Useful for checking
        whether LayoutDiT is learning a coherent z_s distribution.

    Geometry stage:
        Take real z_s from the first val-batch encoder (not generated).
        Generate z_g from noise conditioned on real z_s.
        Assemble Z = [z_s_real | z_g_gen] and decode.
        This tests the conditional distribution P(z_g | z_s_real) directly —
        the strongest qualitative signal available during geometry training.

    Completion stage:
        Take the first val-batch, mask 40% of tokens, run CompletionDiT,
        decode both the partial input and the completed output side-by-side.
    """
    ply_dir = save_dir / "generated_gaussians" / f"epoch_{epoch:04d}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Grab one val batch
    batch = next(iter(val_loader))
    features = batch["features"].float().to(device)[:num_samples]
    B = features.shape[0]

    try:
        if stage == "layout":
            # Generate z_s from noise; decode with zero z_g
            z_s_noise = torch.randn(B, 16, 32, device=device)
            z_s_gen   = euler_sample(raw_model, z_s_noise, num_steps=num_steps)
            z_g_zero  = torch.zeros(B, 496, 32, device=device)
            Z         = torch.cat([z_s_gen, z_g_zero], dim=1)   # [B, 512, 32]
            recon, _  = shape_model.decode(Z, volume_queries=None,
                                           return_semantic_features=False, shape_embed=None)
            preds = recon.reshape(B, 40000, 14).cpu().numpy()
            mean_colors = batch["mean_color"].numpy()[:B] if color_residual else None
            if color_residual and mean_colors is not None:
                for i in range(B):
                    preds[i, :, 3:6] = np.clip(preds[i, :, 3:6] + mean_colors[i], 0, 1)
            save_reconstructed_gaussians(
                predictions=preds, output_dir=ply_dir / "layout_gen",
                epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")
            print(f"  [VIS] Layout PLY: {ply_dir/'layout_gen'}  ({B} scenes, z_g=zeros)")

        elif stage == "geometry":
            # Real z_s from encoder + generated z_g
            z_s_real, z_g_real, z_clean, _ = encode_batch(shape_model, features, strategy, {})

            z_g_noise = torch.randn(B, 496, 32, device=device)
            if zs_conditioning == "adaLN":
                z_g_gen = euler_sample(raw_model, z_g_noise, num_steps=num_steps,
                                        z_s_clean=z_s_real)
            else:
                z_g_gen = euler_sample(raw_model, z_g_noise, num_steps=num_steps,
                                        z_s_clean=z_s_real)

            Z_gen    = torch.cat([z_s_real, z_g_gen], dim=1)
            recon, _ = shape_model.decode(Z_gen, volume_queries=None,
                                           return_semantic_features=False, shape_embed=None)
            preds_gen = recon.reshape(B, 40000, 14).cpu().numpy()

            # Also decode ground truth for comparison
            recon_gt, _ = shape_model.decode(z_clean, volume_queries=None,
                                              return_semantic_features=False, shape_embed=None)
            preds_gt = recon_gt.reshape(B, 40000, 14).cpu().numpy()

            mean_colors = batch["mean_color"].numpy()[:B] if color_residual else None
            if color_residual and mean_colors is not None:
                for i in range(B):
                    preds_gen[i, :, 3:6] = np.clip(preds_gen[i, :, 3:6] + mean_colors[i], 0, 1)
                    preds_gt[i, :, 3:6]  = np.clip(preds_gt[i, :, 3:6]  + mean_colors[i], 0, 1)

            save_reconstructed_gaussians(
                predictions=preds_gen, output_dir=ply_dir / "geom_gen",
                epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")
            save_reconstructed_gaussians(
                predictions=preds_gt, output_dir=ply_dir / "geom_gt",
                epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")
            print(f"  [VIS] Geometry PLY: {ply_dir}  (gen + gt, {B} scenes, "
                  f"zs_cond={zs_conditioning})")

        elif stage == "completion":
            z_s_real, z_g_real, z_clean, z_layout = encode_batch(
                shape_model, features, "B1", {}
            )
            if z_layout is None:
                print("  [VIS] Skipped — layout_projector not available")
                return

            coverage  = 0.4
            obs_mask  = sample_voxel_mask(B, 512, device=device,
                                          coverage_range=(coverage, coverage))
            mask_exp  = obs_mask.unsqueeze(-1)
            z_noise   = torch.randn_like(z_clean)
            z_init    = z_clean * mask_exp + z_noise * (1.0 - mask_exp)

            def masked_model(x, t, **kw2):
                v = raw_model(x, t, z_layout=z_layout, obs_mask=obs_mask)
                return v * (1.0 - mask_exp)

            z_completed = euler_sample(masked_model, z_init, num_steps=num_steps)
            z_completed = z_completed * (1.0 - mask_exp) + z_clean * mask_exp

            # Partial input (masked out unobserved = zeros for vis)
            z_partial = z_clean * mask_exp

            for z_arr, name in [(z_completed, "completed"), (z_partial, "partial"),
                                 (z_clean, "gt_full")]:
                recon, _ = shape_model.decode(z_arr, volume_queries=None,
                                               return_semantic_features=False, shape_embed=None)
                preds = recon.reshape(B, 40000, 14).cpu().numpy()
                mean_colors = batch["mean_color"].numpy()[:B] if color_residual else None
                if color_residual and mean_colors is not None:
                    for i in range(B):
                        preds[i, :, 3:6] = np.clip(preds[i, :, 3:6] + mean_colors[i], 0, 1)
                save_reconstructed_gaussians(
                    predictions=preds, output_dir=ply_dir / name,
                    epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")
            print(f"  [VIS] Completion PLY: {ply_dir}  ({B} scenes, coverage={coverage:.0%})")

    except Exception as e:
        print(f"  [VIS] PLY generation failed: {e}")


# ============================================================================
# Flow matching diagnostics
# ============================================================================

def compute_flow_diagnostics(
    model,
    x_clean: torch.Tensor,
    model_kwargs: dict,
    n_bins: int = 4,
) -> dict:
    """
    Run one forward pass at several timesteps and collect diagnostic stats.
    Printed periodically to verify the velocity field is being learned.

    Returns a dict with:
      t_mean / t_std         : should be ~0.5 / ~0.29  (Uniform(0,1))
      vtarget_mean / _std    : velocity target statistics
      vpred_mean / _std      : velocity prediction statistics  → should converge to vtarget
      vpred_vtarget_cos_sim  : cosine similarity between v_pred and v_target
                               0 = orthogonal (random), 1 = perfect prediction
      loss_by_t_bin          : per-quartile MSE loss
                               all bins should decrease; early t (noisy) is hardest
    """
    B = x_clean.shape[0]
    device = x_clean.device

    t = torch.rand(B, device=device)
    x_noise = torch.randn_like(x_clean)

    # x_t = t * x_clean + (1-t) * x_noise   (ICPlan)
    t_exp    = t.view(B, *([1] * (x_clean.ndim - 1)))
    x_t      = t_exp * x_clean + (1.0 - t_exp) * x_noise
    v_target = x_clean - x_noise                            # [B, N, D]

    with torch.no_grad():
        v_pred = model(x_t, t, **model_kwargs)

    # Per-bin loss
    bins     = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_loss = {}
    for i in range(n_bins):
        mask = (t >= bins[i]) & (t < bins[i + 1])
        if mask.sum() > 0:
            err = ((v_pred[mask] - v_target[mask]) ** 2).mean().item()
            bin_loss[f"loss_t{i}"] = err

    # Cosine similarity between v_pred and v_target (flattened per sample)
    vp_flat = v_pred.reshape(B, -1)
    vt_flat = v_target.reshape(B, -1)
    cos_sim = (
        (vp_flat * vt_flat).sum(dim=1) /
        (vp_flat.norm(dim=1) * vt_flat.norm(dim=1) + 1e-8)
    ).mean().item()

    return {
        "t_mean":              t.mean().item(),
        "t_std":               t.std().item(),
        "xt_norm_mean":        x_t.norm(dim=-1).mean().item(),
        "vtarget_mean":        v_target.mean().item(),
        "vtarget_std":         v_target.std().item(),
        "vpred_mean":          v_pred.mean().item(),
        "vpred_std":           v_pred.std().item(),
        "vpred_vtarget_cosine": cos_sim,
        **bin_loss,
    }


# ============================================================================
# Model factory
# ============================================================================

def build_stage2_model(strategy, stage, size, zs_conditioning="cross_attn"):
    if stage == "layout":
        return LayoutDiT_models[f"LayoutDiT-{size}"]()
    elif stage == "geometry":
        if zs_conditioning == "adaLN":
            return GeometryDiT_adaLN_models[f"GeometryDiT_adaLN-{size}"]()
        else:
            suffix = "A" if strategy == "A" else "D"
            return GeometryDiT_models[f"GeometryDiT{suffix}-{size}"]()
    elif stage == "completion":
        assert strategy == "B1"
        return CompletionDiT_models[f"CompletionDiT-{size}"]()
    else:
        raise ValueError(f"Unknown stage '{stage}'")


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Can3Tok Stage 2 Training")

    # Required
    p.add_argument("--strategy",          type=str, required=True, choices=["A", "D", "B1"])
    p.add_argument("--stage",             type=str, required=True,
                   choices=["layout", "geometry", "completion"])
    p.add_argument("--stage1_checkpoint", type=str, required=True)

    # Model
    p.add_argument("--model_size",        type=str, default="B", choices=["S", "B", "L"])
    p.add_argument("--resume_checkpoint", type=str, default=None)

    # z_s conditioning (geometry stage only)
    p.add_argument("--zs_conditioning",   type=str, default="cross_attn",
                   choices=["cross_attn", "adaLN"],
                   help="How z_s conditions GeometryDiT. cross_attn (default): dedicated "
                        "cross-attention sublayer per block. adaLN (ablation): mean-pooled "
                        "z_s added to t_embed, no cross-attention.")

    # Training
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_epochs",    type=int,   default=500)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--warmup_steps",  type=int,   default=200)
    p.add_argument("--lr_min_ratio",  type=float, default=0.1)
    p.add_argument("--eval_every",    type=int,   default=25)

    # Dataset
    p.add_argument("--train_scenes",  type=int,  default=None)
    p.add_argument("--val_scenes",    type=int,  default=50)
    p.add_argument("--data_path",     type=str,
                   default="/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"
                           "/train_grid1.0cm_chunk8x8_stride6x6")

    # Flow matching
    p.add_argument("--path_type",     type=str, default="Linear",
                   choices=["Linear", "GVP", "VP"])
    p.add_argument("--prediction",    type=str, default="velocity",
                   choices=["velocity", "noise", "score"])

    # Visualization
    p.add_argument("--vis_freq",      type=int,  default=0,
                   help="Save generated PLY every N epochs. 0=disabled. "
                        "Geometry stage: uses real z_s from val encoder + generated z_g. "
                        "Completion stage: 40pct coverage completion. "
                        "Layout stage: generated z_s + zero z_g.")
    p.add_argument("--vis_num_scenes",type=int,  default=4,
                   help="Number of scenes to generate per visualization.")
    p.add_argument("--vis_num_steps", type=int,  default=50,
                   help="Euler steps for PLY generation (default 50).")

    # Flow matching diagnostics
    p.add_argument("--flow_diag_freq",type=int,  default=0,
                   help="Print per-timestep flow matching diagnostics every N epochs. "
                        "0=disabled. Prints cosine similarity v_pred vs v_target, "
                        "per-quartile loss, velocity std. Good sanity check.")

    # Stage 1 config
    p.add_argument("--stage1_config", type=str,
                   default="./model/configs/aligned_shape_latents/shapevae-256.yaml")

    return p.parse_args()


# ============================================================================
# LR schedule
# ============================================================================

def build_lr_lambda(warmup_steps, total_steps, lr_min_ratio):
    cosine_steps = max(total_steps - warmup_steps, 1)
    def schedule(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        t = step - warmup_steps
        return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * t / cosine_steps))
    return schedule


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.stage == "completion" and args.strategy != "B1":
        raise ValueError("--stage completion requires --strategy B1")
    if args.stage == "layout" and args.strategy == "B1":
        raise ValueError("Strategy B1 has no layout stage")
    if args.zs_conditioning == "adaLN" and args.stage != "geometry":
        print(f"  [Warning] --zs_conditioning adaLN ignored for --stage {args.stage}")

    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device      = accelerator.device

    job_id    = os.environ.get("SLURM_JOB_ID", "local")
    cond_tag  = f"_{args.zs_conditioning}" if args.stage == "geometry" else ""
    run_name  = f"stage2_{args.strategy}_{args.stage}{cond_tag}_{args.model_size}_{job_id}"
    save_path = Path(
        f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints_stage2/{run_name}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"  CAN3TOK STAGE 2 — Strategy {args.strategy} | Stage {args.stage}")
        if args.stage == "geometry":
            print(f"  z_s conditioning:  {args.zs_conditioning}")
        print(f"  Model size: {args.model_size}   Save: {save_path}")
        if args.stage == "completion":
            print(f"\n  NOTE: B1 completion loss scale will be ~25-40x larger than")
            print(f"  geometry loss because z_clean has no KL-regularised scale.")
            print(f"  Compare relative improvement within this run, not absolute value.")
        if args.vis_freq > 0:
            print(f"  PLY visualization every {args.vis_freq} epochs  "
                  f"({args.vis_num_scenes} scenes, {args.vis_num_steps} steps)")
        if args.flow_diag_freq > 0:
            print(f"  Flow diagnostics every {args.flow_diag_freq} epochs")
        print(f"{'='*70}\n")

    # Stage 1 (frozen)
    shape_model, s1_meta = load_stage1(args.stage1_checkpoint, args.stage1_config, device)

    # Stage 2 model
    model = build_stage2_model(args.strategy, args.stage, args.model_size, args.zs_conditioning)
    if accelerator.is_main_process:
        print(f"  Stage 2 model: {model}\n")

    start_epoch   = 0
    best_val_loss = float("inf")
    if args.resume_checkpoint:
        ckpt2 = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt2["model_state_dict"])
        start_epoch   = ckpt2.get("epoch", 0) + 1
        best_val_loss = ckpt2.get("val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}  val_loss={best_val_loss:.5f}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )

    color_residual = s1_meta["color_residual"]
    ds_kwargs = dict(
        root=args.data_path, resol=200,
        sampling_method="opacity",
        normalize=True, normalize_colors=True, target_radius=10.0,
        scale_norm_mode="linear", color_residual=color_residual,
        scene_layout_head=False, position_scaffold=False,
    )
    train_ds = gs_dataset(**ds_kwargs, random_permute=True,  train=True,
                          max_scenes=args.train_scenes)
    val_ds   = gs_dataset(**ds_kwargs, random_permute=False, train=True,
                          max_scenes=args.val_scenes)
    train_loader = Data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=False,
    )
    val_loader = Data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=False,
    )
    if accelerator.is_main_process:
        print(f"  Train: {len(train_ds)} scenes | Val: {len(val_ds)} scenes\n")

    bpe         = max(1, len(train_ds) // (args.batch_size * accelerator.num_processes))
    total_steps = bpe * args.num_epochs
    elapsed     = bpe * start_epoch
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(
            max(0, args.warmup_steps - elapsed),
            max(1, total_steps - elapsed),
            args.lr_min_ratio,
        ),
    )

    transport = create_transport(path_type=args.path_type, prediction=args.prediction)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    raw_model = accelerator.unwrap_model(model)

    ckpt_meta = {
        "strategy":          args.strategy,
        "stage":             args.stage,
        "zs_conditioning":   args.zs_conditioning,
        "model_size":        args.model_size,
        "path_type":         args.path_type,
        "prediction":        args.prediction,
        "stage1_checkpoint": args.stage1_checkpoint,
        "s1_latent_disentangle":        s1_meta["latent_disentangle"],
        "s1_semantic_dims":             s1_meta["semantic_dims"],
        "s1_color_residual":            s1_meta["color_residual"],
        "s1_decoder_zs_cross_attn":     s1_meta["decoder_zs_cross_attn"],
        "s1_decoder_layout_cross_attn": s1_meta["decoder_layout_cross_attn"],
    }

    print(f"Starting training — epoch {start_epoch} → {args.num_epochs - 1}\n")

    for epoch in tqdm(range(start_epoch, args.num_epochs),
                      disable=not accelerator.is_main_process):
        model.train()
        epoch_loss     = 0.0
        epoch_vtgt_std = 0.0   # track velocity target std across batches
        epoch_vpred_std = 0.0
        n_batches      = 0

        for batch in train_loader:
            features = batch["features"].float().to(device)
            B        = features.shape[0]

            optimizer.zero_grad()

            z_s_clean, z_g_clean, z_clean, z_layout = encode_batch(
                shape_model, features, args.strategy, s1_meta
            )

            if args.stage == "layout":
                terms = transport.training_losses(raw_model, z_s_clean)
                loss  = terms["loss"].mean()
                # Track velocity stats
                epoch_vtgt_std  += (z_s_clean - torch.randn_like(z_s_clean)).std().item()

            elif args.stage == "geometry":
                terms = transport.training_losses(
                    raw_model, z_g_clean,
                    model_kwargs={"z_s_clean": z_s_clean},
                )
                loss = terms["loss"].mean()

            elif args.stage == "completion":
                loss = completion_training_step(
                    raw_model, z_clean, z_layout, transport.path_sampler,
                )

            # Track predicted velocity std (proxy for model output activity)
            if "pred" in (terms if args.stage != "completion" else {}):
                epoch_vpred_std += terms["pred"].std().item()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss    = epoch_loss / max(n_batches, 1)
        lr_now      = scheduler.get_last_lr()[0]
        vtgt_std    = epoch_vtgt_std / max(n_batches, 1)

        if accelerator.is_main_process:
            print(f"Epoch {epoch:04d} | Loss={avg_loss:.5f} | LR={lr_now:.2e}", end="")
            if args.stage != "completion":
                vpred_std = epoch_vpred_std / max(n_batches, 1)
                print(f" | v_pred_std={vpred_std:.4f}", end="")
            print()

        # ── Flow matching diagnostics ────────────────────────────────────────
        if (args.flow_diag_freq > 0 and epoch % args.flow_diag_freq == 0
                and accelerator.is_main_process and args.stage != "completion"):
            model.eval()
            try:
                diag_batch   = next(iter(val_loader))
                diag_feats   = diag_batch["features"].float().to(device)
                with torch.no_grad():
                    zs, zg, zc, zl = encode_batch(shape_model, diag_feats,
                                                   args.strategy, s1_meta)
                target = zg if args.stage == "geometry" else zs
                mkw    = {"z_s_clean": zs} if args.stage == "geometry" else {}
                diag   = compute_flow_diagnostics(raw_model, target, mkw)

                print(f"  [FLOW DIAG epoch {epoch}]")
                print(f"    t:          mean={diag['t_mean']:.3f}  std={diag['t_std']:.3f}  "
                      f"(expect ~0.500 / ~0.289 for Uniform(0,1))")
                print(f"    v_target:   mean={diag['vtarget_mean']:+.4f}  "
                      f"std={diag['vtarget_std']:.4f}  "
                      f"(z_clean − z_noise; std should be ~sqrt(2)≈1.41)")
                print(f"    v_pred:     mean={diag['vpred_mean']:+.4f}  "
                      f"std={diag['vpred_std']:.4f}  "
                      f"(should converge toward v_target stats)")
                cos = diag['vpred_vtarget_cosine']
                print(f"    cosine(v_pred, v_target) = {cos:.4f}  "
                      f"(0=random, 1=perfect; should increase during training)")
                print(f"    loss by t-bin  (t: [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]):")
                for i in range(4):
                    k = f"loss_t{i}"
                    if k in diag:
                        print(f"      t_bin_{i}: {diag[k]:.5f}")
            except Exception as e:
                print(f"  [FLOW DIAG] Failed: {e}")
            model.train()

        # ── Validation ──────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
            model.eval()
            val_loss = 0.0
            n_val    = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].float().to(device)
                    zs, zg, zc, zl = encode_batch(shape_model, features,
                                                   args.strategy, s1_meta)

                    if args.stage == "layout":
                        terms    = transport.training_losses(raw_model, zs)
                        val_loss += terms["loss"].mean().item()
                    elif args.stage == "geometry":
                        terms    = transport.training_losses(
                            raw_model, zg, model_kwargs={"z_s_clean": zs})
                        val_loss += terms["loss"].mean().item()
                    elif args.stage == "completion":
                        val_loss += completion_training_step(
                            raw_model, zc, zl, transport.path_sampler).item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)

            if accelerator.is_main_process:
                print(f"  Val loss = {avg_val:.5f}")
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save(
                        {"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                         "val_loss": avg_val, **ckpt_meta},
                        save_path / "best_model.pth",
                    )
                    print(f"  [NEW BEST] saved  val_loss={best_val_loss:.5f}")

            model.train()

        # ── PLY visualization ────────────────────────────────────────────────
        if (args.vis_freq > 0 and epoch % args.vis_freq == 0
                and epoch > 0 and accelerator.is_main_process):
            model.eval()
            generate_and_save_ply(
                raw_model=raw_model,
                shape_model=shape_model,
                val_loader=val_loader,
                strategy=args.strategy,
                stage=args.stage,
                zs_conditioning=args.zs_conditioning,
                save_dir=save_path,
                epoch=epoch,
                device=device,
                color_residual=color_residual,
                num_samples=args.vis_num_scenes,
                num_steps=args.vis_num_steps,
            )
            model.train()

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(
            {"epoch": args.num_epochs - 1,
             "model_state_dict": raw_model.state_dict(),
             "best_val_loss": best_val_loss, **ckpt_meta},
            save_path / "final.pth",
        )
        print(f"\nDone. Best val loss: {best_val_loss:.5f}")
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()