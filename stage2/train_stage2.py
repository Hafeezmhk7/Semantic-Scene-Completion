"""
Can3Tok Stage 2 Training — schema-aware (supports all 7 Stage 1 experiments)
============================================================================
One script, two objectives, four stages. The latent schema is detected from the
Stage 1 checkpoint (FLAT vs SPLIT, see stage1_bridge.py); you only pick a --stage.

Objective 1 — unconditional 3D generation
  FLAT  checkpoints (exp 1-5):  --stage scene      (one DiT over Z [512,32])
  SPLIT checkpoints (exp 6-7):  --stage layout     (z_s [16,32])
                                --stage geometry   (z_g [512,32] cond z_s [16,32])

Objective 2 — scene completion (mask part of the scene, fill it in)
  SPLIT       (exp 6-7):  --stage completion       (CompletionDiT, z_s cross-attn)
  FLAT struct (exp 5):    --stage completion       (CompletionDiTUncond, self-attn)
  FLAT global (exp 1-4):  rejected (tokens are not spatial)

Notes vs the previous Stage 2:
  * --strategy / --zs_conditioning are gone (schema is auto-detected; SPLIT geometry
    always uses cross-attention to mirror the local_disentangle decoder).
  * The Latent Perceptual Loss path was removed: it was wired to the old A-schema
    (z_s as a prefix inside the 512) and would be incorrect for these checkpoints.
    It can be reintroduced for the SPLIT geometry stage by calling
    get_decoder_transformer_features(z_g_est, z_layout=z_s_clean).
  * The clean targets and decode are produced by stage1_bridge, which reads ALL the
    new flags (local_encoder / local_disentangle / token_local_decoder /
    position_scaffold / num_gaussians / embed_dim) and mirrors the Stage 1 input
    distribution, so every one of the seven checkpoints loads and trains correctly.
"""

import os
import sys
import math
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data as Data
from accelerate import Accelerator, DistributedDataParallelKwargs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gs_dataset_scenesplat as gds_module
from gs_dataset_scenesplat import gs_dataset
from gs_ply_reconstructor import save_reconstructed_gaussians

from stage2.external.transport import create_transport
from stage2.stage1_bridge import (
    load_stage1, encode_clean, decode_latent, build_stage2_model,
    validate_stage_for_schema, is_structured, stage1_data_kwargs,
)
from stage2.models.completion_dit import completion_training_step
from stage2.models.flat_dit import completion_training_step_uncond, sample_block_mask

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ============================================================================
# Euler sampler (guards non-Module closures used in completion eval)
# ============================================================================

@torch.no_grad()
def euler_sample(model, x_init, num_steps=50, **kw):
    is_module = isinstance(model, torch.nn.Module)
    if is_module:
        model.eval()
    x, dt = x_init, 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i / num_steps, device=x.device, dtype=x.dtype)
        x = x + model(x, t, **kw) * dt
    if is_module:
        model.train()
    return x


# ============================================================================
# Loss dispatch (shared by train and val)
# ============================================================================

def compute_loss(stage, schema, raw_model, transport, z_s, z_g, mean_color=None):
    """z_g is the full latent for FLAT (scene / dc / uncond completion) or the
    geometry latent for SPLIT; z_s is None for FLAT and [B,16,32] for SPLIT.
    mean_color [B,3] is only used by the dc stage."""
    if stage == "scene":
        return transport.training_losses(raw_model, z_g)["loss"].mean()
    if stage == "dc":
        # DCHead returns the Gaussian NLL of q(DC|Z) against the GT mean colour.
        return raw_model(z_g, mean_color)
    if stage == "layout":
        return transport.training_losses(raw_model, z_s)["loss"].mean()
    if stage == "geometry":
        return transport.training_losses(
            raw_model, z_g, model_kwargs={"z_s_clean": z_s})["loss"].mean()
    if stage == "completion":
        if schema == "split":
            return completion_training_step(raw_model, z_g, z_s, transport.path_sampler)
        return completion_training_step_uncond(raw_model, z_g, transport.path_sampler)
    raise ValueError(stage)


# ============================================================================
# Flow diagnostics (skipped for completion)
# ============================================================================

def compute_flow_diagnostics(model, x_clean, model_kwargs, n_bins=4):
    B, device = x_clean.shape[0], x_clean.device
    t        = torch.rand(B, device=device)
    x_noise  = torch.randn_like(x_clean)
    t_exp    = t.view(B, *([1] * (x_clean.ndim - 1)))
    x_t      = t_exp * x_clean + (1.0 - t_exp) * x_noise
    v_target = x_clean - x_noise
    with torch.no_grad():
        v_pred = model(x_t, t, **model_kwargs)
    vp = v_pred.reshape(B, -1); vt = v_target.reshape(B, -1)
    cos = ((vp * vt).sum(1) / (vp.norm(dim=1) * vt.norm(dim=1) + 1e-8)).mean().item()
    out = {"t_mean": t.mean().item(), "t_std": t.std().item(),
           "vtarget_std": v_target.std().item(), "vpred_std": v_pred.std().item(),
           "cos": cos}
    return out


# ============================================================================
# PLY saving during evaluation (qualitative monitoring)
# ============================================================================

@torch.no_grad()
def save_eval_ply(raw_model, shape_model, val_loader, flags, schema, stage,
                  save_dir, epoch, device, num_scenes=4, num_steps=50):
    ply_dir = save_dir / "recon_ply" / f"epoch_{epoch:04d}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    batch    = next(iter(val_loader))
    features = batch["features"].float().to(device)[:num_scenes]
    B        = features.shape[0]
    mc_raw   = batch.get("mean_color", None)
    mean_color = mc_raw[:B].to(device) if mc_raw is not None else None

    def _save(preds, name):
        save_reconstructed_gaussians(
            predictions=preds, output_dir=ply_dir / name, epoch=epoch,
            num_scenes=B, max_sh_degree=3, color_mode="1")

    try:
        if stage == "scene":
            z_noise = torch.randn(B, 512, 32, device=device)
            Z_gen   = euler_sample(raw_model, z_noise, num_steps=num_steps)
            _, z_full = encode_clean(shape_model, features, flags, schema)
            _save(decode_latent(shape_model, flags, Z_gen,  None, mean_color), "scene_gen")
            _save(decode_latent(shape_model, flags, z_full, None, mean_color), "scene_gt")
            print(f"  [PLY] epoch {epoch:04d}: scene_gen/ (from noise) + scene_gt/")

        elif stage == "layout":
            z_s_noise = torch.randn(B, 16, 32, device=device)
            z_s_gen   = euler_sample(raw_model, z_s_noise, num_steps=num_steps)
            z_g_zero  = torch.zeros(B, 512, 32, device=device)
            _save(decode_latent(shape_model, flags, z_g_zero, z_s_gen, mean_color), "layout_gen")
            print(f"  [PLY] epoch {epoch:04d}: layout_gen/ (z_s from noise, z_g=0)")

        elif stage == "geometry":
            z_s_real, z_g_real = encode_clean(shape_model, features, flags, schema)
            z_g_noise = torch.randn(B, 512, 32, device=device)
            z_g_gen   = euler_sample(raw_model, z_g_noise, num_steps=num_steps,
                                     z_s_clean=z_s_real)
            _save(decode_latent(shape_model, flags, z_g_gen,  z_s_real, mean_color), "geom_gen")
            _save(decode_latent(shape_model, flags, z_g_real, z_s_real, mean_color), "geom_gt")
            print(f"  [PLY] epoch {epoch:04d}: geom_gen/ (gen z_g, real z_s) + geom_gt/")

        elif stage == "completion":
            z_s_real, z_g_real = encode_clean(shape_model, features, flags, schema)
            coverage = 0.4
            obs_mask = sample_block_mask(B, 512, device, (coverage, coverage))
            mask_exp = obs_mask.unsqueeze(-1)
            z_init   = z_g_real * mask_exp + torch.randn_like(z_g_real) * (1.0 - mask_exp)

            if schema == "split":
                def masked_model(x, t, **_kw):
                    v = raw_model(x, t, z_s_real, obs_mask)
                    return v * (1.0 - mask_exp)
            else:
                def masked_model(x, t, **_kw):
                    v = raw_model(x, t, obs_mask)
                    return v * (1.0 - mask_exp)

            z_comp = euler_sample(masked_model, z_init, num_steps=num_steps)
            z_comp = z_comp * (1.0 - mask_exp) + z_g_real * mask_exp
            z_part = z_g_real * mask_exp

            z_s_for_decode = z_s_real if schema == "split" else None
            _save(decode_latent(shape_model, flags, z_comp,  z_s_for_decode, mean_color), "completed")
            _save(decode_latent(shape_model, flags, z_part,  z_s_for_decode, mean_color), "partial")
            _save(decode_latent(shape_model, flags, z_g_real, z_s_for_decode, mean_color), "gt_full")
            print(f"  [PLY] epoch {epoch:04d}: completed/ partial/ gt_full/ "
                  f"(coverage={coverage:.0%})")

    except Exception as exc:
        import traceback
        print(f"  [PLY] FAILED at epoch {epoch}: {exc}")
        traceback.print_exc()


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
# Dataset — replicate the Stage 1 combined training set
# ============================================================================

def build_train_dataset(args, ds_kwargs):
    """
    Reproduce the Stage 1 training mix so the frozen encoder is fed the same
    distribution it learned on. With --train_data chunks and --extra_train_paths,
    Stage 1 trains on ConcatDataset([interior chunks, interior full, arkitscenes
    full, scannetpp full]); this rebuilds that exactly. data_path is the BASE dir
    (e.g. .../interior_gs); the chunk root and full root are derived from it.
    """
    base       = args.data_path
    chunk_root = os.path.join(base, "train_grid1.0cm_chunk8x8_stride6x6")
    full_root  = os.path.join(base, "train")

    if args.train_data == "chunks":
        main = gs_dataset(root=chunk_root, random_permute=True, train=True,
                          max_scenes=args.train_scenes, preload=False, **ds_kwargs)
    elif args.train_data == "full":
        main = gs_dataset(root=full_root, random_permute=True, train=True,
                          max_scenes=args.train_scenes, preload=False, **ds_kwargs)
    else:  # combined
        mf = max(1, args.train_scenes // 2) if args.train_scenes else None
        mc = (args.train_scenes - mf) if args.train_scenes else None
        df = gs_dataset(root=full_root,  random_permute=True, train=True,
                        max_scenes=mf, preload=False, **ds_kwargs)
        dc = gs_dataset(root=chunk_root, random_permute=True, train=True,
                        max_scenes=mc, preload=False, **ds_kwargs)
        main = Data.ConcatDataset([df, dc])

    extras = []
    if args.extra_train_paths:
        paths  = [p.strip() for p in args.extra_train_paths.split(":") if p.strip()]
        scenes = ([s.strip() for s in args.extra_train_scenes.split(":") if s.strip()]
                  if args.extra_train_scenes else [])
        while len(scenes) < len(paths):
            scenes.append("0")
        for ep, es in zip(paths, scenes[:len(paths)]):
            ms = int(es) if es and es != "0" else None
            try:
                extras.append(gs_dataset(root=ep, random_permute=True, train=True,
                                         max_scenes=ms, disable_semantics=True,
                                         preload=False, **ds_kwargs))
            except Exception as e:
                print(f"  [warn] extra path failed: {ep}: {e}")

    return Data.ConcatDataset([main] + extras) if extras else main


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Can3Tok Stage 2 Training (schema-aware)")
    p.add_argument("--stage", type=str, required=True,
                   choices=["scene", "dc", "layout", "geometry", "completion"])
    p.add_argument("--stage1_checkpoint", type=str, required=True)
    p.add_argument("--stage1_config", type=str,
                   default="./model/configs/aligned_shape_latents/shapevae-256.yaml")
    p.add_argument("--model_size", type=str, default="B", choices=["S", "B", "L"])
    p.add_argument("--rope_type", type=str, default="learned_ape",
                   choices=["learned_ape", "1d", "3d"],
                   help="Positional encoding for scene/geometry/uncond-completion DiTs. "
                        "Ignored for the layout stage.")
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--run_tag", type=str, default="",
                   help="Optional tag prepended to the output folder name, e.g. exp1, "
                        "so different checkpoints' runs of the same stage don't collide.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--lr_min_ratio", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--train_scenes", type=int, default=None)
    p.add_argument("--val_scenes", type=int, default=50)
    p.add_argument("--data_path", type=str,
                   default="/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs",
                   help="BASE dir; the chunk root (train_grid...), full root (train/), "
                        "and val/ are derived from it, matching Stage 1.")
    p.add_argument("--train_data", type=str, default="chunks",
                   choices=["chunks", "full", "combined"])
    p.add_argument("--extra_train_paths", type=str, default="",
                   help="Colon-separated extra full-scene roots, as in Stage 1.")
    p.add_argument("--extra_train_scenes", type=str, default="",
                   help="Colon-separated max-scene counts matching --extra_train_paths.")
    p.add_argument("--path_type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    p.add_argument("--prediction", type=str, default="velocity",
                   choices=["velocity", "noise", "score"])
    p.add_argument("--vis_num_scenes", type=int, default=4)
    p.add_argument("--vis_num_steps", type=int, default=50)
    p.add_argument("--flow_diag_freq", type=int, default=0)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device      = accelerator.device

    job_id    = os.environ.get("SLURM_JOB_ID", "local")
    tag       = f"{args.run_tag}_" if args.run_tag else ""
    run_name  = f"stage2_{tag}{args.stage}_{args.model_size}_{args.rope_type}_{job_id}"
    save_path = Path(
        f"/home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion]/checkpoints_stage2/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 (frozen) ─────────────────────────────────────────────────────
    shape_model, flags, schema = load_stage1(
        args.stage1_checkpoint, args.stage1_config, device)
    structured = is_structured(flags)
    validate_stage_for_schema(args.stage, schema, structured)
    if args.stage == "dc" and not flags["color_residual"]:
        raise ValueError("--stage dc only applies when Stage 1 used color_residual "
                         "(there is no DC/AC split to recover otherwise).")

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"  CAN3TOK STAGE 2 — stage={args.stage}  schema={schema}  "
              f"structured={structured}")
        print(f"  model_size={args.model_size}  rope={args.rope_type}")
        print(f"  save: {save_path}")
        print(f"{'='*70}\n")

    model = build_stage2_model(schema, structured, args.stage, args.model_size,
                               args.rope_type, embed_dim=int(flags["embed_dim"]))
    if accelerator.is_main_process:
        print(f"  Stage 2 model: {model}\n")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch, best_val_loss = 0, float("inf")
    if args.resume_checkpoint:
        ck = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        start_epoch   = ck.get("epoch", 0) + 1
        best_val_loss = ck.get("val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}  val_loss={best_val_loss:.5f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # ── Data (replicate Stage 1 combined training mix) ───────────────────────
    gds_module.TARGET_POINTS = int(flags["num_gaussians"])   # match Stage 1 point count
    ds_kwargs = stage1_data_kwargs(flags)
    train_ds  = build_train_dataset(args, ds_kwargs)
    val_root  = os.path.join(args.data_path, "val")           # held-out full scenes (Stage 1 primary val)
    val_ds    = gs_dataset(root=val_root, random_permute=False, train=True,
                           max_scenes=args.val_scenes, **ds_kwargs)
    train_loader = Data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=8, pin_memory=True)
    val_loader   = Data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=8, pin_memory=True)
    if accelerator.is_main_process:
        print(f"  Train: {len(train_ds)} scenes | Val: {len(val_ds)} scenes\n")

    bpe         = max(1, len(train_ds) // (args.batch_size * accelerator.num_processes))
    total_steps = bpe * args.num_epochs
    elapsed     = bpe * start_epoch
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_lambda(
            max(0, args.warmup_steps - elapsed),
            max(1, total_steps - elapsed), args.lr_min_ratio))

    transport = create_transport(path_type=args.path_type, prediction=args.prediction)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)
    raw_model = accelerator.unwrap_model(model)

    ckpt_meta = {
        "stage": args.stage, "schema": schema, "structured": structured,
        "model_size": args.model_size, "rope_type": args.rope_type,
        "path_type": args.path_type, "prediction": args.prediction,
        "stage1_checkpoint": args.stage1_checkpoint,
        "num_gaussians": int(flags["num_gaussians"]),
        "embed_dim": int(flags["embed_dim"]),
        "color_residual": flags["color_residual"],
    }

    print(f"Starting training — epoch {start_epoch} → {args.num_epochs - 1}\n")

    for epoch in tqdm(range(start_epoch, args.num_epochs),
                      disable=not accelerator.is_main_process):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in train_loader:
            features = batch["features"].float().to(device)
            mean_color = batch.get("mean_color", None)
            if mean_color is not None:
                mean_color = mean_color.float().to(device)
            optimizer.zero_grad()
            z_s, z_g = encode_clean(shape_model, features, flags, schema)
            loss = compute_loss(args.stage, schema, raw_model, transport,
                                z_s, z_g, mean_color)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = scheduler.get_last_lr()[0]
        if accelerator.is_main_process:
            print(f"Epoch {epoch:04d} | Loss={avg_loss:.5f} | LR={lr_now:.2e}")

        # ── Flow diagnostics ─────────────────────────────────────────────────
        if (args.flow_diag_freq > 0 and epoch % args.flow_diag_freq == 0
                and accelerator.is_main_process and args.stage != "completion"):
            model.eval()
            try:
                db = next(iter(val_loader))
                df = db["features"].float().to(device)
                zs, zg = encode_clean(shape_model, df, flags, schema)
                target = zs if args.stage == "layout" else zg
                mkw    = {"z_s_clean": zs} if args.stage == "geometry" else {}
                d = compute_flow_diagnostics(raw_model, target, mkw)
                print(f"  [FLOW DIAG {epoch}] t={d['t_mean']:.3f}/{d['t_std']:.3f}  "
                      f"vtarget_std={d['vtarget_std']:.3f}  vpred_std={d['vpred_std']:.3f}  "
                      f"cos={d['cos']:.4f}")
            except Exception as e:
                print(f"  [FLOW DIAG] failed: {e}")
            model.train()

        # ── Validation + PLY + best ──────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
            model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].float().to(device)
                    mean_color = batch.get("mean_color", None)
                    if mean_color is not None:
                        mean_color = mean_color.float().to(device)
                    zs, zg = encode_clean(shape_model, features, flags, schema)
                    val_loss += compute_loss(args.stage, schema, raw_model,
                                             transport, zs, zg, mean_color).item()
                    n_val += 1
            avg_val = val_loss / max(n_val, 1)

            if accelerator.is_main_process:
                print(f"  Val loss = {avg_val:.5f}")
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                                "val_loss": avg_val, **ckpt_meta}, save_path / "best_model.pth")
                    print(f"  [NEW BEST] val_loss={best_val_loss:.5f}")
                if epoch > 0 and args.stage != "dc":
                    save_eval_ply(raw_model, shape_model, val_loader, flags, schema,
                                  args.stage, save_path, epoch, device,
                                  num_scenes=args.vis_num_scenes, num_steps=args.vis_num_steps)
            model.train()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save({"epoch": args.num_epochs - 1,
                    "model_state_dict": raw_model.state_dict(),
                    "best_val_loss": best_val_loss, **ckpt_meta}, save_path / "final.pth")
        print(f"\nDone. Best val loss: {best_val_loss:.5f}\nSaved to: {save_path}")


if __name__ == "__main__":
    main()