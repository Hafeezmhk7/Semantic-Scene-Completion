"""
Can3Tok Stage 2 Training
========================
Unified training script for all three Stage 2 models.

Usage
-----
# Strategy A — Stage 2a (Layout DiT, shared with D)
accelerate launch --config_file job_scripts/accelerate_config.yaml train_stage2.py \
    --strategy A --stage layout \
    --stage1_checkpoint /path/to/best_model.pth \
    --model_size B

# Strategy A — Stage 2b (Geometry DiT, prefix conditioning)
accelerate launch ... train_stage2.py \
    --strategy A --stage geometry \
    --stage1_checkpoint /path/to/best_model.pth \
    --layout_checkpoint /path/to/layout_dit.pth   # optional warm-start

# Strategy D — Stage 2b (Geometry DiT, cross-attention conditioning)
accelerate launch ... train_stage2.py \
    --strategy D --stage geometry \
    --stage1_checkpoint /path/to/best_model.pth

# Strategy B1 — Completion DiT
accelerate launch ... train_stage2.py \
    --strategy B1 --stage completion \
    --stage1_checkpoint /path/to/best_model.pth

What happens in each case
--------------------------
layout   : trains LayoutDiT on z_s tokens [B, 16, 32]
           used by both Strategy A and D — train once, share checkpoints
geometry : trains GeometryDiTA (Strategy A, prefix) or GeometryDiTD (Strategy D,
           cross-attn) on z_g tokens [B, 496, 32] conditioned on z_s
completion: trains CompletionDiT (Strategy B1) on masked Z [B, 512, 32]
            conditioned on z_layout [B, 16, 32]

Stage 1 model
-------------
The Stage 1 checkpoint is loaded in FROZEN eval mode.
Flags (latent_disentangle, semantic_dims, etc.) are read from checkpoint
metadata so they do not need to be re-specified on the command line.
"""

import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data as Data
from accelerate import Accelerator, DistributedDataParallelKwargs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # Can3Tok root

from gs_dataset_scenesplat import gs_dataset
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

from stage2.external.transport import create_transport
from stage2.models.layout_dit    import LayoutDiT_models
from stage2.models.geometry_dit  import GeometryDiT_models
from stage2.models.completion_dit import (
    CompletionDiT_models, completion_training_step,
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ============================================================================
# Stage 1 loader
# ============================================================================

def load_stage1(checkpoint_path: str, config_path: str, device: torch.device):
    """
    Load the frozen Stage 1 model (AlignedShapeAsLatentPLModule).

    Flags are restored from checkpoint metadata — no need to re-specify them.
    All parameters are frozen (requires_grad=False).

    Returns
    -------
    shape_model : AlignedShapeLatentPerceiver   (the inner model)
    s1_meta     : dict  checkpoint metadata
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── Read flags from checkpoint ────────────────────────────────────────
    s1 = {
        "latent_disentangle":       ckpt.get("latent_disentangle",       False),
        "semantic_dims":            ckpt.get("semantic_dims",            512),
        "color_residual":           ckpt.get("color_residual",           False),
        "decoder_fourier_pe":       ckpt.get("decoder_fourier_pe",       False),
        "decoder_layout_cross_attn":ckpt.get("decoder_layout_cross_attn",False),
        "decoder_zs_cross_attn":    ckpt.get("decoder_zs_cross_attn",    False),
        "structured_layout_tokens": ckpt.get("structured_layout_tokens", False),
        "scene_layout_head":        ckpt.get("scene_layout_head",        False),
        "scene_semantic_head":      ckpt.get("scene_semantic_head",      False),
        "semantic_token_heads":     ckpt.get("semantic_token_heads",     False),
    }

    # ── Build Stage 1 model with minimum required flags ───────────────────
    model_config = get_config_from_file(config_path).model
    p = model_config.params.shape_module_cfg.params

    p.latent_disentangle       = s1["latent_disentangle"]
    p.semantic_dims            = s1["semantic_dims"]
    p.color_residual           = s1["color_residual"]
    p.decoder_fourier_pe       = s1["decoder_fourier_pe"]
    p.decoder_layout_cross_attn = s1["decoder_layout_cross_attn"]
    p.decoder_zs_cross_attn    = s1["decoder_zs_cross_attn"]
    p.structured_layout_tokens = s1["structured_layout_tokens"]
    p.scene_layout_head        = s1["scene_layout_head"]
    p.scene_semantic_head      = s1["scene_semantic_head"]
    p.semantic_token_heads     = s1["semantic_token_heads"]
    # Disable all inference-only modules not needed for encoding
    p.semantic_mode            = "none"
    p.predict_seg_labels       = False
    p.position_scaffold        = False
    p.jepa_idea1               = False
    p.token_cond               = False
    p.decoder_pos_enc          = False
    p.decoder_layout_additive  = False

    stage1 = instantiate_from_config(model_config)
    missing, unexpected = stage1.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [Stage 1] {len(missing)} missing keys (expected — Stage 2 doesn't need all heads)")
    if unexpected:
        print(f"  [Stage 1] {len(unexpected)} unexpected keys")

    shape_model = stage1.shape_model
    shape_model.to(device)
    shape_model.eval()
    for p in shape_model.parameters():
        p.requires_grad_(False)

    print(f"  Stage 1 loaded: {checkpoint_path}")
    print(f"  latent_disentangle={s1['latent_disentangle']}  "
          f"semantic_dims={s1['semantic_dims']}  "
          f"decoder_layout_cross_attn={s1['decoder_layout_cross_attn']}")
    return shape_model, s1


# ============================================================================
# Encode batch with frozen Stage 1
# ============================================================================

@torch.no_grad()
def encode_batch(shape_model, features, strategy, s1_meta):
    """
    Encode a batch of 3DGS features with the frozen Stage 1 model.

    Returns
    -------
    z_s_clean : [B, 16, 32] or None   — semantic tokens (mode, no sampling noise)
    z_g_clean : [B, 496, 32] or None  — geometry tokens
    z_clean   : [B, 512, 32]          — full latent Z (mode)
    z_layout  : [B, 16, 32] or None   — from Layout16Projector (B1 only)
    """
    B = features.shape[0]

    # Use mode (mu) as clean target — no sampling noise in Stage 2 training targets
    shape_embed, mu, log_var, z, _ = shape_model.encode(
        pc=features, feats=features, sample_posterior=False
    )

    # mu is [B, 16384], reshape to [B, 512, 32]
    z_clean   = mu.reshape(B, 512, 32)
    z_s_clean = z_clean[:, :16,  :]   # [B, 16, 32]
    z_g_clean = z_clean[:, 16:, :]    # [B, 496, 32]

    z_layout = None
    if strategy == "B1" and shape_model.layout_projector is not None:
        z_layout = shape_model.layout_projector(shape_embed)   # [B, 16, 32]

    return z_s_clean, z_g_clean, z_clean, z_layout


# ============================================================================
# Model factory
# ============================================================================

def build_stage2_model(strategy: str, stage: str, size: str):
    """Instantiate the correct Stage 2 model."""
    if stage == "layout":
        key = f"LayoutDiT-{size}"
        assert key in LayoutDiT_models, f"Unknown model '{key}'"
        model = LayoutDiT_models[key]()

    elif stage == "geometry":
        suffix = "A" if strategy == "A" else "D"
        key    = f"GeometryDiT{suffix}-{size}"
        assert key in GeometryDiT_models, f"Unknown model '{key}'"
        model  = GeometryDiT_models[key]()

    elif stage == "completion":
        assert strategy == "B1", "--stage completion requires --strategy B1"
        key   = f"CompletionDiT-{size}"
        assert key in CompletionDiT_models, f"Unknown model '{key}'"
        model = CompletionDiT_models[key]()

    else:
        raise ValueError(f"Unknown --stage '{stage}'")

    return model


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Can3Tok Stage 2 Training")

    # ── Required ─────────────────────────────────────────────────────────────
    p.add_argument("--strategy",          type=str, required=True, choices=["A", "D", "B1"])
    p.add_argument("--stage",             type=str, required=True, choices=["layout", "geometry", "completion"])
    p.add_argument("--stage1_checkpoint", type=str, required=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_size",        type=str, default="B", choices=["S", "B", "L"])
    p.add_argument("--resume_checkpoint", type=str, default=None)

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--num_epochs",  type=int,   default=500)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--warmup_steps",type=int,   default=200)
    p.add_argument("--lr_min_ratio",type=float, default=0.1)
    p.add_argument("--eval_every",  type=int,   default=25)

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--train_scenes", type=int,  default=None)
    p.add_argument("--val_scenes",   type=int,  default=50)
    p.add_argument("--data_path",    type=str,
                   default="/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"
                           "/train_grid1.0cm_chunk8x8_stride6x6")

    # ── Flow matching ─────────────────────────────────────────────────────────
    p.add_argument("--path_type",   type=str,  default="Linear", choices=["Linear", "GVP", "VP"])
    p.add_argument("--prediction",  type=str,  default="velocity", choices=["velocity", "noise", "score"])

    # ── Stage 1 config ────────────────────────────────────────────────────────
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
    args = argparse.ArgumentParser().parse_known_args()[1]  # let parse_args run properly
    args = parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if args.stage == "completion" and args.strategy != "B1":
        raise ValueError("--stage completion requires --strategy B1")
    if args.stage == "layout" and args.strategy == "B1":
        raise ValueError("Strategy B1 has no layout stage — use --stage completion")

    # ── Accelerate ────────────────────────────────────────────────────────────
    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device      = accelerator.device

    # ── Save path ─────────────────────────────────────────────────────────────
    job_id    = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"stage2_{args.strategy}_{args.stage}_{args.model_size}_{job_id}"
    save_path = Path(f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints_stage2/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"  CAN3TOK STAGE 2 — Strategy {args.strategy} | Stage {args.stage}")
        print(f"  Model size: {args.model_size}   Save: {save_path}")
        print(f"{'='*70}\n")

    # ── Stage 1 (frozen) ──────────────────────────────────────────────────────
    shape_model, s1_meta = load_stage1(args.stage1_checkpoint, args.stage1_config, device)

    # ── Stage 2 model ─────────────────────────────────────────────────────────
    model = build_stage2_model(args.strategy, args.stage, args.model_size)
    if accelerator.is_main_process:
        print(f"  Stage 2 model: {model}")

    # ── Optionally resume ─────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")
    if args.resume_checkpoint:
        ckpt2 = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt2["model_state_dict"])
        start_epoch   = ckpt2.get("epoch", 0) + 1
        best_val_loss = ckpt2.get("val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}  val_loss={best_val_loss:.4f}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    color_residual = s1_meta["color_residual"]
    train_ds = gs_dataset(
        root=args.data_path, resol=200, random_permute=True, train=True,
        sampling_method="opacity", max_scenes=args.train_scenes,
        normalize=True, normalize_colors=True, target_radius=10.0,
        scale_norm_mode="linear", color_residual=color_residual,
        scene_layout_head=False, position_scaffold=False,
    )
    val_ds = gs_dataset(
        root=args.data_path, resol=200, random_permute=False, train=True,
        sampling_method="opacity", max_scenes=args.val_scenes,
        normalize=True, normalize_colors=True, target_radius=10.0,
        scale_norm_mode="linear", color_residual=color_residual,
        scene_layout_head=False, position_scaffold=False,
    )
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

    # ── LR scheduler ──────────────────────────────────────────────────────────
    bpe          = max(1, len(train_ds) // (args.batch_size * accelerator.num_processes))
    total_steps  = bpe * args.num_epochs
    elapsed      = bpe * start_epoch
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(
            max(0, args.warmup_steps - elapsed),
            total_steps - elapsed,
            args.lr_min_ratio,
        ),
    )

    # ── Flow matching transport ────────────────────────────────────────────────
    transport = create_transport(path_type=args.path_type, prediction=args.prediction)

    # ── Accelerate prepare ────────────────────────────────────────────────────
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    raw_model = accelerator.unwrap_model(model)

    # ── Checkpoint metadata ───────────────────────────────────────────────────
    ckpt_meta = {
        "strategy":          args.strategy,
        "stage":             args.stage,
        "model_size":        args.model_size,
        "path_type":         args.path_type,
        "prediction":        args.prediction,
        "stage1_checkpoint": args.stage1_checkpoint,
        "s1_latent_disentangle":        s1_meta["latent_disentangle"],
        "s1_semantic_dims":             s1_meta["semantic_dims"],
        "s1_decoder_layout_cross_attn": s1_meta["decoder_layout_cross_attn"],
    }

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    print(f"Starting training — epoch {start_epoch} → {args.num_epochs - 1}\n")

    for epoch in tqdm(range(start_epoch, args.num_epochs), disable=not accelerator.is_main_process):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            features = batch["features"].float().to(device)  # [B, 40000, 18]
            B        = features.shape[0]

            optimizer.zero_grad()

            # ── Encode with frozen Stage 1 ──────────────────────────────────
            z_s_clean, z_g_clean, z_clean, z_layout = encode_batch(
                shape_model, features, args.strategy, s1_meta
            )

            # ── Flow matching loss ──────────────────────────────────────────
            if args.stage == "layout":
                # Target: z_s [B, 16, 32]
                terms = transport.training_losses(raw_model, z_s_clean)
                loss  = terms["loss"].mean()

            elif args.stage == "geometry":
                # Target: z_g [B, 496, 32], conditioned on z_s
                terms = transport.training_losses(
                    raw_model, z_g_clean,
                    model_kwargs={"z_s_clean": z_s_clean},
                )
                loss = terms["loss"].mean()

            elif args.stage == "completion":
                # Custom masking + loss on unobserved tokens only
                loss = completion_training_step(
                    raw_model, z_clean, z_layout,
                    transport.path_sampler,
                )

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = scheduler.get_last_lr()[0]

        if accelerator.is_main_process:
            print(f"Epoch {epoch:04d} | Loss={avg_loss:.5f} | LR={lr_now:.2e}")

        # ── Validation ─────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
            model.eval()
            val_loss = 0.0
            n_val    = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].float().to(device)
                    B        = features.shape[0]

                    z_s_clean, z_g_clean, z_clean, z_layout = encode_batch(
                        shape_model, features, args.strategy, s1_meta
                    )

                    if args.stage == "layout":
                        terms    = transport.training_losses(raw_model, z_s_clean)
                        val_loss += terms["loss"].mean().item()
                    elif args.stage == "geometry":
                        terms    = transport.training_losses(
                            raw_model, z_g_clean,
                            model_kwargs={"z_s_clean": z_s_clean},
                        )
                        val_loss += terms["loss"].mean().item()
                    elif args.stage == "completion":
                        val_loss += completion_training_step(
                            raw_model, z_clean, z_layout, transport.path_sampler
                        ).item()
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

        # ── Periodic save ──────────────────────────────────────────────────
        if epoch > 0 and epoch % 100 == 0 and accelerator.is_main_process:
            torch.save(
                {"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                 "train_loss": avg_loss, **ckpt_meta},
                save_path / f"epoch_{epoch}.pth",
            )

    # ── Final save ─────────────────────────────────────────────────────────
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