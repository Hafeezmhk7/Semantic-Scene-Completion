"""
Can3Tok — Stage 1 GS Decoder Robustness Fine-Tuning
====================================================

Problem
-------
The GS decoder (777M params) was trained exclusively on 300 encoder outputs.
When Stage 2 generates z_g_gen that are ~RMSE 0.09–0.20 off the encoder
manifold, the decoder extrapolates poorly and produces blurry Gaussians.

Solution (LV-RAE, arXiv:2602.08620)
------------------------------------
Fine-tune ONLY the GS decoder to produce the same output for Stage 2
generated latents as it does for encoder latents from the same scene.
Everything else (encoder, bottleneck, decoder transformer, Stage 2 DiT)
stays frozen.

Method
------
For each training batch:
  1. Encode scenes → z_s_clean, z_g_clean                   [frozen encoder]
  2. Generate z_g_gen with Stage 2 DiT, 20 Euler steps      [frozen DiT]
  3. Mix: mix_ratio of samples use z_g_gen, rest use z_g_clean
  4. Get transformer hidden states for both                  [frozen transformer]
       H_flat_clean = transformer(post_kl(Z_clean)).reshape(B,-1)
       H_flat_mixed = transformer(post_kl(Z_mixed)).reshape(B,-1)
  5. Compute reference target with FROZEN GS decoder:
       target = gs_decoder_frozen(H_flat_clean)
  6. Compute prediction with UPDATING GS decoder:
       preds  = gs_decoder_update(H_flat_mixed)
  7. Loss = MSE(preds, target)
       For clean samples:  teaches decoder(Z_clean) ≈ original_decoder(Z_clean)
                           → anchor: prevents forgetting
       For gen samples:    teaches decoder(Z_gen) ≈ original_decoder(Z_clean)
                           → robustness: generated latents produce same Gaussians

Gradient path
-------------
  Loss → preds (from GS_decoder weights) → ∅  (H_flat has no grad, frozen path)
  Only GS_decoder parameters are updated.
  All other Stage 1 and Stage 2 parameters: requires_grad=False throughout.

Memory (single H100, batch=32)
-------------------------------
  GS decoder ×2 (update + frozen ref): 2 × 1.55GB = 3.10GB
  Stage 1 model:                                     ~2.00GB
  Stage 2 DiT:                                       ~0.08GB
  AdamW state (GS decoder only):                     ~6.22GB
  Activations:                                       ~0.05GB
  Total:                                             ~11.5GB  (H100: 80GB ✓)

Output
------
  best_model.pth — full Stage 1 model state_dict with fine-tuned GS decoder
                   Drop-in replacement for the original Stage 1 checkpoint.
                   Load exactly as you load the original Stage 1 checkpoint.
"""

import os
import sys
import copy
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data as Data

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gs_dataset_scenesplat import gs_dataset
from stage2.train_stage2 import (
    load_stage1, encode_batch, euler_sample, build_stage2_model,
    compute_latent_distribution_gap,
)
from gs_ply_reconstructor import save_reconstructed_gaussians


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Can3Tok GS Decoder Robustness Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--stage1_checkpoint",   type=str, required=True,
                   help="Stage 1 best_model.pth (Strategy A)")
    p.add_argument("--geometry_checkpoint", type=str, required=True,
                   help="Stage 2 geometry DiT best_model.pth")
    p.add_argument("--output_dir",          type=str, required=True,
                   help="Directory to save fine-tuned checkpoint")

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--stage1_config",  type=str,
                   default="./model/configs/aligned_shape_latents/shapevae-256.yaml")
    p.add_argument("--data_path",      type=str,
                   default="/home/yli11/scratch/datasets/gaussian_world/preprocessed"
                           "/interior_gs/train_grid1.0cm_chunk8x8_stride6x6")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--num_epochs",   type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-5,
                   help="Learning rate. Much smaller than Stage 1 (1e-4) to avoid forgetting")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_scenes", type=int,   default=None,
                   help="Number of training scenes (None = all)")
    p.add_argument("--val_scenes",   type=int,   default=50)

    # ── Fine-tuning specific ──────────────────────────────────────────────────
    p.add_argument("--mix_ratio",    type=float, default=0.5,
                   help="Fraction of each batch that uses Stage 2 generated latents. "
                        "0.5 = half clean + half generated.")
    p.add_argument("--euler_steps",  type=int,   default=20,
                   help="Euler steps for Stage 2 DiT sampling. "
                        "20 steps is ~4× faster than inference (50) and sufficient "
                        "to produce off-manifold latents representative of Stage 2.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument("--eval_every",   type=int,   default=10,
                   help="Evaluate + save checkpoint every N epochs")
    p.add_argument("--vis_every",    type=int,   default=50,
                   help="Save PLY visualisation every N epochs (0=off)")
    p.add_argument("--vis_scenes",   type=int,   default=4)

    return p.parse_args()


# ============================================================================
# Helper: visualise decoded Gaussians from fine-tuned decoder
# ============================================================================

@torch.no_grad()
def save_vis_ply(shape_model, geometry_model, val_loader, save_dir, epoch,
                 device, color_residual, s1_meta, n_scenes=4, euler_steps=20):
    """Save PLY files: fine-tuned decoder vs frozen decoder, both from Stage 2 latents."""
    save_dir.mkdir(parents=True, exist_ok=True)

    batch    = next(iter(val_loader))
    features = batch["features"].float().to(device)[:n_scenes]
    B        = features.shape[0]

    z_s_clean, z_g_clean, z_clean, _ = encode_batch(
        shape_model, features, "A", s1_meta
    )
    z_g_noise = torch.randn(B, 496, 32, device=device)
    z_g_gen   = euler_sample(geometry_model, z_g_noise, num_steps=euler_steps,
                             z_s_clean=z_s_clean)

    Z_gen   = torch.cat([z_s_clean, z_g_gen],   dim=1)
    Z_clean = torch.cat([z_s_clean, z_g_clean], dim=1)

    for Z, name in [(Z_gen, "gen"), (Z_clean, "clean_ref")]:
        recon, _ = shape_model.decode(Z, return_semantic_features=False)
        preds    = recon.reshape(B, 40000, 14).cpu().numpy()
        if color_residual:
            mc = batch.get("mean_color", None)
            if mc is not None:
                mc = mc[:B].numpy()
                for i in range(B):
                    preds[i, :, 3:6] = preds[i, :, 3:6] + mc[i]
        save_reconstructed_gaussians(
            predictions=preds,
            output_dir=save_dir / name,
            epoch=epoch,
            num_scenes=B,
            max_sh_degree=3,
            color_mode="1",
        )

    print(f"  [VIS] epoch {epoch:04d}: {save_dir}/gen/ and clean_ref/")


# ============================================================================
# Main
# ============================================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Can3Tok — GS Decoder Robustness Fine-Tuning")
    print(f"{'='*65}")
    print(f"  Stage 1:         {args.stage1_checkpoint}")
    print(f"  Stage 2 DiT:     {args.geometry_checkpoint}")
    print(f"  Output:          {output_dir}")
    print(f"  Mix ratio:       {args.mix_ratio}  "
          f"({int(args.mix_ratio*100)}% generated, "
          f"{100-int(args.mix_ratio*100)}% clean)")
    print(f"  Euler steps:     {args.euler_steps}  (Stage 2 sampling, fast)")
    print(f"  LR:              {args.lr}  (10× smaller than Stage 1)")
    print(f"{'='*65}\n")

    # ── Load Stage 1 ─────────────────────────────────────────────────────────
    print("Loading Stage 1 model...")
    shape_model, s1_meta = load_stage1(
        args.stage1_checkpoint, args.stage1_config, device
    )
    # Freeze everything in Stage 1
    for p in shape_model.parameters():
        p.requires_grad_(False)
    color_residual = s1_meta["color_residual"]

    # Identify the GS decoder to fine-tune
    # Strategy A → shape_model.GS_decoder
    # Strategy B1 → shape_model.GS_decoder_B
    if (s1_meta.get("decoder_layout_cross_attn", False)
            and hasattr(shape_model, "GS_decoder_B")
            and shape_model.GS_decoder_B is not None):
        gs_decoder_live = shape_model.GS_decoder_B
        print("  Fine-tuning: GS_decoder_B (Strategy B1)")
    else:
        gs_decoder_live = shape_model.GS_decoder
        print("  Fine-tuning: GS_decoder (Strategy A)")

    n_params = sum(p.numel() for p in gs_decoder_live.parameters())
    print(f"  Parameters to update: {n_params/1e6:.1f}M")

    # Unfreeze only the GS decoder
    for p in gs_decoder_live.parameters():
        p.requires_grad_(True)

    # ── Frozen reference copy of GS decoder ──────────────────────────────────
    # Used to compute targets.  Never updated.
    print("\nCreating frozen reference copy of GS decoder...")
    gs_decoder_ref = copy.deepcopy(gs_decoder_live)
    gs_decoder_ref.to(device).eval()
    for p in gs_decoder_ref.parameters():
        p.requires_grad_(False)
    print(f"  Reference copy: {n_params/1e6:.1f}M params (frozen)")

    # Validate get_decoder_transformer_features is available
    if not hasattr(shape_model, "get_decoder_transformer_features"):
        raise AttributeError(
            "shape_model does not have get_decoder_transformer_features(). "
            "Please ensure sal_perceiver_dist_changes.py contains the LPL method."
        )
    print("  get_decoder_transformer_features() found ✓")

    # ── Load Stage 2 geometry DiT ────────────────────────────────────────────
    print("\nLoading Stage 2 geometry DiT...")
    geom_ckpt = torch.load(
        args.geometry_checkpoint, map_location="cpu", weights_only=False
    )
    geom_strategy = geom_ckpt.get("strategy",        "A")
    geom_size     = geom_ckpt.get("model_size",       "B")
    geom_zs_cond  = geom_ckpt.get("zs_conditioning", "cross_attn")
    geom_rope     = geom_ckpt.get("rope_type",        "learned_ape")

    geometry_model = build_stage2_model(geom_strategy, "geometry", geom_size,
                                        geom_zs_cond, geom_rope)
    geometry_model.load_state_dict(geom_ckpt["model_state_dict"])
    geometry_model.to(device).eval()
    for p in geometry_model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded: strategy={geom_strategy}, size={geom_size}, "
          f"zs_cond={geom_zs_cond}")
    print(f"  Val loss at saved epoch: "
          f"{geom_ckpt.get('val_loss', '?'):.5f}"
          if isinstance(geom_ckpt.get("val_loss"), float) else "")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\nBuilding datasets...")
    ds_kwargs = dict(
        root=args.data_path, resol=200, sampling_method="opacity",
        normalize=True, normalize_colors=True, target_radius=10.0,
        scale_norm_mode="linear", color_residual=color_residual,
        scene_layout_head=False, position_scaffold=False,
    )
    train_ds = gs_dataset(**ds_kwargs, random_permute=True,  train=True,
                          max_scenes=args.train_scenes)
    val_ds   = gs_dataset(**ds_kwargs, random_permute=False, train=True,
                          max_scenes=args.val_scenes)
    train_loader = Data.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = Data.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"  Train: {len(train_ds)} scenes | Val: {len(val_ds)} scenes")

    # ── Optimiser (GS decoder only) ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        gs_decoder_live.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    # Cosine schedule: LR → 10% of peak over full training
    total_steps = len(train_loader) * args.num_epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    print(f"\n  Optimiser: AdamW lr={args.lr}, wd={args.weight_decay}")
    print(f"  Scheduler: cosine, {total_steps} steps, eta_min={args.lr*0.1}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_metric = float("inf")

    for epoch in tqdm(range(args.num_epochs), desc="Decoder fine-tune"):
        shape_model.train()     # enables GS decoder training mode (dropout etc.)
        geometry_model.eval()   # Stage 2 DiT always in eval

        n_batches      = 0
        sum_loss       = 0.0
        sum_anchor     = 0.0   # loss on clean samples only
        sum_robust     = 0.0   # loss on generated samples only

        for batch in train_loader:
            features = batch["features"].float().to(device)
            B        = features.shape[0]
            optimizer.zero_grad()

            # ── Step 1: encode → clean latents ───────────────────────────────
            with torch.no_grad():
                z_s_clean, z_g_clean, z_clean, _ = encode_batch(
                    shape_model, features, geom_strategy, s1_meta
                )

            # ── Step 2: generate Stage 2 latents ─────────────────────────────
            # 20 Euler steps: ~4× faster than inference quality (50 steps)
            # Produces off-manifold latents representative of Stage 2 outputs
            with torch.no_grad():
                z_g_noise = torch.randn(B, 496, 32, device=device)
                z_g_gen   = euler_sample(
                    geometry_model, z_g_noise,
                    num_steps=args.euler_steps,
                    z_s_clean=z_s_clean,
                )

            # ── Step 3: build mixed latents ───────────────────────────────────
            # mix_mask[b] = True  → sample b uses Stage 2 generated z_g_gen
            # mix_mask[b] = False → sample b uses encoder z_g_clean
            mix_mask  = torch.rand(B, device=device) < args.mix_ratio  # [B]
            z_g_mixed = z_g_clean.clone()
            if mix_mask.any():
                z_g_mixed[mix_mask] = z_g_gen[mix_mask]

            Z_clean = torch.cat([z_s_clean, z_g_clean], dim=1)  # [B, 512, 32]
            Z_mixed = torch.cat([z_s_clean, z_g_mixed], dim=1)  # [B, 512, 32]

            # ── Step 4: frozen transformer features ───────────────────────────
            # get_decoder_transformer_features under no_grad:
            #   no checkpoint issue (no backward graph needed here)
            with torch.no_grad():
                H_clean = shape_model.get_decoder_transformer_features(Z_clean)
                H_mixed = shape_model.get_decoder_transformer_features(Z_mixed)

                # Flatten: [B, 512, 384] → [B, 196608]
                H_flat_clean = H_clean.reshape(B, -1)
                H_flat_mixed = H_mixed.reshape(B, -1)

                # FROZEN REFERENCE: what the original decoder produces for clean latents
                # This is the target for BOTH clean and generated samples.
                #   For clean:   teach gs_decoder_live(H_clean) ≈ gs_decoder_ref(H_clean)
                #                → prevents decoder forgetting Stage 1 quality (anchor)
                #   For generated: teach gs_decoder_live(H_gen) ≈ gs_decoder_ref(H_clean)
                #                → decoder becomes robust to Stage 2 off-manifold inputs
                target = gs_decoder_ref(H_flat_clean)   # [B, 40000*14]  frozen

            # ── Step 5: updating GS decoder forward ───────────────────────────
            # H_flat_mixed: requires_grad=False (computed under no_grad)
            # gs_decoder_live weights: requires_grad=True
            # backward: gradients flow through GS_decoder weights only ✓
            preds = gs_decoder_live(H_flat_mixed)        # [B, 40000*14]

            # ── Step 6: losses ────────────────────────────────────────────────
            preds_3d  = preds.reshape(B, 40000, 14)
            target_3d = target.reshape(B, 40000, 14).detach()

            loss = F.mse_loss(preds_3d, target_3d)

            # Separate logging (no extra compute — just slice)
            with torch.no_grad():
                if (~mix_mask).any():
                    sum_anchor += F.mse_loss(
                        preds_3d[~mix_mask], target_3d[~mix_mask]
                    ).item()
                if mix_mask.any():
                    sum_robust += F.mse_loss(
                        preds_3d[mix_mask], target_3d[mix_mask]
                    ).item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            sum_loss  += loss.item()
            n_batches += 1

        avg_loss   = sum_loss   / max(n_batches, 1)
        avg_anchor = sum_anchor / max(n_batches, 1)
        avg_robust = sum_robust / max(n_batches, 1)
        lr_now     = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:04d} | "
              f"Loss={avg_loss:.5f} | "
              f"Anchor={avg_anchor:.5f} | "
              f"Robust={avg_robust:.5f} | "
              f"LR={lr_now:.2e}")

        # ── Evaluation ────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
            shape_model.eval()
            val_anchor = 0.0
            val_robust = 0.0
            n_val      = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].float().to(device)
                    B = features.shape[0]

                    z_s_clean, z_g_clean, _, _ = encode_batch(
                        shape_model, features, geom_strategy, s1_meta
                    )
                    z_g_noise = torch.randn(B, 496, 32, device=device)
                    z_g_gen   = euler_sample(
                        geometry_model, z_g_noise,
                        num_steps=args.euler_steps,
                        z_s_clean=z_s_clean,
                    )

                    Z_clean = torch.cat([z_s_clean, z_g_clean], dim=1)
                    Z_gen   = torch.cat([z_s_clean, z_g_gen],   dim=1)

                    H_clean = shape_model.get_decoder_transformer_features(Z_clean)
                    H_gen   = shape_model.get_decoder_transformer_features(Z_gen)

                    H_flat_clean = H_clean.reshape(B, -1)
                    H_flat_gen   = H_gen.reshape(B, -1)

                    # Reference: what the frozen original decoder produces for clean
                    target      = gs_decoder_ref(H_flat_clean).reshape(B, 40000, 14)

                    # Anchor: does the updated decoder still match for clean inputs?
                    p_clean     = gs_decoder_live(H_flat_clean).reshape(B, 40000, 14)
                    val_anchor += F.mse_loss(p_clean, target).item()

                    # Robust: does the updated decoder produce good output for gen inputs?
                    p_gen       = gs_decoder_live(H_flat_gen).reshape(B, 40000, 14)
                    val_robust += F.mse_loss(p_gen, target).item()

                    n_val += 1

            avg_va = val_anchor / max(n_val, 1)
            avg_vr = val_robust / max(n_val, 1)
            ratio  = avg_vr / max(avg_va, 1e-9)

            print(f"  [VAL] anchor={avg_va:.5f}  robust={avg_vr:.5f}  "
                  f"ratio={ratio:.2f}×")
            print(f"        Ideal: anchor→0 (no forgetting), "
                  f"robust→anchor (Stage 2 latents treated like clean)")

            # ── Save checkpoint ───────────────────────────────────────────────
            val_metric = avg_va + avg_vr   # minimise both
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                torch.save(
                    {
                        "epoch":           epoch,
                        "val_anchor":      avg_va,
                        "val_robust":      avg_vr,
                        "val_ratio":       ratio,
                        # Full model state dict with fine-tuned GS decoder
                        # Drop-in replacement for the original Stage 1 checkpoint
                        "model_state_dict": shape_model.state_dict(),
                        # Also save just the GS decoder for convenience
                        "gs_decoder_state_dict": gs_decoder_live.state_dict(),
                        # Metadata
                        "stage1_checkpoint":  args.stage1_checkpoint,
                        "geometry_checkpoint": args.geometry_checkpoint,
                        "mix_ratio":          args.mix_ratio,
                        "euler_steps":        args.euler_steps,
                        # Stage 1 flags (needed when loading as Stage 1 model)
                        **{k: v for k, v in s1_meta.items()},
                    },
                    output_dir / "best_model.pth",
                )
                print(f"  [SAVED] best model  epoch={epoch}  metric={best_val_metric:.5f}")

        # ── PLY visualisation ─────────────────────────────────────────────────
        if args.vis_every > 0 and (
            epoch % args.vis_every == 0 or epoch == args.num_epochs - 1
        ):
            vis_dir = output_dir / "vis" / f"epoch_{epoch:04d}"
            try:
                save_vis_ply(
                    shape_model, geometry_model, val_loader,
                    vis_dir, epoch, device, color_residual, s1_meta,
                    n_scenes=args.vis_scenes, euler_steps=args.euler_steps,
                )
            except Exception as exc:
                print(f"  [VIS] FAILED at epoch {epoch}: {exc}")

    print(f"\nFine-tuning complete.")
    print(f"Best val metric: {best_val_metric:.5f}")
    print(f"Saved to:  {output_dir / 'best_model.pth'}")
    print()
    print("To use the fine-tuned model in Stage 2 inference:")
    print("  In sample_stage2.py, set:")
    print(f"    --stage1_checkpoint {output_dir / 'best_model.pth'}")
    print("  The fine-tuned Stage 1 checkpoint is a drop-in replacement.")


if __name__ == "__main__":
    main()