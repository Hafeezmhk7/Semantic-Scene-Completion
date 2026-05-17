"""
Can3Tok Stage 2 Training — v2
==============================
Key fixes vs previous version:
  1. euler_sample: fixed AttributeError when a Python closure (masked_model for B1)
     is passed instead of an nn.Module — model.eval() / model.train() now only
     called when the argument is actually an nn.Module.

  2. PLY saving during evaluation: every eval_every epochs, the training loop
     now saves reconstructed PLY files for a small number of val scenes using
     the CURRENT DiT weights.  This mirrors exactly what Stage 1 does with
     recon_ply_freq and gives a direct qualitative view of generation quality
     as training progresses.

     For geometry stage: encode z_s_real from val encoder (frozen) →
       generate z_g with current GeometryDiT → decode Z=[z_s_real|z_g_gen].
       Two sub-folders saved: geom_gen/ and geom_gt/ so you can compare.

     For layout stage: generate z_s from noise → decode with zero z_g.

     For completion: 40% mask → complete with CompletionDiT → decode.
       Three sub-folders: completed/, partial/, gt_full/.

  3. mean_color handling: patched to safely fall back to zeros when the key is
     absent from the batch (shouldn't happen but makes the code robust).
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

from gs_dataset_scenesplat import gs_dataset
from gs_ply_reconstructor import save_reconstructed_gaussians
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

from stage2.external.transport import create_transport
from stage2.models.layout_dit    import LayoutDiT_models
from stage2.models.geometry_dit  import GeometryDiT_models, GeometryDiT_adaLN_models
from stage2.models.completion_dit import (
    CompletionDiT_models, completion_training_step, sample_voxel_mask,
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Latent distribution diagnostics ──────────────────────────────────────────
# Inline here so no extra file dependency on the cluster

def compute_latent_distribution_gap(
    z_g_clean: torch.Tensor,
    z_g_gen:   torch.Tensor,
    epoch:     int,
) -> dict:
    """
    Measures the aggregate posterior hole problem: how far z_g_gen
    (from Stage 2 flow matching) deviates from z_g_clean (what the
    Stage 1 decoder was trained to receive).

    Key metrics
    -----------
    mean_shift       : overall mean bias in generated tokens
    std_ratio        : ratio of generated to GT token variance
    dim_mean_rmse    : RMSE of per-dimension means (15872 dims) → 0 = perfect
    dim_std_rmse     : RMSE of per-dimension stds             → 0 = perfect
    frechet_per_dim  : per-dim Fréchet distance (mean² + std²) → 0 = perfect
    mean_kl          : mean KL(gen||clean) per dimension       → 0 = perfect
    kl_clean_prior   : mean KL(clean||N(0,I)) per dim
                       LARGE → kl_weight too weak in Stage 1
                       This is the aggregate posterior hole size
    kl_gen_prior     : mean KL(gen||N(0,I)) per dim
                       Should be close to kl_clean_prior if Stage 2 is working
    """
    zc = z_g_clean.float().cpu()  # [B, 496, 32]
    zg = z_g_gen.float().cpu()    # [B, 496, 32]
    B  = zc.shape[0]

    clean_mean = zc.mean().item()
    clean_std  = zc.std().item()
    gen_mean   = zg.mean().item()
    gen_std    = zg.std().item()
    mean_shift = abs(gen_mean - clean_mean)
    std_ratio  = gen_std / (clean_std + 1e-8)

    zc_flat = zc.reshape(B, -1)   # [B, 15872]
    zg_flat = zg.reshape(B, -1)

    dmc = zc_flat.mean(0); dsc = zc_flat.std(0)
    dmg = zg_flat.mean(0); dsg = zg_flat.std(0)

    dim_mean_rmse   = ((dmg - dmc)**2).mean().item()**0.5
    dim_std_rmse    = ((dsg - dsc)**2).mean().item()**0.5
    frechet_per_dim = ((dmg - dmc)**2).mean().item() + ((dsg - dsc)**2).mean().item()

    eps = 1e-8
    sc  = dsc.clamp(min=eps); sg = dsg.clamp(min=eps)
    kl  = (torch.log(sc/sg) + (sg**2 + (dmg-dmc)**2)/(2*sc**2) - 0.5).mean().item()

    kl_c_prior = (0.5*(dmc**2 + dsc**2 - torch.log(dsc**2+eps) - 1)).mean().item()
    kl_g_prior = (0.5*(dmg**2 + dsg**2 - torch.log(dsg**2+eps) - 1)).mean().item()

    print(f"\n{'='*65}")
    print(f"  LATENT DISTRIBUTION GAP  —  epoch {epoch:04d}")
    print(f"{'='*65}")
    print(f"  Overall:")
    print(f"    z_g_clean : mean={clean_mean:+.4f}  std={clean_std:.4f}")
    print(f"    z_g_gen   : mean={gen_mean:+.4f}  std={gen_std:.4f}")
    print(f"    mean shift={mean_shift:.4f}  std ratio={std_ratio:.4f}  (ideal: 0, 1.0)")
    print(f"  Per-dimension (15,872 dims):")
    print(f"    RMSE means : {dim_mean_rmse:.5f}   (0 = generated means match GT)")
    print(f"    RMSE stds  : {dim_std_rmse:.5f}   (0 = generated variance matches GT)")
    print(f"    Fréchet/dim: {frechet_per_dim:.5f}   (sum of above, 0 = perfect match)")
    print(f"    Mean KL(gen||clean): {kl:.5f}   (0 = perfect match)")
    print(f"  Aggregate posterior hole (KL to N(0,I)):")
    print(f"    KL(z_g_clean || N(0,I)): {kl_c_prior:.4f}")
    print(f"    KL(z_g_gen   || N(0,I)): {kl_g_prior:.4f}")
    print(f"    If KL(clean||prior) >> 0: Stage 1 kl_weight too weak.")
    print(f"    If KL(gen||prior) < KL(clean||prior): DiT undershoots —")
    print(f"      generated tokens are closer to N(0,I) than decoder expects.")
    print(f"{'='*65}\n")

    return {
        "mean_shift": mean_shift, "std_ratio": std_ratio,
        "dim_mean_rmse": dim_mean_rmse, "dim_std_rmse": dim_std_rmse,
        "frechet_per_dim": frechet_per_dim, "mean_kl": kl,
        "kl_clean_prior": kl_c_prior, "kl_gen_prior": kl_g_prior,
    }


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
    if missing:    print(f"  [Stage 1] {len(missing)} missing keys (expected)")
    if unexpected: print(f"  [Stage 1] {len(unexpected)} unexpected keys")

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
# Encode batch (frozen Stage 1)
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
    z_layout  = None
    if strategy == "B1" and hasattr(shape_model, "layout_projector") and \
            shape_model.layout_projector is not None:
        z_layout = shape_model.layout_projector(shape_embed)
    return z_s_clean, z_g_clean, z_clean, z_layout


# ============================================================================
# Euler sampler — FIX: handle non-Module callables (e.g. masked_model closure)
# ============================================================================

@torch.no_grad()
def euler_sample(model, x_init: torch.Tensor, num_steps: int = 50, **kw) -> torch.Tensor:
    """
    Fixed Euler ODE sampler.

    Bug fixed: the original version called model.eval() and model.train() unconditionally.
    When model is a Python closure (e.g. the masked_model lambda used for B1 completion),
    this raised AttributeError because functions don't have .eval() / .train() methods.
    We now guard with isinstance(model, torch.nn.Module).
    """
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
# Decode Z with frozen Stage 1 decoder and handle color_residual
# ============================================================================

@torch.no_grad()
def decode_z(shape_model, Z: torch.Tensor, color_residual: bool,
             mean_color: torch.Tensor = None,
             z_layout: torch.Tensor = None) -> np.ndarray:
    """
    Decode latent Z [B, 512, 32] → numpy array [B, 40000, 14].
    Adds mean_color back if color_residual=True.

    z_layout is REQUIRED for Strategy B1 (decoder_layout_cross_attn=True).
    The B1 decoder has 12 ZSCondTransformerBlock layers where z_layout
    provides K and V in cross-attention at every layer. Without it, those
    12 cross-attention outputs are zero/garbage regardless of Z quality,
    causing:
      - positions collapsed to ±2m (no spatial reference frame)
      - scales of ~108cm (large diffuse blobs as fallback)
      - opacity stuck at 0.5 (sigmoid(0) = 0.5 from zero hidden states)
    For Strategy A and D z_layout=None is correct — their decoder uses
    self-attention over all 512 tokens with no separate z_layout.
    """
    B = Z.shape[0]
    recon, _ = shape_model.decode(Z, volume_queries=None,
                                   return_semantic_features=False,
                                   shape_embed=None,
                                   z_layout=z_layout)
    preds = recon.reshape(B, 40000, 14).cpu().numpy()
    if color_residual and mean_color is not None:
        mc = mean_color.cpu().numpy() if isinstance(mean_color, torch.Tensor) else mean_color
        for i in range(B):
            preds[i, :, 3:6] = np.clip(preds[i, :, 3:6] + mc[i], 0.0, 1.0)
    return preds


# ============================================================================
# PLY saving during evaluation
# ============================================================================

@torch.no_grad()
def save_eval_ply(
    raw_model,
    shape_model,
    val_loader,
    strategy:       str,
    stage:          str,
    zs_conditioning: str,
    save_dir:       Path,
    epoch:          int,
    device:         torch.device,
    color_residual: bool,
    num_scenes:     int = 4,
    num_steps:      int = 50,
):
    """
    Save PLY files during training evaluation.

    This mirrors what Stage 1 does with recon_ply_freq — gives you a direct
    qualitative view of how the Stage 2 DiT is improving each eval cycle.

    Geometry stage
    --------------
    For each of num_scenes val scenes:
      1. Encode → get z_s_real from frozen Stage 1 encoder (NOT generated)
      2. Run euler_sample with current GeometryDiT to generate z_g
      3. Assemble Z = [z_s_real | z_g_gen] → decode with frozen Stage 1 decoder
      4. Also decode the ground-truth Z for direct comparison

    Saved to:
      {save_dir}/recon_ply/epoch_{NNNN}/geom_gen/scene_NNN_epoch_NNN.ply
      {save_dir}/recon_ply/epoch_{NNNN}/geom_gt/scene_NNN_epoch_NNN.ply

    Open geom_gen/ and geom_gt/ side-by-side in SuperSplat — they should look
    increasingly similar as training progresses.

    Layout stage
    ------------
    Generate z_s from pure noise → decode with z_g = zeros.
    Only the colour and coarse scene type are visible (z_g is uninformative),
    but this shows whether LayoutDiT is learning the right z_s distribution.

    Completion stage (B1)
    ---------------------
    Take val scenes, mask 40% → complete with CompletionDiT → decode.
    Three outputs: partial/ (input), completed/ (output), gt_full/ (reference).
    """
    ply_dir = save_dir / "recon_ply" / f"epoch_{epoch:04d}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Grab one batch from val loader
    batch    = next(iter(val_loader))
    features = batch["features"].float().to(device)[:num_scenes]
    B        = features.shape[0]
    # mean_color is always in the batch from gs_dataset; safe to fetch
    mean_color_raw = batch.get("mean_color", None)
    mean_color = mean_color_raw[:B] if mean_color_raw is not None else None

    try:
        if stage == "geometry":
            # ── Encode real z_s and z_g ─────────────────────────────────────
            z_s_real, z_g_real, z_clean, _ = encode_batch(
                shape_model, features, strategy, {}
            )

            # ── Generate z_g conditioned on real z_s ────────────────────────
            z_g_noise = torch.randn(B, 496, 32, device=device)
            z_g_gen   = euler_sample(raw_model, z_g_noise,
                                     num_steps=num_steps, z_s_clean=z_s_real)

            Z_gen  = torch.cat([z_s_real, z_g_gen], dim=1)     # [B, 512, 32]

            # ── Latent distribution gap diagnostic ───────────────────────────
            # Compare statistics of z_g_clean (what decoder was trained on)
            # vs z_g_gen (what Stage 2 produces).
            # Measures the aggregate posterior hole problem: how far generated
            # tokens deviate from the decoder's expected input distribution.
            compute_latent_distribution_gap(z_g_real, z_g_gen, epoch)

            # ── Decode both generated and GT ─────────────────────────────────
            preds_gen = decode_z(shape_model, Z_gen,    color_residual, mean_color)
            preds_gt  = decode_z(shape_model, z_clean,  color_residual, mean_color)

            save_reconstructed_gaussians(
                predictions=preds_gen, output_dir=ply_dir / "geom_gen",
                epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")
            save_reconstructed_gaussians(
                predictions=preds_gt,  output_dir=ply_dir / "geom_gt",
                epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")

            print(f"  [PLY] epoch {epoch:04d}: {B} scene(s) → {ply_dir}")
            print(f"        geom_gen/  ← generated z_g, real z_s")
            print(f"        geom_gt/   ← ground-truth decode (Stage 1 upper bound)")

        elif stage == "layout":
            # ── Generate z_s from noise; decode with zero z_g ───────────────
            z_s_noise = torch.randn(B, 16, 32, device=device)
            z_s_gen   = euler_sample(raw_model, z_s_noise, num_steps=num_steps)
            z_g_zero  = torch.zeros(B, 496, 32, device=device)
            Z_gen     = torch.cat([z_s_gen, z_g_zero], dim=1)

            preds_gen = decode_z(shape_model, Z_gen, color_residual, mean_color)
            save_reconstructed_gaussians(
                predictions=preds_gen, output_dir=ply_dir / "layout_gen",
                epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")

            print(f"  [PLY] epoch {epoch:04d}: layout_gen/ ({B} scenes, z_g=zeros)")

        elif stage == "completion":
            # ── Encode, mask, complete, decode ──────────────────────────────
            z_s_real, _, z_clean, z_layout = encode_batch(
                shape_model, features, "B1", {}
            )
            if z_layout is None:
                print("  [PLY] skipped — layout_projector not available")
                return

            coverage = 0.4
            obs_mask = sample_voxel_mask(B, 512, device=device,
                                         coverage_range=(coverage, coverage))
            mask_exp = obs_mask.unsqueeze(-1)
            z_noise  = torch.randn_like(z_clean)
            z_init   = z_clean * mask_exp + z_noise * (1.0 - mask_exp)

            # Closure — mask observed tokens throughout sampling
            # Note: passed to euler_sample which handles non-Module callables
            def masked_model(x, t, **_kw):
                v = raw_model(x, t, z_layout=z_layout, obs_mask=obs_mask)
                return v * (1.0 - mask_exp)

            z_comp = euler_sample(masked_model, z_init, num_steps=num_steps)
            z_comp = z_comp * (1.0 - mask_exp) + z_clean * mask_exp  # restore observed
            z_part = z_clean * mask_exp                               # zero out unobserved

            for z_arr, name in [(z_comp, "completed"), (z_part, "partial"), (z_clean, "gt_full")]:
                preds = decode_z(shape_model, z_arr, color_residual, mean_color,
                                 z_layout=z_layout)   # critical: B1 decoder needs this
                save_reconstructed_gaussians(
                    predictions=preds, output_dir=ply_dir / name,
                    epoch=epoch, num_scenes=B, max_sh_degree=3, color_mode="1")

            print(f"  [PLY] epoch {epoch:04d}: completion ({B} scenes, "
                  f"coverage={coverage:.0%})")
            print(f"        completed/ | partial/ | gt_full/")

    except Exception as exc:
        import traceback
        print(f"  [PLY] FAILED at epoch {epoch}: {exc}")
        traceback.print_exc()


# ============================================================================
# Flow diagnostics
# ============================================================================

def compute_flow_diagnostics(model, x_clean, model_kwargs, n_bins=4):
    B, device = x_clean.shape[0], x_clean.device
    t       = torch.rand(B, device=device)
    x_noise = torch.randn_like(x_clean)
    t_exp   = t.view(B, *([1] * (x_clean.ndim - 1)))
    x_t     = t_exp * x_clean + (1.0 - t_exp) * x_noise
    v_target = x_clean - x_noise

    with torch.no_grad():
        v_pred = model(x_t, t, **model_kwargs)

    bins     = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_loss = {}
    for i in range(n_bins):
        mask = (t >= bins[i]) & (t < bins[i + 1])
        if mask.sum() > 0:
            bin_loss[f"loss_t{i}"] = ((v_pred[mask] - v_target[mask]) ** 2).mean().item()

    vp = v_pred.reshape(B, -1);  vt = v_target.reshape(B, -1)
    cos = ((vp * vt).sum(1) / (vp.norm(1) * vt.norm(1) + 1e-8)).mean().item()

    return {
        "t_mean": t.mean().item(), "t_std": t.std().item(),
        "vtarget_mean": v_target.mean().item(), "vtarget_std": v_target.std().item(),
        "vpred_mean": v_pred.mean().item(),     "vpred_std":  v_pred.std().item(),
        "vpred_vtarget_cosine": cos, **bin_loss,
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
        suffix = "A" if strategy == "A" else "D"
        return GeometryDiT_models[f"GeometryDiT{suffix}-{size}"]()
    elif stage == "completion":
        assert strategy == "B1"
        return CompletionDiT_models[f"CompletionDiT-{size}"]()
    raise ValueError(f"Unknown stage '{stage}'")


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Can3Tok Stage 2 Training")
    p.add_argument("--strategy",           type=str, required=True, choices=["A","D","B1"])
    p.add_argument("--stage",              type=str, required=True,
                   choices=["layout","geometry","completion"])
    p.add_argument("--stage1_checkpoint",  type=str, required=True)
    p.add_argument("--model_size",         type=str, default="B", choices=["S","B","L"])
    p.add_argument("--resume_checkpoint",  type=str, default=None)
    p.add_argument("--zs_conditioning",    type=str, default="cross_attn",
                   choices=["cross_attn","adaLN"])
    p.add_argument("--batch_size",         type=int,   default=64)
    p.add_argument("--num_epochs",         type=int,   default=500)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--weight_decay",       type=float, default=1e-2)
    p.add_argument("--warmup_steps",       type=int,   default=200)
    p.add_argument("--lr_min_ratio",       type=float, default=0.1)
    p.add_argument("--eval_every",         type=int,   default=25)
    p.add_argument("--train_scenes",       type=int,   default=None)
    p.add_argument("--val_scenes",         type=int,   default=50)
    p.add_argument("--data_path",          type=str,
                   default="/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"
                           "/train_grid1.0cm_chunk8x8_stride6x6")
    p.add_argument("--path_type",          type=str, default="Linear",
                   choices=["Linear","GVP","VP"])
    p.add_argument("--prediction",         type=str, default="velocity",
                   choices=["velocity","noise","score"])
    # PLY vis: now triggered at every eval_every (not a separate flag)
    p.add_argument("--vis_freq",           type=int, default=0,
                   help="DEPRECATED — PLY is now saved automatically at every "
                        "eval_every epoch. Set to 0 to keep old behaviour; "
                        "non-zero still works as before for backward compat.")
    p.add_argument("--vis_num_scenes",     type=int, default=4)
    p.add_argument("--vis_num_steps",      type=int, default=50)
    p.add_argument("--flow_diag_freq",     type=int, default=0)
    p.add_argument("--stage1_config",      type=str,
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
        return lr_min_ratio + (1-lr_min_ratio)*0.5*(1+math.cos(math.pi*t/cosine_steps))
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

    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device      = accelerator.device

    job_id   = os.environ.get("SLURM_JOB_ID", "local")
    cond_tag = f"_{args.zs_conditioning}" if args.stage == "geometry" else ""
    run_name = f"stage2_{args.strategy}_{args.stage}{cond_tag}_{args.model_size}_{job_id}"
    save_path = Path(
        f"/home/yli11/scratch-project/Hafeez_thesis/Can3Tok/checkpoints_stage2/{run_name}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"  CAN3TOK STAGE 2 — Strategy {args.strategy} | Stage {args.stage}")
        if args.stage == "geometry":
            print(f"  z_s conditioning: {args.zs_conditioning}")
        print(f"  Model size: {args.model_size}   Save: {save_path}")
        print(f"  PLY saved every {args.eval_every} epochs (at each eval step)")
        if args.flow_diag_freq > 0:
            print(f"  Flow diagnostics every {args.flow_diag_freq} epochs")
        print(f"{'='*70}\n")

    shape_model, s1_meta = load_stage1(args.stage1_checkpoint, args.stage1_config, device)
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
        root=args.data_path, resol=200, sampling_method="opacity",
        normalize=True, normalize_colors=True, target_radius=10.0,
        scale_norm_mode="linear", color_residual=color_residual,
        scene_layout_head=False, position_scaffold=False,
    )
    train_ds     = gs_dataset(**ds_kwargs, random_permute=True,  train=True,
                              max_scenes=args.train_scenes)
    val_ds       = gs_dataset(**ds_kwargs, random_permute=False, train=True,
                              max_scenes=args.val_scenes)
    train_loader = Data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=8, pin_memory=True)
    val_loader   = Data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                   num_workers=8, pin_memory=True)

    if accelerator.is_main_process:
        print(f"  Train: {len(train_ds)} scenes | Val: {len(val_ds)} scenes\n")

    bpe         = max(1, len(train_ds) // (args.batch_size * accelerator.num_processes))
    total_steps = bpe * args.num_epochs
    elapsed     = bpe * start_epoch
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_lambda(
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
        "strategy": args.strategy, "stage": args.stage,
        "zs_conditioning": args.zs_conditioning, "model_size": args.model_size,
        "path_type": args.path_type, "prediction": args.prediction,
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
        epoch_loss      = 0.0
        epoch_vpred_std = 0.0
        n_batches       = 0

        for batch in train_loader:
            features = batch["features"].float().to(device)
            optimizer.zero_grad()

            z_s_clean, z_g_clean, z_clean, z_layout = encode_batch(
                shape_model, features, args.strategy, s1_meta
            )

            if args.stage == "layout":
                terms = transport.training_losses(raw_model, z_s_clean)
                loss  = terms["loss"].mean()
                epoch_vpred_std += terms["pred"].std().item()

            elif args.stage == "geometry":
                terms = transport.training_losses(
                    raw_model, z_g_clean,
                    model_kwargs={"z_s_clean": z_s_clean},
                )
                loss = terms["loss"].mean()
                epoch_vpred_std += terms["pred"].std().item()

            elif args.stage == "completion":
                loss = completion_training_step(
                    raw_model, z_clean, z_layout, transport.path_sampler,
                )

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = scheduler.get_last_lr()[0]

        if accelerator.is_main_process:
            print(f"Epoch {epoch:04d} | Loss={avg_loss:.5f} | LR={lr_now:.2e}", end="")
            if args.stage != "completion":
                print(f" | v_pred_std={epoch_vpred_std/max(n_batches,1):.4f}", end="")
            print()

        # ── Flow diagnostics ────────────────────────────────────────────────
        if (args.flow_diag_freq > 0 and epoch % args.flow_diag_freq == 0
                and accelerator.is_main_process and args.stage != "completion"):
            model.eval()
            try:
                db = next(iter(val_loader))
                df = db["features"].float().to(device)
                with torch.no_grad():
                    zs, zg, zc, zl = encode_batch(shape_model, df, args.strategy, s1_meta)
                target = zg if args.stage == "geometry" else zs
                mkw    = {"z_s_clean": zs} if args.stage == "geometry" else {}
                diag   = compute_flow_diagnostics(raw_model, target, mkw)
                print(f"  [FLOW DIAG epoch {epoch}]")
                print(f"    t:       mean={diag['t_mean']:.3f}  std={diag['t_std']:.3f}  "
                      f"(expect ~0.500/~0.289)")
                print(f"    vtarget: mean={diag['vtarget_mean']:+.4f}  "
                      f"std={diag['vtarget_std']:.4f}  (expect ~0/~1.41)")
                print(f"    vpred:   mean={diag['vpred_mean']:+.4f}  "
                      f"std={diag['vpred_std']:.4f}")
                print(f"    cosine(vpred,vtarget) = {diag['vpred_vtarget_cosine']:.4f}  "
                      f"(0=random → 1=perfect)")
                for i in range(4):
                    k = f"loss_t{i}"
                    if k in diag:
                        print(f"    t_bin_{i} [{i/4:.2f},{(i+1)/4:.2f}]: {diag[k]:.5f}")
            except Exception as e:
                print(f"  [FLOW DIAG] Failed: {e}")
            model.train()

        # ── Validation + PLY saving ──────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
            model.eval()
            val_loss = 0.0;  n_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].float().to(device)
                    zs, zg, zc, zl = encode_batch(shape_model, features,
                                                   args.strategy, s1_meta)
                    if args.stage == "layout":
                        val_loss += transport.training_losses(raw_model, zs)["loss"].mean().item()
                    elif args.stage == "geometry":
                        val_loss += transport.training_losses(
                            raw_model, zg, model_kwargs={"z_s_clean": zs}
                        )["loss"].mean().item()
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

                # ── Save PLY at every eval step ──────────────────────────────
                # Skip epoch 0 (model is random — not worth saving disk space)
                if epoch > 0:
                    save_eval_ply(
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
                        num_scenes=args.vis_num_scenes,
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