"""
Can3Tok Stage 2 — Generation and Completion Inference (schema-aware)
====================================================================
Supports all seven Stage 1 checkpoints under both objectives. The Stage 2
checkpoint records its stage/schema/size/rope, so the right model is rebuilt
automatically; you only pass the checkpoint(s) and an --objective.

Objective 1 — unconditional generation
  FLAT  checkpoint (exp 1-5):  --scene_checkpoint <scene best_model.pth>
  SPLIT checkpoint (exp 6-7):  --layout_checkpoint <...> --geometry_checkpoint <...>

Objective 2 — scene completion
  --completion_checkpoint <...>   (CompletionDiT for SPLIT, CompletionDiTUncond
  for the structured FLAT case). Real scenes are loaded from the dataset and a
  fraction of their tokens is masked to simulate a partial scan, which guarantees
  the frozen encoder receives a valid input. (Loading an arbitrary partial scan
  from disk would require reproducing the dataset's exact feature construction.)

Examples
--------
# Generation, FLAT (e.g. experiment 5)
python sample_stage2.py --objective generation \
    --stage1_checkpoint /path/stage1_best.pth \
    --scene_checkpoint  /path/stage2_scene_best.pth \
    --num_samples 8 --num_steps 50 --output_dir ./gen_scene/

# Generation, SPLIT (experiment 6 or 7)
python sample_stage2.py --objective generation \
    --stage1_checkpoint   /path/stage1_best.pth \
    --layout_checkpoint   /path/stage2_layout_best.pth \
    --geometry_checkpoint /path/stage2_geometry_best.pth \
    --num_samples 8 --num_steps 50 --output_dir ./gen_split/

# Completion (experiment 5, 6 or 7)
python sample_stage2.py --objective completion \
    --stage1_checkpoint     /path/stage1_best.pth \
    --completion_checkpoint /path/stage2_completion_best.pth \
    --num_samples 4 --coverage 0.4 --num_steps 50 --output_dir ./completed/
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.utils.data as Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gs_dataset_scenesplat as gds_module
from gs_dataset_scenesplat import gs_dataset
from gs_ply_reconstructor import save_reconstructed_gaussians

from stage2.stage1_bridge import (
    load_stage1, encode_clean, decode_latent, build_stage2_model,
    is_structured, stage1_data_kwargs,
)
from stage2.models.flat_dit import sample_block_mask


# ============================================================================
# Euler sampler (guards non-Module closures used in completion)
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
    return x


# ============================================================================
# Stage 2 model loader (rebuilds from the checkpoint's recorded metadata)
# ============================================================================

def load_stage2_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    stage      = ck["stage"]
    schema     = ck["schema"]
    structured = ck.get("structured", True)
    size       = ck["model_size"]
    rope       = ck.get("rope_type", "learned_ape")
    ed         = int(ck.get("embed_dim", 32))
    model = build_stage2_model(schema, structured, stage, size, rope, embed_dim=ed)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Stage 2 loaded: stage={stage} schema={schema} size={size} rope={rope}")
    return model, ck


# ============================================================================
# Generation
# ============================================================================

@torch.no_grad()
def generate(args, shape_model, flags, schema, device):
    B = args.num_samples
    if schema == "flat":
        assert args.scene_checkpoint, "FLAT checkpoint needs --scene_checkpoint"
        scene_model, _ = load_stage2_model(args.scene_checkpoint, device)
        print(f"Generating {B} scenes (flat, steps={args.num_steps})")
        Z_gen = euler_sample(scene_model, torch.randn(B, 512, 32, device=device),
                             num_steps=args.num_steps)
        # DC colour: Stage 1's mean_color_head reads shape_embed (not generated), so
        # recover the per-scene DC from the DCHead trained on q(DC|Z). Without a dc
        # checkpoint the colours are residual-only.
        mean_color = None
        if flags["color_residual"] and args.dc_checkpoint:
            dc_head, _ = load_stage2_model(args.dc_checkpoint, device)
            mean_color = dc_head.sample(Z_gen, mode=args.dc_mode)
            print(f"  DC colour from DCHead (mode={args.dc_mode})")
        elif flags["color_residual"]:
            print("  [warn] color_residual Stage 1 but no --dc_checkpoint: colours "
                  "are AC residual only. Train --stage dc and pass --dc_checkpoint.")
        return decode_latent(shape_model, flags, Z_gen, None, mean_color=mean_color)

    assert args.layout_checkpoint and args.geometry_checkpoint, \
        "SPLIT checkpoint needs --layout_checkpoint and --geometry_checkpoint"
    layout_model, _   = load_stage2_model(args.layout_checkpoint,   device)
    geometry_model, _ = load_stage2_model(args.geometry_checkpoint, device)
    print(f"Generating {B} scenes (split, steps={args.num_steps})")
    z_s_gen = euler_sample(layout_model, torch.randn(B, 16, 32, device=device),
                           num_steps=args.num_steps)
    z_g_gen = euler_sample(geometry_model, torch.randn(B, 512, 32, device=device),
                           num_steps=args.num_steps, z_s_clean=z_s_gen)
    # DC colour for free: lay_color_head reads z_s token 0, and z_s IS generated.
    mean_color = None
    if flags["color_residual"] and getattr(shape_model, "lay_color_head", None) is not None:
        mean_color = shape_model.lay_color_head(z_s_gen[:, 0, :])
        print("  DC colour from Stage 1 lay_color_head(z_s)")
    return decode_latent(shape_model, flags, z_g_gen, z_s_gen, mean_color=mean_color)


# ============================================================================
# Completion (mask real scenes from the dataset)
# ============================================================================

@torch.no_grad()
def complete(args, shape_model, flags, schema, device, out: Path):
    assert args.completion_checkpoint, "completion needs --completion_checkpoint"
    comp_model, _ = load_stage2_model(args.completion_checkpoint, device)

    gds_module.TARGET_POINTS = int(flags["num_gaussians"])
    ds_kwargs = stage1_data_kwargs(flags)
    val_root  = os.path.join(args.data_path, "val")   # held-out full scenes
    ds = gs_dataset(root=val_root, random_permute=False, train=True,
                    max_scenes=args.num_samples, **ds_kwargs)
    loader = Data.DataLoader(ds, batch_size=args.num_samples, shuffle=False,
                             num_workers=4, pin_memory=True)

    batch    = next(iter(loader))
    features = batch["features"].float().to(device)
    B        = features.shape[0]
    mc_raw   = batch.get("mean_color", None)
    mean_color = mc_raw[:B].to(device) if mc_raw is not None else None

    z_s, z_g = encode_clean(shape_model, features, flags, schema)   # z_s None if flat

    obs_mask = sample_block_mask(B, 512, device, (args.coverage, args.coverage))
    mask_exp = obs_mask.unsqueeze(-1)
    z_init   = z_g * mask_exp + torch.randn_like(z_g) * (1.0 - mask_exp)

    if schema == "split":
        def masked_model(x, t, **_kw):
            return comp_model(x, t, z_s, obs_mask) * (1.0 - mask_exp)
    else:
        def masked_model(x, t, **_kw):
            return comp_model(x, t, obs_mask) * (1.0 - mask_exp)

    z_comp = euler_sample(masked_model, z_init, num_steps=args.num_steps)
    z_comp = z_comp * (1.0 - mask_exp) + z_g * mask_exp           # restore observed
    z_part = z_g * mask_exp

    z_s_dec = z_s if schema == "split" else None
    for z_arr, name in [(z_comp, "completed"), (z_part, "partial"), (z_g, "gt_full")]:
        preds = decode_latent(shape_model, flags, z_arr, z_s_dec, mean_color)
        save_reconstructed_gaussians(predictions=preds, output_dir=out / name,
                                     epoch=0, num_scenes=B, max_sh_degree=3, color_mode="1")
    print(f"Saved completed/ partial/ gt_full/ ({B} scenes, coverage={args.coverage:.0%})")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Can3Tok Stage 2 Sampling (schema-aware)")
    p.add_argument("--objective", type=str, required=True,
                   choices=["generation", "completion"])
    p.add_argument("--stage1_checkpoint", type=str, required=True)
    p.add_argument("--stage1_config", type=str,
                   default="./model/configs/aligned_shape_latents/shapevae-256.yaml")
    # generation
    p.add_argument("--scene_checkpoint",    type=str, default=None)
    p.add_argument("--layout_checkpoint",   type=str, default=None)
    p.add_argument("--geometry_checkpoint", type=str, default=None)
    p.add_argument("--dc_checkpoint",       type=str, default=None,
                   help="DCHead checkpoint (FLAT generation only) to recover the DC "
                        "mean colour. Omit for residual-only colours.")
    p.add_argument("--dc_mode",             type=str, default="sample",
                   choices=["sample", "mean"],
                   help="'sample' draws DC ~ q(DC|Z) (palette diversity); "
                        "'mean' uses E[DC|Z] (deterministic).")
    # completion
    p.add_argument("--completion_checkpoint", type=str, default=None)
    p.add_argument("--coverage", type=float, default=0.4)
    p.add_argument("--data_path", type=str,
                   default="/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs",
                   help="BASE dir; completion demo scenes are read from <base>/val.")
    # shared
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--num_steps",   type=int, default=50)
    p.add_argument("--output_dir",  type=str, default="./stage2_samples")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 1 ...")
    shape_model, flags, schema = load_stage1(
        args.stage1_checkpoint, args.stage1_config, device)
    print(f"  schema={schema}  structured={is_structured(flags)}")

    if args.objective == "generation":
        preds = generate(args, shape_model, flags, schema, device)
        print(f"Saving {len(preds)} PLYs to {out} ...")
        save_reconstructed_gaussians(predictions=preds, output_dir=out, epoch=0,
                                     num_scenes=len(preds), max_sh_degree=3, color_mode="1")
    else:
        if schema == "flat" and not is_structured(flags):
            raise ValueError("Completion is not meaningful for global-flat checkpoints "
                             "(experiments 1-4): tokens are not spatial.")
        complete(args, shape_model, flags, schema, device, out)

    print("Done.")


if __name__ == "__main__":
    main()