"""
Can3Tok Training
================
Step 1:    Color Residual             (--color_residual)
Move 1:    Scene Semantic Head        (--scene_semantic_head)
Scaffold:  Position Scaffold          (--position_scaffold)
Option 1:  Decoder Shape Prepend      (--decoder_shape_prepend)    ← NEW
Option 2:  Decoder Shape Cross-Attn   (--decoder_shape_cross_attn) ← NEW
InfoNCE:   Per-Gaussian contrastive   (--semantic_mode hidden)

══════════════════════════════════════════════════════════════════
DECODER SHAPE CONDITIONING
══════════════════════════════════════════════════════════════════

OPTION 1  --decoder_shape_prepend
  shape_embed [B,384] is projected and prepended as token 0 to the
  512 geometry tokens before the decoder transformer. All 12 self-attn
  layers can directly attend to the global scene token.
  n_ctx in the transformer is already 513 — exact match.
  Extra parameters: ~300K (one Linear + LayerNorm).

OPTION 2  --decoder_shape_cross_attn  --decoder_cross_attn_layers N
  N cross-attention blocks are applied BEFORE the decoder transformer,
  conditioning the 512 geometry tokens on shape_embed. Each geometry token
  independently attends to shape_embed. The transformer then propagates
  this global context through all 12 self-attn layers.
  Extra parameters: ~N × 1.1M (each ResidualCrossAttentionBlock).
  Default N=4 (~4.4M extra params).

Both options pass shape_embed into decode() as a conditioning signal.
shape_embed is the pre-KL encoder output [B, 384], cached in encode().

STAGE 2 NOTE:
  At generation time, shape_embed must be approximated from the generated z
  tokens. A lightweight ShapeEmbedPredictor MLP (z.mean(-2) → shape_embed)
  can be trained separately. The diffusion model can also generate shape_embed
  jointly as part of the latent.

ABLATION NAMING CONVENTION:
  _shapeprepend      Option 1 only
  _shapecrossattn    Option 2 only
  _shapeprepend_shapecrossattn  both options combined
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
import torch.utils.data as Data

from semantic_losses import compute_semantic_loss
from distribution_loss import compute_distribution_loss
from pca_feature_visualization import visualize_comparison
from gs_ply_reconstructor import save_reconstructed_gaussians

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# PARAMETER INDICES
# ============================================================================

PARAM_SLICES = {
    'position': slice(0, 3),
    'color':    slice(3, 6),
    'opacity':  slice(6, 7),
    'scale':    slice(7, 10),
    'rotation': slice(10, 14),
}

GEOMETRIC_INDICES = (
    list(range(4, 7))
  + list(range(7, 10))
  + [10]
  + list(range(11, 14))
  + list(range(14, 18))
)

# ============================================================================
# LOSS HELPERS
# ============================================================================

def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0):
    if color_weight == 1.0:
        return torch.norm(prediction - target, p=2) / batch_size
    loss_pos   = torch.norm(prediction[:, :, 0:3] - target[:, :, 0:3], p=2)
    loss_color = torch.norm(prediction[:, :, 3:6] - target[:, :, 3:6], p=2) * color_weight
    loss_other = torch.norm(prediction[:, :, 6:]  - target[:, :, 6:],  p=2)
    return (loss_pos + loss_color + loss_other) / batch_size


def compute_individual_losses(prediction, target):
    return {
        name: torch.norm(prediction[:, :, sl] - target[:, :, sl], p=2).item()
        for name, sl in PARAM_SLICES.items()
    }


def scene_semantic_kl_loss(p_hat, p_s, eps=1e-8):
    p_hat_clamped = torch.clamp(p_hat, min=eps)
    return (p_s * (torch.log(p_s + eps) - torch.log(p_hat_clamped))).sum(dim=-1).mean()

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Training')

# Core
parser.add_argument('--batch_size',           type=int,   default=64)
parser.add_argument('--num_epochs',           type=int,   default=1000)
parser.add_argument('--lr',                   type=float, default=1e-4)
parser.add_argument('--kl_weight',            type=float, default=1e-5)
parser.add_argument('--eval_every',           type=int,   default=20)
parser.add_argument('--failure_threshold',    type=float, default=100.0)

# Dataset
parser.add_argument('--train_scenes',         type=int,   default=None)
parser.add_argument('--val_scenes',           type=int,   default=None)
parser.add_argument('--sampling_method',      type=str,   default='opacity',
                    choices=['random', 'opacity', 'hybrid'])

# Semantic InfoNCE
parser.add_argument('--semantic_mode',        type=str,   default='none',
                    choices=['none', 'hidden', 'geometric', 'attention', 'dist'])
parser.add_argument('--segment_loss_weight',  type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample',   type=int,   default=2000)
parser.add_argument('--sampling_strategy',    type=str,   default='balanced',
                    choices=['random', 'balanced'])

# Step 1: Color residual
parser.add_argument('--color_residual',       action='store_true', default=False)
parser.add_argument('--mean_color_weight',    type=float, default=1.0)

# Move 1: Scene semantic head
parser.add_argument('--scene_semantic_head',   action='store_true', default=False)
parser.add_argument('--scene_semantic_weight', type=float, default=0.3)

# Scaffold
parser.add_argument('--position_scaffold',     action='store_true', default=False)
parser.add_argument('--anchor_loss_weight',    type=float, default=1.0)

# ── Option 1: Decoder shape prepend ──────────────────────────────────────────
parser.add_argument('--decoder_shape_prepend', action='store_true', default=False,
    help='Option 1: Project shape_embed [B,384] and prepend as token 0 to the '
         '512 geometry tokens before the decoder transformer. '
         'Transformer sees 513 tokens (n_ctx=513, exact match). '
         'Shape token dropped after transformer. '
         'Geometry tokens can directly attend to global scene context via self-attn. '
         '~300K extra params.')

# ── Option 2: Decoder shape cross-attention ───────────────────────────────────
parser.add_argument('--decoder_shape_cross_attn', action='store_true', default=False,
    help='Option 2: Apply N cross-attention blocks BEFORE the decoder transformer. '
         'Geometry tokens attend to shape_embed as a single key/value token. '
         'All 12 transformer self-attn layers then process shape-conditioned tokens. '
         'N controlled by --decoder_cross_attn_layers. ~N×1.1M extra params.')
parser.add_argument('--decoder_cross_attn_layers', type=int, default=4,
    help='Number of cross-attention layers for Option 2 (default 4). '
         'Each adds ~1.1M params. Typical range: 2-6. '
         'Higher = richer conditioning but more memory and compute.')

# Label input
parser.add_argument('--label_input',          action='store_true', default=False)
parser.add_argument('--no_label_input',       dest='label_input', action='store_false')

# Normalization
parser.add_argument('--scale_norm_mode',      type=str, default='linear',
                    choices=['log', 'linear'])
parser.add_argument('--color_loss_weight',    type=float, default=1.0)
norm_grp = parser.add_mutually_exclusive_group()
norm_grp.add_argument('--use_canonical_norm', dest='use_canonical_norm',
                      action='store_true', default=True)
norm_grp.add_argument('--no_canonical_norm',  dest='use_canonical_norm',
                      action='store_false')
color_norm_grp = parser.add_mutually_exclusive_group()
color_norm_grp.add_argument('--normalize_colors',    dest='normalize_colors',
                            action='store_true', default=True)
color_norm_grp.add_argument('--no_normalize_colors', dest='normalize_colors',
                            action='store_false')

# Visualization
parser.add_argument('--pca_vis_freq',         type=int,   default=50)
parser.add_argument('--pca_brightness',       type=float, default=1.25)
parser.add_argument('--pca_num_scenes',       type=int,   default=3)
parser.add_argument('--recon_ply_freq',       type=int,   default=50)
parser.add_argument('--recon_ply_num_scenes', type=int,   default=3)
parser.add_argument('--recon_ply_max_sh',     type=int,   default=3)

# W&B
parser.add_argument('--use_wandb',            action='store_true', default=False)
parser.add_argument('--wandb_project',        type=str, default='Can3Tok-SceenSplat-7K')
parser.add_argument('--wandb_entity',         type=str, default='3D-SSC')

# Resuming
parser.add_argument('--resume_checkpoint',    type=str, default=None)
parser.add_argument('--resume_epoch',         type=int, default=None)

args = parser.parse_args()

semantic_requested      = (args.semantic_mode != 'none')
semantic_loss_enabled   = (args.segment_loss_weight > 0 or args.instance_loss_weight > 0)
enable_semantic         = semantic_requested and semantic_loss_enabled
effective_semantic_mode = args.semantic_mode if enable_semantic else 'none'
need_segment_labels     = enable_semantic or args.scene_semantic_head

# ============================================================================
# W&B
# ============================================================================

wandb_enabled = False
if args.use_wandb:
    try:
        import wandb
        job_id   = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"can3tok_job_{job_id}_{effective_semantic_mode}"
        if args.color_residual:             run_name += "_colorresidual"
        if args.scene_semantic_head:        run_name += "_scenesemantic"
        if args.position_scaffold:          run_name += "_scaffold"
        if args.decoder_shape_prepend:      run_name += "_shapeprepend"
        if args.decoder_shape_cross_attn:   run_name += "_shapecrossattn"
        if args.label_input:                run_name += "_labelinput"
        if enable_semantic:                 run_name += f"_beta{args.segment_loss_weight}"
        wandb_run = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            name=run_name, config=vars(args))
        wandb_enabled = True
    except Exception as e:
        print(f"W&B failed: {e}")

# ============================================================================
# DEVICE + PATHS
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = "/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs"

job_id = os.environ.get('SLURM_JOB_ID', None)
tag    = (f"RGB_job_{job_id}_{effective_semantic_mode}" if job_id
          else f"RGB_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{effective_semantic_mode}")
if args.color_residual:             tag += "_colorresidual"
if args.scene_semantic_head:        tag += "_scenesemantic"
if args.position_scaffold:          tag += "_scaffold"
if args.decoder_shape_prepend:      tag += "_shapeprepend"
if args.decoder_shape_cross_attn:   tag += "_shapecrossattn"
if args.label_input:                tag += "_labelinput"
if enable_semantic:                 tag += f"_beta{args.segment_loss_weight}"
if not args.use_canonical_norm:     tag += "_raw"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"CAN3TOK TRAINING")
print(f"{'='*70}")
print(f"  Semantic mode:              {effective_semantic_mode}")
print(f"  Color residual:             {args.color_residual}  (Step 1 — MeanColorHead)")
print(f"  Scene semantic head:        {args.scene_semantic_head}  (Move 1)")
print(f"  Position scaffold:          {args.position_scaffold}  (Scaffold-GS)")
print(f"  Decoder shape prepend:      {args.decoder_shape_prepend}  (Option 1)")
print(f"  Decoder shape cross-attn:   {args.decoder_shape_cross_attn}  (Option 2)")
if args.decoder_shape_cross_attn:
    print(f"  Cross-attn layers:          {args.decoder_cross_attn_layers}")
if args.decoder_shape_prepend:
    print(f"\n  OPTION 1 ACTIVE:")
    print(f"    shape_embed → project → prepend as token 0 in decoder transformer")
    print(f"    Transformer: [shape_token | 512 geometry tokens] = 513 tokens (n_ctx=513 ✓)")
    print(f"    shape token dropped after transformer")
if args.decoder_shape_cross_attn:
    print(f"\n  OPTION 2 ACTIVE:")
    print(f"    {args.decoder_cross_attn_layers} cross-attn layers BEFORE decoder transformer")
    print(f"    geometry tokens attend to shape_embed (single key/value)")
    print(f"    all 12 transformer layers process shape-conditioned tokens")
print(f"  Save path: {save_path}")
print(f"{'='*70}\n")

# ============================================================================
# MODEL
# ============================================================================

print("Loading model config...")
config_path  = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config = get_config_from_file(config_path).model
p = model_config.params.shape_module_cfg.params
p.semantic_mode              = effective_semantic_mode
p.color_residual             = args.color_residual
p.scene_semantic_head        = args.scene_semantic_head
p.position_scaffold          = args.position_scaffold
p.decoder_shape_prepend      = args.decoder_shape_prepend      # Option 1
p.decoder_shape_cross_attn   = args.decoder_shape_cross_attn   # Option 2
p.decoder_cross_attn_layers  = args.decoder_cross_attn_layers  # Option 2 depth

cfg_point_feats = p.point_feats
expected_feats  = 12 if args.label_input else 11
if cfg_point_feats != expected_feats:
    raise ValueError(f"point_feats mismatch: yaml={cfg_point_feats}, "
                     f"label_input={args.label_input} requires {expected_feats}.")
print(f"  point_feats={cfg_point_feats} OK")

gs_autoencoder = instantiate_from_config(model_config)
gs_autoencoder.to(device)
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999])

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

start_epoch   = 0
best_val_loss = float('inf')
best_epoch    = 0

if args.resume_checkpoint:
    print(f"\nResuming from: {args.resume_checkpoint}")
    ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

    # Architecture flags that change model structure — must match exactly
    for flag_name, current_val in [
        ('color_residual',            args.color_residual),
        ('label_input',               args.label_input),
        ('position_scaffold',         args.position_scaffold),
        ('decoder_shape_prepend',     args.decoder_shape_prepend),
        ('decoder_shape_cross_attn',  args.decoder_shape_cross_attn),
        ('decoder_cross_attn_layers', args.decoder_cross_attn_layers),
    ]:
        saved_val = ckpt.get(flag_name, False if flag_name != 'decoder_cross_attn_layers' else 4)
        if saved_val != current_val:
            raise ValueError(
                f"{flag_name} mismatch: checkpoint={saved_val}, current={current_val}. "
                f"This flag changes model architecture — cannot resume across it.")

    saved_ssh  = ckpt.get('scene_semantic_head', False)
    saved_mode = ckpt.get('semantic_mode', 'none')
    strict = (saved_ssh == args.scene_semantic_head and
              saved_mode == effective_semantic_mode)
    if not strict:
        print(f"  Architecture changed, loading strict=False")
    gs_autoencoder.load_state_dict(ckpt['model_state_dict'], strict=strict)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch   = ckpt.get('epoch', 0) + 1
    if args.resume_epoch is not None:
        start_epoch = args.resume_epoch
    best_val_loss = ckpt.get('val_l2_error', ckpt.get('best_val_l2', float('inf')))
    best_epoch    = ckpt.get('epoch', 0)
    print(f"  Resumed at epoch {start_epoch} (saved val L2: {best_val_loss:.4f})")

# ============================================================================
# DATASETS
# ============================================================================

from gs_dataset_scenesplat import gs_dataset

print(f"\n--- Training Dataset ---")
gs_dataset_train = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=200, random_permute=True, train=True,
    sampling_method=args.sampling_method, max_scenes=args.train_scenes,
    normalize=args.use_canonical_norm, normalize_colors=args.normalize_colors,
    target_radius=10.0, scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input, color_residual=args.color_residual,
    position_scaffold=args.position_scaffold)
trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_train, batch_size=args.batch_size,
    shuffle=True, num_workers=9, pin_memory=True, persistent_workers=True)

print(f"\n--- Validation Dataset ---")
gs_dataset_val = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=200, random_permute=False, train=True,
    sampling_method=args.sampling_method, max_scenes=args.val_scenes,
    normalize=args.use_canonical_norm, normalize_colors=args.normalize_colors,
    target_radius=10.0, scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input, color_residual=args.color_residual,
    position_scaffold=args.position_scaffold)
valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val, batch_size=args.batch_size,
    shuffle=False, num_workers=9, pin_memory=True, persistent_workers=True)

print(f"\n{'='*70}")
print(f"  Train: {len(gs_dataset_train)} scenes, {len(trainDataLoader)} batches/epoch")
print(f"  Val:   {len(gs_dataset_val)} scenes,  {len(valDataLoader)} batches")
print(f"  Total steps: {len(trainDataLoader) * args.num_epochs:,}")
print(f"{'='*70}\n")

# ============================================================================
# CHECKPOINT METADATA
# ============================================================================

_ckpt_meta = {
    'semantic_mode':             effective_semantic_mode,
    'enable_semantic':           enable_semantic,
    'label_input':               args.label_input,
    'color_residual':            args.color_residual,
    'scene_semantic_head':       args.scene_semantic_head,
    'position_scaffold':         args.position_scaffold,
    'decoder_shape_prepend':     args.decoder_shape_prepend,      # Option 1
    'decoder_shape_cross_attn':  args.decoder_shape_cross_attn,   # Option 2
    'decoder_cross_attn_layers': args.decoder_cross_attn_layers,  # Option 2 depth
    'mean_color_weight':         args.mean_color_weight,
    'scene_semantic_weight':     args.scene_semantic_weight,
    'anchor_loss_weight':        args.anchor_loss_weight,
    'color_loss_weight':         args.color_loss_weight,
    'use_canonical_norm':        args.use_canonical_norm,
    'scale_norm_mode':           args.scale_norm_mode,
}

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device, epoch=None):
    model.eval()
    total_l2 = total_kl = 0.0
    total_color_pred_loss   = 0.0
    total_scene_semantic_kl = 0.0
    total_anchor_loss       = 0.0
    per_param  = {k: 0.0 for k in PARAM_SLICES}
    n_scenes   = 0
    recon_preds_list = []
    recon_means_list = []
    recon_tids_list  = []
    recon_anch_list  = []
    do_recon = (epoch is not None and epoch % args.recon_ply_freq == 0)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            UV_gs_batch   = batch_data['features'].float().to(device)
            mean_color_gt = batch_data['mean_color'].float().to(device)
            B = UV_gs_batch.shape[0]

            (shape_embed, mu, log_var, z,
             UV_gs_recover, _) = model(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3])

            mean_color_pred     = model.shape_model.last_mean_color_pred
            scene_semantic_pred = model.shape_model.last_scene_semantic_pred
            anchor_pred         = model.shape_model.last_anchor_pred

            target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            if args.position_scaffold:
                position_offsets = batch_data['position_offsets'].float().to(device)
                target = target_abs.clone()
                target[:, :, 0:3] = position_offsets
            else:
                target = target_abs

            pred_3d    = UV_gs_recover.reshape(B, -1, 14)
            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
            kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            if mean_color_pred is not None and args.color_residual:
                total_color_pred_loss += F.mse_loss(mean_color_pred, mean_color_gt).item() * B
            if scene_semantic_pred is not None and args.scene_semantic_head:
                p_s = batch_data['label_dist'].float().to(device)
                total_scene_semantic_kl += scene_semantic_kl_loss(
                    scene_semantic_pred, p_s).item() * B
            if anchor_pred is not None and args.position_scaffold:
                scaffold_anchors = batch_data['scaffold_anchors'].float().to(device)
                total_anchor_loss += F.mse_loss(anchor_pred, scaffold_anchors).item() * B

            total_l2 += recon_loss.item()
            total_kl += kl_loss.sum().item()
            n_scenes  += B
            ind = compute_individual_losses(pred_3d, target)
            for k in per_param:
                per_param[k] += ind[k]

            if do_recon and len(recon_preds_list) < args.recon_ply_num_scenes:
                preds_np = pred_3d.cpu().numpy()
                means_np = mean_color_gt.cpu().numpy()
                tids_np  = batch_data['scaffold_token_ids'].numpy()
                anch_np  = (anchor_pred.cpu().numpy() if anchor_pred is not None
                            else batch_data['scaffold_anchors'].numpy())
                for si in range(B):
                    if len(recon_preds_list) >= args.recon_ply_num_scenes:
                        break
                    recon_preds_list.append(preds_np[si])
                    recon_means_list.append(means_np[si])
                    recon_tids_list.append(tids_np[si])
                    recon_anch_list.append(anch_np[si])

    if do_recon and recon_preds_list and save_path:
        try:
            all_preds = np.stack(recon_preds_list, axis=0)
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si, :, 3:6] += recon_means_list[si]
                    all_preds[si, :, 3:6]  = np.clip(all_preds[si, :, 3:6], 0, 1)
            if args.position_scaffold:
                for si in range(len(all_preds)):
                    all_preds[si, :, 0:3] += recon_anch_list[si][recon_tids_list[si]]
            recon_dir = Path(save_path) / "reconstructed_gaussians" / f"epoch_{epoch:03d}"
            save_reconstructed_gaussians(
                predictions=all_preds, output_dir=recon_dir, epoch=epoch,
                num_scenes=len(all_preds), max_sh_degree=args.recon_ply_max_sh,
                color_mode="1", prefix="scene")
        except Exception as e:
            print(f"  PLY save error: {e}")

    model.train()
    n = max(n_scenes, 1)
    return {
        'avg_l2_error':      total_l2,
        'avg_kl':            total_kl / n,
        'color_pred_loss':   total_color_pred_loss / n,
        'scene_semantic_kl': total_scene_semantic_kl / n,
        'anchor_loss':       total_anchor_loss / n,
        **{f'{k}_loss': v / n for k, v in per_param.items()},
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"{'='*70}")
print(f"STARTING TRAINING  (epoch {start_epoch} -> {args.num_epochs - 1})")
print(f"{'='*70}\n")

global_step = 0

for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Training"):
    gs_autoencoder.train()

    epoch_loss = epoch_recon = epoch_kl = epoch_sem = 0.0
    epoch_color_pred = epoch_scene_semantic = epoch_anchor = 0.0
    epoch_pos = epoch_col = epoch_opa = epoch_scl = epoch_rot = 0.0

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)
        B = UV_gs_batch.shape[0]

        segment_labels  = None
        instance_labels = None
        if need_segment_labels:
            segment_labels  = batch_data['segment_labels'].long().to(device)
            if enable_semantic:
                instance_labels = batch_data['instance_labels'].long().to(device)

        if args.position_scaffold:
            scaffold_anchors   = batch_data['scaffold_anchors'].float().to(device)
            scaffold_token_ids = batch_data['scaffold_token_ids'].long().to(device)
            position_offsets   = batch_data['position_offsets'].float().to(device)

        optimizer.zero_grad()

        # ── Forward ──────────────────────────────────────────────────────────
        # shape_embed is now used inside the decoder (Options 1/2) via
        # self._shape_embed_cache which is set in encode() and accessed in
        # the overridden decode() in AlignedShapeLatentPerceiver.
        (shape_embed, mu, log_var, z,
         UV_gs_recover, per_gaussian_features) = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3])

        mean_color_pred     = gs_autoencoder.shape_model.last_mean_color_pred
        scene_semantic_pred = gs_autoencoder.shape_model.last_scene_semantic_pred
        anchor_pred         = gs_autoencoder.shape_model.last_anchor_pred

        # ── Reconstruction target ─────────────────────────────────────────────
        target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        if args.position_scaffold:
            target = target_abs.clone()
            target[:, :, 0:3] = position_offsets
        else:
            target = target_abs

        pred_3d    = UV_gs_recover.reshape(B, -1, 14)
        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)

        # ── KL loss ───────────────────────────────────────────────────────────
        KL_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        # ── Auxiliary losses ──────────────────────────────────────────────────
        color_pred_loss = torch.tensor(0.0, device=device)
        if mean_color_pred is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mean_color_pred, mean_color_gt)

        scene_semantic_loss = torch.tensor(0.0, device=device)
        if scene_semantic_pred is not None and args.scene_semantic_head:
            p_s = batch_data['label_dist'].float().to(device)
            scene_semantic_loss = scene_semantic_kl_loss(scene_semantic_pred, p_s)

        anchor_loss = torch.tensor(0.0, device=device)
        if anchor_pred is not None and args.position_scaffold:
            anchor_loss = F.mse_loss(anchor_pred, scaffold_anchors)

        # ── InfoNCE ────────────────────────────────────────────────────────────
        semantic_loss    = torch.tensor(0.0, device=device)
        semantic_metrics = {}
        if enable_semantic and segment_labels is not None and per_gaussian_features is not None:
            if args.semantic_mode == 'dist':
                semantic_loss, semantic_metrics = compute_distribution_loss(
                    dist_logits=per_gaussian_features,
                    segment_labels=segment_labels,
                    weight=args.segment_loss_weight)
            else:
                semantic_loss, semantic_metrics = compute_semantic_loss(
                    embeddings=per_gaussian_features,
                    segment_labels=segment_labels,
                    instance_labels=instance_labels,
                    batch_size=B,
                    segment_weight=args.segment_loss_weight,
                    instance_weight=args.instance_loss_weight,
                    temperature=args.semantic_temperature,
                    subsample=args.semantic_subsample,
                    sampling_strategy=args.sampling_strategy)

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (recon_loss
                + args.kl_weight             * KL_loss
                + args.mean_color_weight     * color_pred_loss
                + args.scene_semantic_weight * scene_semantic_loss
                + args.anchor_loss_weight    * anchor_loss
                + semantic_loss)
        loss.backward()
        optimizer.step()

        ind = compute_individual_losses(pred_3d, target)
        epoch_loss           += loss.item()
        epoch_recon          += recon_loss.item()
        epoch_kl             += KL_loss.item()
        epoch_sem            += semantic_loss.item()
        epoch_color_pred     += color_pred_loss.item()
        epoch_scene_semantic += scene_semantic_loss.item()
        epoch_anchor         += anchor_loss.item()
        epoch_pos += ind['position']
        epoch_col += ind['color']
        epoch_opa += ind['opacity']
        epoch_scl += ind['scale']
        epoch_rot += ind['rotation']

        # Epoch 0 first-batch diagnostic
        if epoch == start_epoch and i_batch == 0:
            print(f"\nEPOCH {epoch} DIAGNOSTIC (batch 0):")
            print(f"  mu range:    [{mu.min().item():.3f}, {mu.max().item():.3f}]")
            print(f"  recon_loss:  {recon_loss.item():.4f}")
            if args.color_residual and mean_color_pred is not None:
                print(f"  color_pred:  {color_pred_loss.item():.6f}")
            if args.position_scaffold and anchor_pred is not None:
                print(f"  anchor_loss: {anchor_loss.item():.4f}")
            # Shape conditioning diagnostic: verify decode sees shape_embed
            if args.decoder_shape_prepend or args.decoder_shape_cross_attn:
                print(f"  shape_embed range: [{shape_embed.min().item():.3f}, "
                      f"{shape_embed.max().item():.3f}]  (conditioning decoder)")
                # A quick sanity check: reconstruction loss with shape conditioning
                # should not be worse than without at epoch 0 (they're both random)
                print(f"  decoder conditioning: "
                      f"{'prepend+' if args.decoder_shape_prepend else ''}"
                      f"{'cross-attn' if args.decoder_shape_cross_attn else ''} active")

        if wandb_enabled:
            log = {
                "train/step_loss":           loss.item(),
                "train/step_recon":          recon_loss.item(),
                "train/step_kl":             KL_loss.item(),
                "train/step_color_pred":     color_pred_loss.item(),
                "train/step_scene_semantic": scene_semantic_loss.item(),
                "train/step_anchor":         anchor_loss.item(),
                "train/step_position":       ind['position'],
                "train/step_color":          ind['color'],
                "train/step_opacity":        ind['opacity'],
                "train/step_scale":          ind['scale'],
                "train/step_rotation":       ind['rotation'],
            }
            if semantic_metrics:
                log.update({f"train/step_{k}": v for k, v in semantic_metrics.items()})
            wandb_run.log(log, step=global_step)

        global_step += 1

    # ── End-of-epoch logging ──────────────────────────────────────────────────
    nb = len(trainDataLoader)
    print(f"\nEpoch {epoch} | Loss={epoch_loss/nb:.4f} | Recon={epoch_recon/nb:.4f} | "
          f"KL={epoch_kl/nb:.4f} | InfoNCE={epoch_sem/nb:.4f} | "
          f"ColorPred={epoch_color_pred/nb:.6f} | SceneSem={epoch_scene_semantic/nb:.4f} | "
          f"Anchor={epoch_anchor/nb:.4f}")
    print(f"  Pos={epoch_pos/nb:.3f} | Col={epoch_col/nb:.3f} | "
          f"Opa={epoch_opa/nb:.3f} | Scl={epoch_scl/nb:.3f} | Rot={epoch_rot/nb:.3f}")

    # ── Validation ────────────────────────────────────────────────────────────
    val_metrics = None
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
        print(f"\n--- Validation (epoch {epoch}) ---")
        val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, epoch=epoch)
        print(f"  L2:              {val_metrics['avg_l2_error']:.4f}")
        print(f"  Position:        {val_metrics['position_loss']:.4f}"
              f"  {'(offsets)' if args.position_scaffold else '(absolute)'}")
        print(f"  Color:           {val_metrics['color_loss']:.4f}"
              f"  {'(residuals)' if args.color_residual else '(absolute)'}")
        print(f"  Opacity:         {val_metrics['opacity_loss']:.4f}")
        print(f"  Scale:           {val_metrics['scale_loss']:.4f}")
        print(f"  Rotation:        {val_metrics['rotation_loss']:.4f}")
        if args.color_residual:
            print(f"  ColorPredMSE:    {val_metrics['color_pred_loss']:.6f}")
        if args.scene_semantic_head:
            print(f"  SceneSemanticKL: {val_metrics['scene_semantic_kl']:.4f}")
        if args.position_scaffold:
            print(f"  AnchorMSE:       {val_metrics['anchor_loss']:.4f}")

        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch    = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     gs_autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_l2_error':         val_metrics['avg_l2_error'],
                **_ckpt_meta,
            }, os.path.join(save_path, "best_model.pth"))
            print(f"  [NEW BEST] L2={best_val_loss:.4f} saved")

    if wandb_enabled and val_metrics:
        wandb_run.log({
            "val/l2_error":          val_metrics['avg_l2_error'],
            "val/position_loss":     val_metrics['position_loss'],
            "val/color_loss":        val_metrics['color_loss'],
            "val/color_pred_mse":    val_metrics['color_pred_loss'],
            "val/scene_semantic_kl": val_metrics['scene_semantic_kl'],
            "val/anchor_mse":        val_metrics['anchor_loss'],
            "val/opacity_loss":      val_metrics['opacity_loss'],
            "val/scale_loss":        val_metrics['scale_loss'],
            "val/rotation_loss":     val_metrics['rotation_loss'],
            "best/val_l2":           best_val_loss,
            "best/epoch":            best_epoch,
            "train/epoch":           epoch,
        }, step=global_step)

    if epoch >= 10 and epoch % 50 == 0:
        torch.save({
            'epoch':      epoch,
            'model_state_dict':     gs_autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss / nb,
            **_ckpt_meta,
        }, os.path.join(save_path, f"epoch_{epoch}.pth"))
        print(f"  Checkpoint saved: epoch_{epoch}.pth")

# ============================================================================
# FINAL SAVE
# ============================================================================

print(f"\n{'='*70}\nTRAINING COMPLETE\n{'='*70}")
final_metrics = evaluate_model(gs_autoencoder, valDataLoader, device,
                               epoch=args.num_epochs - 1)
print(f"\nFinal L2:  {final_metrics['avg_l2_error']:.4f}")
print(f"Best L2:   {best_val_loss:.4f} (epoch {best_epoch})")

torch.save({
    'epoch':        args.num_epochs - 1,
    'model_state_dict':     gs_autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_l2': final_metrics['avg_l2_error'],
    'best_val_l2':  best_val_loss,
    'best_epoch':   best_epoch,
    **_ckpt_meta,
    'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
}, os.path.join(save_path, "final.pth"))

print(f"\nSaved: {save_path}final.pth")
if wandb_enabled:
    wandb_run.summary.update({
        "final_val_l2": final_metrics['avg_l2_error'],
        "best_val_l2":  best_val_loss, "best_epoch": best_epoch})
    wandb_run.finish()
print("Done.")