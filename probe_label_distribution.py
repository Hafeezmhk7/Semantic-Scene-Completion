"""
probe_label_distribution.py
============================
Phase 1: Probing Experiment — Freeze VAE, Train Distribution Prediction Head

WHAT THIS DOES:
  - Loads a frozen Can3Tok VAE checkpoint (baseline OR label_input)
  - Auto-detects label_input from checkpoint metadata — no manual flag needed
  - Encodes all scenes → extracts mu [16384] per scene
  - Computes ground-truth label distribution p_s [72] from segment.npy
  - Trains a small MLP head to predict p_s from frozen mu
  - Evaluates per-label MAE, Recall, Precision, F1

KEY FIX vs OLD VERSION:
  Old probe always built 18-col tensors → wrong for label_input checkpoints
  (encoder input_proj is Linear(12,384) not Linear(11,384) → dimension crash)

  New probe:
    1. Reads checkpoint['label_input'] (True/False)
    2. Sets point_feats=12 in model config before instantiating
    3. SceneChunkDataset appends normalized label col 18 when label_input=True
    4. extract_mu feeds (40000,19) tensors — matching training exactly

ARCHITECTURE (probe head, VAE frozen):
  mu [16384] → Linear(16384→512) → LayerNorm → ReLU
             → Linear(512→256)   → LayerNorm → ReLU  → h [256]
             → Linear(256→72)    → Softmax           → p̂ [72]
  Loss: KL divergence D_KL(p_s || p̂_s)  (= soft cross-entropy)

USAGE:
  python probe_label_distribution.py \\
      --checkpoint /path/to/best_model.pth \\
      --data_dir   /path/to/train_grid1.0cm_chunk8x8_stride6x6 \\
      --n_total 300 --n_val 50 \\
      --epochs 100 --lr 3e-4 \\
      --output_dir ./probe_results

  label_input is auto-detected. No --label_input flag needed.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Probe VAE latent for label distribution')

parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--data_dir',   type=str, required=True)
parser.add_argument('--n_total',    type=int, default=300)
parser.add_argument('--n_val',      type=int, default=50)
parser.add_argument('--proj_hidden',type=int, default=128)
parser.add_argument('--proj_out',   type=int, default=64)
parser.add_argument('--epochs',     type=int, default=100)
parser.add_argument('--lr',         type=float, default=3e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--presence_thresh', type=float, default=0.05)
parser.add_argument('--min_label_freq',  type=int,   default=3)
parser.add_argument('--sampling_method', type=str, default='opacity')
parser.add_argument('--scale_norm_mode', type=str, default='linear')
parser.add_argument('--target_radius',   type=float, default=10.0)
parser.add_argument('--normalize_colors', action='store_true', default=True)
parser.add_argument('--output_dir', type=str, default='./probe_results')
parser.add_argument('--run_name',   type=str, default=None)

args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt_name  = Path(args.checkpoint).stem
run_name   = args.run_name or f'probe_{ckpt_name}'
output_dir = Path(args.output_dir) / run_name
output_dir.mkdir(parents=True, exist_ok=True)

n_train = args.n_total - args.n_val
NUM_LABELS = 72

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CHECKPOINT METADATA FIRST
# We need label_input BEFORE instantiating the model so we can set the right
# point_feats in the config. Without this, load_state_dict crashes with a
# shape mismatch on input_proj.weight.
# ─────────────────────────────────────────────────────────────────────────────

print(f'\n{"="*70}')
print(f'LABEL DISTRIBUTION PROBING EXPERIMENT')
print(f'{"="*70}')
print(f'  Checkpoint:  {args.checkpoint}')
print(f'  Data dir:    {args.data_dir}')
print(f'  Scenes:      total={args.n_total}  head_train={n_train}  head_val={args.n_val}')
print(f'  Output:      {output_dir}')

print('\nReading checkpoint metadata...')
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

ckpt_label_input    = checkpoint.get('label_input',    False)
ckpt_semantic_mode  = checkpoint.get('semantic_mode',  'none')
ckpt_scale_norm     = checkpoint.get('scale_norm_mode', args.scale_norm_mode)
ckpt_canonical_norm = checkpoint.get('use_canonical_norm', True)
ckpt_epoch          = checkpoint.get('epoch', '?')
ckpt_val_l2         = checkpoint.get('val_l2_error', checkpoint.get('best_val_l2', '?'))
ckpt_position_scale = checkpoint.get('position_scale', 1.0)

# Feature width: 19 cols if label_input, else 18
feature_width  = 19 if ckpt_label_input else 18
# point_feats:   12 if label_input, else 11
# (this is what input_proj receives: cols 7 onward of the feature tensor)
point_feats    = 12 if ckpt_label_input else 11

print(f'\n  ── Checkpoint metadata ──────────────────────────────────────────')
print(f'  label_input:      {ckpt_label_input}')
print(f'  semantic_mode:    {ckpt_semantic_mode}')
print(f'  scale_norm_mode:  {ckpt_scale_norm}')
print(f'  canonical_norm:   {ckpt_canonical_norm}')
print(f'  position_scale:   {ckpt_position_scale}')
print(f'  saved epoch:      {ckpt_epoch}')
print(f'  val L2:           {ckpt_val_l2}')
print(f'  ── Derived settings ─────────────────────────────────────────────')
print(f'  feature_width:    {feature_width} cols  (dataset tensor shape)')
print(f'  point_feats:      {point_feats}          (encoder input_proj dim)')
print(f'  ─────────────────────────────────────────────────────────────────')

if ckpt_label_input:
    print('\n  ✓  LABEL_INPUT mode detected — dataset will append normalized')
    print('     ScanNet72 label as col 18. Encoder expects Linear(12, d_model).')
else:
    print('\n  ✓  BASELINE mode — 18-col dataset. Encoder expects Linear(11, d_model).')

print(f'{"="*70}\n')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD VAE MODEL  (point_feats set from checkpoint before instantiation)
# ─────────────────────────────────────────────────────────────────────────────

print('Loading VAE model...')
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

config_path  = './model/configs/aligned_shape_latents/shapevae-256.yaml'
model_config = get_config_from_file(config_path).model

# ── CRITICAL: set point_feats and semantic_mode BEFORE instantiation ──────────
model_config.params.shape_module_cfg.params.semantic_mode = ckpt_semantic_mode
model_config.params.shape_module_cfg.params.point_feats   = point_feats
print(f'  Config: point_feats={point_feats}, semantic_mode={ckpt_semantic_mode}')

vae = instantiate_from_config(model_config)
vae.to(device)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

for p in vae.parameters():
    p.requires_grad_(False)

n_params = sum(p.numel() for p in vae.parameters())
print(f'  VAE: {n_params/1e6:.1f}M parameters — ALL FROZEN')
print(f'  input_proj: Linear({point_feats}, d_model)  ✓')

# ─────────────────────────────────────────────────────────────────────────────
# MU EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

GEOMETRIC_INDICES = (list(range(4,7)) + list(range(7,10))
                     + [10] + list(range(11,14)) + list(range(14,18)))

@torch.no_grad()
def extract_mu(batch_features: torch.Tensor) -> torch.Tensor:
    """Run frozen encoder, return mu [B, 16384]."""
    _, mu, _, _, _, _ = vae(
        batch_features, batch_features,
        batch_features, batch_features[:, :, :3]
    )
    return mu

# ─────────────────────────────────────────────────────────────────────────────
# SCENE CHUNK DATASET
# Key fix: appends normalized label col when label_input=True, matching
# exactly what gs_dataset_scenesplat.py does during training.
# ─────────────────────────────────────────────────────────────────────────────

class SceneChunkDataset(Dataset):
    """
    Loads raw Gaussian scene files and builds feature tensors that match
    the training configuration exactly.

    label_input=False (baseline):  feature shape (40000, 18)
    label_input=True  (Option 1):  feature shape (40000, 19)
                                   col 18 = segment_label / 71.0
                                   missing (-1) → -1/71 ≈ -0.0141
    """
    TARGET_POINTS = 40_000
    LABEL_MAX     = 71.0
    LABEL_MISSING = -1.0 / 71.0   # distinguishable null token

    def __init__(self, scene_dirs, label_input=False,
                 sampling_method='opacity', scale_norm_mode='linear',
                 target_radius=10.0, normalize_colors=True):
        self.scene_dirs     = scene_dirs
        self.label_input    = label_input
        self.sampling_method = sampling_method
        self.scale_norm_mode = scale_norm_mode
        self.target_radius   = target_radius
        self.normalize_colors = normalize_colors
        self.feature_width   = 19 if label_input else 18

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        from gs_dataset_scenesplat import normalize_to_canonical_sphere, voxelize

        scene_dir = self.scene_dirs[idx]

        coord   = np.load(os.path.join(scene_dir, 'coord.npy'))
        color   = np.load(os.path.join(scene_dir, 'color.npy'))
        scale   = np.load(os.path.join(scene_dir, 'scale.npy'))
        quat    = np.load(os.path.join(scene_dir, 'quat.npy'))
        opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))

        try:
            segment  = np.load(os.path.join(scene_dir, 'segment.npy'))
            instance = np.load(os.path.join(scene_dir, 'instance.npy'))
        except FileNotFoundError:
            segment  = np.full(len(coord), -1, dtype=np.int16)
            instance = np.full(len(coord), -1, dtype=np.int32)

        coord, scale = normalize_to_canonical_sphere(
            coord, scale,
            target_radius=self.target_radius,
            scale_norm_mode=self.scale_norm_mode,
        )

        if self.normalize_colors:
            color = color / 255.0

        N = len(coord)
        T = self.TARGET_POINTS
        if N >= T:
            selected = np.argsort(opacity)[-T:]
        else:
            extra    = np.full(T - N, np.argsort(opacity)[-1], dtype=np.int64)
            selected = np.concatenate([np.argsort(opacity), extra])

        coord   = coord   [selected]
        color   = color   [selected]
        scale   = scale   [selected]
        quat    = quat    [selected]
        opacity = opacity [selected]
        segment = segment [selected]

        volume_dims = 40
        resolution  = 16.0 / volume_dims
        origin_offset = np.array([(volume_dims-1)/2]*3) * resolution
        shifted = coord + origin_offset
        vi = np.clip(np.floor(shifted / resolution), 0, volume_dims-1)
        voxel_centers = (vi - (volume_dims-1)/2) * resolution

        uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')
        point_uniq_col = uniq_idx[inv_idx][:, np.newaxis]
        opacity_col    = opacity[:, np.newaxis]

        gs_params = np.concatenate((coord, color, opacity_col, scale, quat), axis=1)
        features  = np.concatenate((voxel_centers, point_uniq_col, gs_params), axis=1)
        # features: (40000, 18)  — voxel_centers(3) + uniq(1) + xyz(3) + rgb(3)
        #                          + opacity(1) + scale(3) + quat(4) = 18

        # ── Option 1: append normalized label as col 18 ───────────────────────
        if self.label_input:
            label_norm = np.where(
                segment >= 0,
                segment.astype(np.float32) / self.LABEL_MAX,
                self.LABEL_MISSING,
            ).astype(np.float32)[:, np.newaxis]          # (40000, 1)
            features = np.concatenate((features, label_norm), axis=1)
            # features: (40000, 19)
        # ─────────────────────────────────────────────────────────────────────

        assert features.shape == (T, self.feature_width), \
            f"Feature shape {features.shape} != ({T}, {self.feature_width})"

        return {
            'features':       features.astype(np.float32),
            'segment_labels': segment,   # raw labels for p_s computation
        }


# ─────────────────────────────────────────────────────────────────────────────
# LABEL DISTRIBUTION DATASET
# Encodes scenes with frozen VAE, stores mu + p_s in memory.
# ─────────────────────────────────────────────────────────────────────────────

class LabelDistributionDataset(Dataset):
    def __init__(self, scene_dirs, vae_model, device, batch_size=4,
                 label_input=False, sampling_method='opacity',
                 scale_norm_mode='linear', target_radius=10.0,
                 normalize_colors=True):

        self.mu_list   = []
        self.p_s_list  = []
        self.scene_ids = []

        print(f'  Pre-encoding {len(scene_dirs)} scenes '
              f'(label_input={label_input}, feature_width={19 if label_input else 18})...')
        sys.stdout.flush()

        raw_dataset = SceneChunkDataset(
            scene_dirs,
            label_input=label_input,
            sampling_method=sampling_method,
            scale_norm_mode=scale_norm_mode,
            target_radius=target_radius,
            normalize_colors=normalize_colors,
        )

        loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

        vae_model.eval()
        n_encoded = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc='    Encoding', leave=False)):
            feats = batch['features'].float().to(device)  # [B, 40000, 18 or 19]
            seg   = batch['segment_labels'].numpy()       # [B, 40000]

            with torch.no_grad():
                mu = extract_mu(feats)   # [B, 16384]

            mu_np = mu.cpu().float().numpy()
            del feats, mu
            torch.cuda.empty_cache()

            for i in range(mu_np.shape[0]):
                labels = seg[i]
                valid  = labels[labels >= 0]

                p_s = np.zeros(NUM_LABELS, dtype=np.float32)
                if len(valid) > 0:
                    for k in range(NUM_LABELS):
                        p_s[k] = float((valid == k).sum()) / len(valid)

                self.mu_list.append(mu_np[i])
                self.p_s_list.append(p_s)
                self.scene_ids.append(scene_dirs[n_encoded + i])

            n_encoded += mu_np.shape[0]

            if (batch_idx + 1) % 10 == 0:
                print(f'    Encoded {n_encoded}/{len(scene_dirs)} scenes')
                sys.stdout.flush()

        self.mu_list  = np.array(self.mu_list,  dtype=np.float32)
        self.p_s_list = np.array(self.p_s_list, dtype=np.float32)

        print(f'  Done. Encoded {len(self.mu_list)} scenes')
        print(f'  mu range:    [{self.mu_list.min():.3f}, {self.mu_list.max():.3f}]')
        active = ((self.p_s_list > 0).sum(axis=0) > 0).sum()
        print(f'  Active labels in split: {active}/72')
        sys.stdout.flush()

    def __len__(self):
        return len(self.mu_list)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.mu_list[idx]),
            torch.tensor(self.p_s_list[idx]),
        )

# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class LabelDistributionHead(nn.Module):
    def __init__(self, mu_dim=16384, proj_hidden=512, proj_out=256, n_labels=72):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(mu_dim, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_out),
            nn.LayerNorm(proj_out),
            nn.ReLU(),
        )
        self.dist_head = nn.Linear(proj_out, n_labels)
        n_params = sum(p.numel() for p in self.parameters())
        print(f'\n[LabelDistributionHead]')
        print(f'  mu [{mu_dim}] → [{proj_hidden}→{proj_out}] → dist [72]')
        print(f'  Parameters: {n_params/1e6:.3f}M  (head only, VAE frozen)')

    def forward(self, mu):
        h      = self.projection(mu)
        logits = self.dist_head(h)
        p_hat  = F.softmax(logits, dim=-1)
        return p_hat, h


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def kl_divergence_loss(p_s, p_hat, eps=1e-8):
    p_hat_c = torch.clamp(p_hat, min=eps)
    kl = (p_s * torch.log(p_s + eps) - p_s * torch.log(p_hat_c)).sum(dim=-1)
    return kl.mean()

def cosine_similarity_loss(p_s, p_hat):
    return F.cosine_similarity(p_s, p_hat, dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PER-LABEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_per_label(p_s_all, p_hat_all, presence_thresh=0.05, min_label_freq=3):
    N  = p_s_all.shape[0]
    y_true = (p_s_all  >= presence_thresh).astype(np.float32)
    y_pred = (p_hat_all >= presence_thresh).astype(np.float32)

    per_label = []
    for k in range(NUM_LABELS):
        freq = int(y_true[:, k].sum())
        if freq < min_label_freq:
            continue

        mae_k = float(np.abs(p_s_all[:, k] - p_hat_all[:, k]).mean())

        true_p = y_true[:, k] == 1
        pred_p = y_pred[:, k] == 1

        tp = float((y_true[:, k] * y_pred[:, k]).sum())
        recall_k    = tp / float(true_p.sum())  if true_p.sum() > 0 else float('nan')
        precision_k = tp / float(pred_p.sum())  if pred_p.sum() > 0 else float('nan')

        if np.isnan(recall_k) or np.isnan(precision_k) or (recall_k + precision_k) == 0:
            f1_k = float('nan')
        else:
            f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)

        mean_prop = float(p_s_all[true_p, k].mean()) if true_p.sum() > 0 else 0.0

        per_label.append({
            'label_idx': k, 'frequency': freq, 'mean_proportion': mean_prop,
            'mae': mae_k, 'recall': recall_k, 'precision': precision_k, 'f1': f1_k,
        })

    per_label.sort(key=lambda x: x['frequency'], reverse=True)

    valid_f1s  = [m['f1']        for m in per_label if not np.isnan(m['f1'])]
    valid_rec  = [m['recall']    for m in per_label if not np.isnan(m['recall'])]
    valid_prec = [m['precision'] for m in per_label if not np.isnan(m['precision'])]
    macro_f1   = float(np.mean(valid_f1s))  if valid_f1s  else float('nan')
    macro_rec  = float(np.mean(valid_rec))  if valid_rec  else float('nan')
    macro_prec = float(np.mean(valid_prec)) if valid_prec else float('nan')

    eps = 1e-8
    kl_vals = []; cos_vals = []; dom_correct = 0
    for i in range(N):
        ps, ph = p_s_all[i], p_hat_all[i]
        kl_vals.append(float(np.sum(ps * (np.log(ps+eps) - np.log(ph+eps)))))
        denom = np.linalg.norm(ps) * np.linalg.norm(ph) + eps
        cos_vals.append(float(np.dot(ps, ph) / denom))
        if ps.argmax() == ph.argmax():
            dom_correct += 1

    summary = {
        'macro_f1':               macro_f1,
        'macro_recall':           macro_rec,
        'macro_precision':        macro_prec,
        'mean_kl_divergence':     float(np.mean(kl_vals)),
        'mean_cosine_similarity': float(np.mean(cos_vals)),
        'dominant_label_accuracy': dom_correct / N,
        'n_qualifying_labels':    len(per_label),
        'n_val_chunks':           N,
        'presence_threshold':     presence_thresh,
    }
    return per_label, summary


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(train_kl, val_kl, train_cos, val_cos, output_dir, label_input):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    mode_str = 'Label Input (Option 1)' if label_input else 'Baseline'
    fig.suptitle(f'Projection Head Training Curves — {mode_str}', fontsize=13, fontweight='bold')

    epochs = list(range(1, len(train_kl)+1))
    ax1.plot(epochs, train_kl, label='Train KL', color='steelblue')
    ax1.plot(epochs, val_kl,   label='Val KL',   color='coral', linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('KL Divergence ↓')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_cos, label='Train Cosine', color='steelblue')
    ax2.plot(epochs, val_cos,   label='Val Cosine',   color='coral', linestyle='--')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Cosine Similarity ↑')
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: training_curves.png')


def plot_per_label_metrics(per_label_metrics, output_dir, presence_thresh, label_input):
    if not per_label_metrics:
        print('  No qualifying labels to plot.')
        return

    labels  = [f"L{m['label_idx']}" for m in per_label_metrics]
    freqs   = [m['frequency']     for m in per_label_metrics]
    maes    = [m['mae']           for m in per_label_metrics]
    recalls = [m['recall']    if not np.isnan(m['recall'])    else 0.0 for m in per_label_metrics]
    precs   = [m['precision'] if not np.isnan(m['precision']) else 0.0 for m in per_label_metrics]
    f1s     = [m['f1']        if not np.isnan(m['f1'])        else 0.0 for m in per_label_metrics]

    n = len(labels); x = np.arange(n)
    w = max(14, n * 0.5)
    fig = plt.figure(figsize=(w, 12))
    gs  = gridspec.GridSpec(3, 1, hspace=0.55)
    mode_str = 'Label Input (Option 1)' if label_input else 'Baseline'

    ax1 = fig.add_subplot(gs[0]); ax1b = ax1.twinx()
    ax1.bar(x, freqs, color='steelblue', alpha=0.6)
    ax1b.plot(x, maes, 'o-', color='coral', linewidth=1.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Frequency'); ax1b.set_ylabel('MAE ↓')
    ax1.set_title(f'[{mode_str}] Frequency and MAE  (θ={presence_thresh})')
    ax1.grid(True, alpha=0.2, axis='y')

    ax2 = fig.add_subplot(gs[1])
    colors2 = ['limegreen' if r>=0.7 else ('orange' if r>=0.4 else 'salmon') for r in recalls]
    ax2.bar(x, recalls, color=colors2, alpha=0.8)
    ax2.axhline(0.7, linestyle='--', color='green',  linewidth=1, label='Good (0.7)')
    ax2.axhline(0.4, linestyle='--', color='orange', linewidth=1, label='Fair (0.4)')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel('Recall ↑')
    ax2.set_title(f'[{mode_str}] Per-Label Recall')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2, axis='y')

    ax3 = fig.add_subplot(gs[2])
    ax3.bar(x-0.2, precs, width=0.4, label='Precision', color='mediumpurple', alpha=0.7)
    ax3.bar(x+0.2, f1s,   width=0.4, label='F1',        color='darkorange',   alpha=0.7)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylim(0, 1.05); ax3.set_ylabel('Score ↑')
    ax3.set_title(f'[{mode_str}] Per-Label Precision and F1')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2, axis='y')

    plt.savefig(output_dir / 'per_label_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: per_label_metrics.png')


def plot_distribution_scatter(p_s_all, p_hat_all, output_dir, label_input, n_examples=6):
    fig, axes = plt.subplots(2, n_examples//2, figsize=(16, 6))
    axes = axes.flatten()
    mode_str = 'Label Input (Option 1)' if label_input else 'Baseline'

    entropies = -np.sum(p_s_all * np.log(p_s_all + 1e-8), axis=1)
    top_mixed = np.argsort(entropies)[-n_examples//2:]
    dom_frac  = p_s_all.max(axis=1)
    top_dom   = np.argsort(dom_frac)[-n_examples//2:]
    examples  = np.concatenate([top_mixed, top_dom])

    for ai, ci in enumerate(examples):
        ax = axes[ai]
        ps, ph = p_s_all[ci], p_hat_all[ci]
        nz = np.where((ps > 0.01) | (ph > 0.01))[0]
        xp = np.arange(len(nz))
        ax.bar(xp-0.2, ps[nz], width=0.4, label='True', color='steelblue', alpha=0.8)
        ax.bar(xp+0.2, ph[nz], width=0.4, label='Pred', color='coral',     alpha=0.8)
        ax.set_xticks(xp)
        ax.set_xticklabels([f'L{k}' for k in nz], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        cos = float(np.dot(ps,ph) / (np.linalg.norm(ps)*np.linalg.norm(ph)+1e-8))
        t   = 'mixed' if ai < n_examples//2 else 'dominant'
        ax.set_title(f'Chunk {ci} ({t})  cos={cos:.2f}', fontsize=9)
        if ai == 0: ax.legend(fontsize=8)

    plt.suptitle(f'[{mode_str}] Predicted vs True Label Distributions', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: distribution_examples.png')


def plot_mu_statistics(mu_list, output_dir, label_input):
    """Extra plot: mu distribution statistics to compare baseline vs label_input."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    mode_str = 'Label Input (Option 1)' if label_input else 'Baseline'

    # 1. Histogram of all mu values
    axes[0].hist(mu_list.flatten(), bins=100, color='steelblue', alpha=0.7, density=True)
    axes[0].set_xlabel('mu value'); axes[0].set_ylabel('Density')
    axes[0].set_title(f'[{mode_str}] mu Distribution\n'
                      f'mean={mu_list.mean():.3f}  std={mu_list.std():.3f}')
    axes[0].grid(True, alpha=0.3)

    # 2. Per-dimension mean and std (first 512 dims)
    n_dims = min(512, mu_list.shape[1])
    dim_means = mu_list[:, :n_dims].mean(axis=0)
    dim_stds  = mu_list[:, :n_dims].std(axis=0)
    axes[1].plot(dim_means, color='steelblue', alpha=0.7, linewidth=0.5, label='mean')
    axes[1].fill_between(np.arange(n_dims),
                         dim_means - dim_stds, dim_means + dim_stds,
                         alpha=0.3, color='steelblue', label='±std')
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[1].set_xlabel(f'Latent dim (first {n_dims})'); axes[1].set_ylabel('Value')
    axes[1].set_title(f'[{mode_str}] Per-Dimension Mean ± Std')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # 3. mu norm per scene (how much information is encoded)
    norms = np.linalg.norm(mu_list, axis=1)
    axes[2].hist(norms, bins=30, color='coral', alpha=0.7, density=True)
    axes[2].set_xlabel('||mu||'); axes[2].set_ylabel('Density')
    axes[2].set_title(f'[{mode_str}] mu Norm per Scene\n'
                      f'mean={norms.mean():.1f}  std={norms.std():.1f}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'mu_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: mu_statistics.png')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SCENE DIRECTORIES AND SPLIT
# ─────────────────────────────────────────────────────────────────────────────

all_dirs = sorted([
    os.path.join(args.data_dir, d)
    for d in os.listdir(args.data_dir)
    if os.path.isdir(os.path.join(args.data_dir, d))
])[:args.n_total]

if len(all_dirs) < args.n_total:
    print(f'WARNING: only {len(all_dirs)} scenes found (requested {args.n_total})')

train_dirs = all_dirs[:n_train]
val_dirs   = all_dirs[n_train:]
print(f'Scene split:  head_train={len(train_dirs)}  head_val={len(val_dirs)}')

# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATASETS  (passes label_input from checkpoint metadata)
# ─────────────────────────────────────────────────────────────────────────────

print('\nBuilding training dataset (encoding with frozen VAE)...')
train_dataset = LabelDistributionDataset(
    train_dirs, vae, device, batch_size=16,
    label_input=ckpt_label_input,
    sampling_method=args.sampling_method,
    scale_norm_mode=ckpt_scale_norm,
    target_radius=args.target_radius,
    normalize_colors=args.normalize_colors,
)

print('\nBuilding validation dataset (encoding with frozen VAE)...')
val_dataset = LabelDistributionDataset(
    val_dirs, vae, device, batch_size=16,
    label_input=ckpt_label_input,
    sampling_method=args.sampling_method,
    scale_norm_mode=ckpt_scale_norm,
    target_radius=args.target_radius,
    normalize_colors=args.normalize_colors,
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2)

print(f'\nTrain: {len(train_dataset)} scenes  |  Val: {len(val_dataset)} scenes')

# Save mu statistics plot before head training
print('\nGenerating mu statistics plot...')
plot_mu_statistics(train_dataset.mu_list, output_dir, ckpt_label_input)

# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION HEAD + TRAINING
# ─────────────────────────────────────────────────────────────────────────────

head = LabelDistributionHead(
    mu_dim=16384, proj_hidden=args.proj_hidden,
    proj_out=args.proj_out, n_labels=NUM_LABELS,
).to(device)

optimizer = torch.optim.AdamW(
    head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

print(f'\n{"="*70}')
print(f'TRAINING PROJECTION HEAD  ({args.epochs} epochs)')
print(f'{"="*70}')

history = {
    'train_kl': [], 'val_kl': [],
    'train_cos': [], 'val_cos': [],
}
best_val_kl  = float('inf')
best_epoch   = 0
best_p_s     = None
best_p_hat   = None


def run_epoch(loader, head, optimizer=None, train=True):
    head.train() if train else head.eval()
    total_kl = total_cos = n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for mu_b, ps_b in loader:
            mu_b = mu_b.to(device); ps_b = ps_b.to(device)
            p_hat, _ = head(mu_b)
            kl  = kl_divergence_loss(ps_b, p_hat)
            cos = cosine_similarity_loss(ps_b, p_hat)
            if train:
                optimizer.zero_grad(); kl.backward(); optimizer.step()
            total_kl  += kl.item()
            total_cos += cos.item()
            n += 1

    return total_kl / n, total_cos / n


for epoch in tqdm(range(1, args.epochs+1), desc='Head training'):
    tr_kl, tr_cos = run_epoch(train_loader, head, optimizer, train=True)
    vl_kl, vl_cos = run_epoch(val_loader,   head, optimizer=None, train=False)
    scheduler.step()

    history['train_kl'].append(tr_kl)
    history['val_kl'].append(vl_kl)
    history['train_cos'].append(tr_cos)
    history['val_cos'].append(vl_cos)

    if epoch % 10 == 0 or epoch == 1:
        print(f'  Epoch {epoch:3d}/{args.epochs} | '
              f'Train KL={tr_kl:.4f} cos={tr_cos:.3f} | '
              f'Val KL={vl_kl:.4f} cos={vl_cos:.3f}')

    if vl_kl < best_val_kl:
        best_val_kl = vl_kl
        best_epoch  = epoch
        head.eval()
        ps_list = []; ph_list = []
        with torch.no_grad():
            for mu_b, ps_b in val_loader:
                ph, _ = head(mu_b.to(device))
                ps_list.append(ps_b.numpy())
                ph_list.append(ph.cpu().numpy())
        best_p_s   = np.concatenate(ps_list,  axis=0)
        best_p_hat = np.concatenate(ph_list, axis=0)
        torch.save({
            'epoch': epoch, 'model_state': head.state_dict(),
            'val_kl': vl_kl, 'label_input': ckpt_label_input,
        }, output_dir / 'best_head.pth')

print(f'\n  Best val KL: {best_val_kl:.4f}  (epoch {best_epoch})')

# ─────────────────────────────────────────────────────────────────────────────
# FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print(f'\n{"="*70}')
print(f'FINAL PER-LABEL EVALUATION')
print(f'{"="*70}')

per_label_metrics, summary = evaluate_per_label(
    best_p_s, best_p_hat,
    presence_thresh=args.presence_thresh,
    min_label_freq=args.min_label_freq,
)

mode_str = 'LABEL INPUT (Option 1)' if ckpt_label_input else 'BASELINE'
print(f'\n  MODE: {mode_str}')
print(f'\n  SUMMARY METRICS:')
print(f'  {"─"*50}')
print(f'  Dominant label accuracy:  {summary["dominant_label_accuracy"]:.3f}')
print(f'  Mean cosine similarity:   {summary["mean_cosine_similarity"]:.3f}')
print(f'  Mean KL divergence:       {summary["mean_kl_divergence"]:.4f}')
print(f'  Macro-averaged F1:        {summary["macro_f1"]:.3f}')
print(f'  Macro-averaged Recall:    {summary["macro_recall"]:.3f}')
print(f'  Macro-averaged Precision: {summary["macro_precision"]:.3f}')
print(f'  Qualifying labels:        {summary["n_qualifying_labels"]}')
print(f'  {"─"*50}')

print(f'\n  PER-LABEL TABLE:')
print(f'  {"Label":>6}  {"Freq":>6}  {"Mean%":>6}  {"MAE":>6}  {"Recall":>7}  {"Precis":>7}  {"F1":>6}')
print(f'  {"─"*60}')
for m in per_label_metrics:
    r = f'{m["recall"]:7.3f}'    if not np.isnan(m['recall'])    else '    N/A'
    p = f'{m["precision"]:7.3f}' if not np.isnan(m['precision']) else '    N/A'
    f = f'{m["f1"]:6.3f}'        if not np.isnan(m['f1'])        else '   N/A'
    print(f'  L{m["label_idx"]:>4d}    {m["frequency"]:>5d}  '
          f'{m["mean_proportion"]*100:>5.1f}%  '
          f'{m["mae"]:>6.4f}  {r}  {p}  {f}')

# Dominant vs mixed breakdown
dom_mask = best_p_s.max(axis=1) >= 0.50
n_dom    = dom_mask.sum()
n_mix    = (~dom_mask).sum()
eps      = 1e-8

dom_acc = dom_kl = mix_kl = float('nan')
if n_dom > 0:
    dom_acc = float(np.mean(
        best_p_s[dom_mask].argmax(axis=1) == best_p_hat[dom_mask].argmax(axis=1)))
    dom_kl = float(np.mean([
        np.sum(best_p_s[dom_mask][i] *
               (np.log(best_p_s[dom_mask][i]+eps) - np.log(best_p_hat[dom_mask][i]+eps)))
        for i in range(n_dom)]))
if n_mix > 0:
    mix_kl = float(np.mean([
        np.sum(best_p_s[~dom_mask][i] *
               (np.log(best_p_s[~dom_mask][i]+eps) - np.log(best_p_hat[~dom_mask][i]+eps)))
        for i in range(n_mix)]))

print(f'\n  DOMINANT vs MIXED CHUNKS (dominant = top label ≥ 50%):')
print(f'  Dominant: n={n_dom}  top-1 acc={dom_acc:.3f}  KL={dom_kl:.4f}')
print(f'  Mixed:    n={n_mix}  KL={mix_kl:.4f}')

summary.update({
    'dominant_chunk_accuracy': dom_acc,
    'dominant_chunk_kl': dom_kl,
    'mixed_chunk_kl': mix_kl,
    'n_dominant_chunks': int(n_dom),
    'n_mixed_chunks': int(n_mix),
    'label_input': ckpt_label_input,
    'ckpt_semantic_mode': ckpt_semantic_mode,
    'ckpt_epoch': str(ckpt_epoch),
    'feature_width': feature_width,
    'point_feats': point_feats,
})

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print(f'\nSaving results to {output_dir}...')
results_json = {
    'run_name': run_name, 'checkpoint': args.checkpoint,
    'mode': mode_str, 'label_input': ckpt_label_input,
    'feature_width': feature_width, 'point_feats': point_feats,
    'n_train_chunks': len(train_dataset), 'n_val_chunks': len(val_dataset),
    'best_val_kl_epoch': best_epoch, 'best_val_kl': best_val_kl,
    'summary': summary, 'per_label_metrics': per_label_metrics,
    'history': history,
    'args': {
        'presence_thresh': args.presence_thresh,
        'min_label_freq': args.min_label_freq,
        'proj_hidden': args.proj_hidden, 'proj_out': args.proj_out,
        'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
    }
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print('  Saved: results.json')

np.save(output_dir / 'p_s_val.npy',   best_p_s)
np.save(output_dir / 'p_hat_val.npy', best_p_hat)
np.save(output_dir / 'mu_train.npy',  train_dataset.mu_list)
np.save(output_dir / 'mu_val.npy',    val_dataset.mu_list)
print('  Saved: p_s_val.npy, p_hat_val.npy, mu_train.npy, mu_val.npy')

print('\nGenerating plots...')
plot_training_curves(
    history['train_kl'], history['val_kl'],
    history['train_cos'], history['val_cos'],
    output_dir, ckpt_label_input,
)
plot_per_label_metrics(per_label_metrics, output_dir, args.presence_thresh, ckpt_label_input)
plot_distribution_scatter(best_p_s, best_p_hat, output_dir, ckpt_label_input)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f'\n{"="*70}')
print(f'PROBING COMPLETE — {mode_str}')
print(f'{"="*70}')
print(f'  Checkpoint:             {Path(args.checkpoint).name}')
print(f'  label_input (auto):     {ckpt_label_input}')
print(f'  feature_width:          {feature_width}  (encoder input_proj: Linear({point_feats}, d_model))')
print(f'  Best val KL:            {best_val_kl:.4f}  (epoch {best_epoch})')
print(f'  Dominant label acc:     {summary["dominant_label_accuracy"]:.1%}')
print(f'  Mean cosine similarity: {summary["mean_cosine_similarity"]:.3f}')
print(f'  Macro F1:               {summary["macro_f1"]:.3f}')
print(f'  Output dir:             {output_dir}')
print(f'{"="*70}')
print()
print('INTERPRETATION GUIDE:')
print('  Dominant label acc ≥ 0.60 → latent encodes primary surface type  ✓')
print('  Mean cosine sim   ≥ 0.70 → distribution shape well captured      ✓')
print('  Macro F1          ≥ 0.50 → semantic structure present in latent  ✓')
print()
print('  Compare these numbers against the baseline probe run to see')
print('  whether adding labels as encoder input improves semantic encoding.')