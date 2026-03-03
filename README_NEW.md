# Can3Tok VAE — Semantic-Aware 3D Scene Tokenizer

A Perceiver-based Variational Autoencoder that encodes full indoor 3DGS scenes into a structured latent space, with DC/AC color decomposition and semantic InfoNCE supervision. Built on SceneSplat-7K as the foundation for generative scene completion.

---

## Architecture Overview

### Input Representation
Each scene is represented as **40,000 Gaussians** (sampled by opacity), each with 14 parameters:

```
[x, y, z,  r, g, b,  opacity,  sx, sy, sz,  qw, qx, qy, qz]
  (3)       (3)       (1)       (3)           (4)
```

The dataset assembles an **18-channel feature vector** per Gaussian for the encoder:

```
[voxel_center(3), voxel_id(1), xyz(3), color(3), opacity(1), scale(3), quat(4)]
```

Input tensor shape: **`[B, 40000, 18]`**

---

### Encoder — `CrossAttentionEncoder`

A Perceiver-style encoder with **dual Fourier positional embeddings**:

```
Input [B, 40000, 18]
  -> Fourier(voxel_centers)   # coarse spatial — assigns each Gaussian to a 40³ voxel grid
  -> Fourier(xyz_positions)   # fine spatial   — exact Gaussian location within voxel
  -> Concatenate [fourier_voxel, fourier_xyz, gaussian_params]
  -> Input projection -> width=384
  -> Cross-attention: 512 learned queries attend to all 40k Gaussians
  -> 6x Self-attention transformer layers
Output: [B, 512, 384]
```

Why dual Fourier? The voxel embedding gives structural context (which part of the room) while the exact xyz gives fine-grained position. Together they prevent aliasing of nearby Gaussians that happen to share a voxel.

---

### VAE Bottleneck

The 512 latent tokens are split into two parts:

```
[B, 512, 384]
  -> split: shape_embed [B, 1, 384]  +  latent_tokens [B, 511, 384]
  -> project to mu, log_var
  -> z = mu + sigma * eps  (reparameterization)
Output: [B, 512, 32]
```

**`shape_embed`** is the global scene token — a single vector meant to carry the overall scene identity. It is the most important token in the representation, and also the hardest to train (see Color Residual section below).

---

### Decoder — `GS_decoder`

A flat MLP decoder that reconstructs all 40,000 Gaussians from the latent:

```
[B, 512, 32]
  -> post-KL projection -> [B, 512, 384]
  -> 12x Transformer self-attention layers
  -> Flatten -> [B, 512*384 = 196608]
  -> 8-layer MLP
  -> Raw output [B, 40000, 14]
  -> Activations:
       colors    -> sigmoid  -> [0, 1]
       opacity   -> sigmoid  -> [0, 1]
       scales    -> softplus -> (0, +∞)
       quats     -> L2 norm  -> ||q|| = 1
Output: [B, 40000, 14]
```

---

### Semantic Head — `SemanticProjectionHead`

Extracts per-Gaussian semantic features for contrastive supervision. Three modes:

| Mode | Input | Path | Output |
|------|-------|------|--------|
| `geometric` | GS params `[B, 40k, 14]` | 3-layer MLP | `[B, 40k, 32]` |
| `hidden` | Decoder hidden state `[B, 1024]` | MLP → reshape | `[B, 40k, 32]` |
| `attention` | Positions + transformer tokens | Cross-attention | `[B, 40k, 32]` |

All outputs are L2-normalized to a unit hypersphere. `hidden` mode gave the best L2 reconstruction. `geometric` mode gave the cleanest scale distribution.

---

## Concept 1 — DC/AC Color Residual Decomposition

This is the primary architectural contribution. It solves a fundamental gradient starvation problem in the original design.

### The Problem: `shape_embed` Gradient Starvation

In the original Can3Tok encoder, the 512 latent tokens are split into:
- `shape_embed`: 1 token → `[B, 1, 384]` → compressed to `[B, 1, 32]`
- `latent_tokens`: 511 tokens → `[B, 511, 384]` → compressed to `[B, 511, 32]`

`shape_embed` is intended to be a **global scene descriptor** — it should encode the overall room identity. However, during training it receives almost no gradient signal because:

1. The decoder reconstructs 40,000 Gaussians from 512 tokens combined
2. The 511 `latent_tokens` have sufficient capacity to explain the per-Gaussian variation
3. `shape_embed` has no dedicated supervision path — it can be nearly zero with no penalty
4. Without a gradient, `shape_embed` collapses to carrying no meaningful information

The result: the encoder bottleneck is broken. `mu` tries to encode both local geometry *and* global scene color, causing reconstruction collapse.

---

### The Solution: Two-Level Color Decomposition

Inspired by VQ-VAE-2's hierarchical latent structure, we split color encoding into two levels:

**Level 1 (DC — Global Color):** `shape_embed` → `MeanColorHead` → predicts the scene mean color

**Level 2 (AC — Local Residuals):** `latent_tokens` / `mu` → encode per-Gaussian color *deviations* from the mean

This gives `shape_embed` a concrete, measurable task with direct gradient flow.

---

### Implementation: Dataset Side

**`gs_dataset_scenesplat.py`** — preprocessing step during data loading:

```python
# Per-scene, applied to color.npy [N, 3] where N ~ millions of Gaussians
scene_mean_color = color.mean(axis=0)          # [3]  — the DC component
color_residuals  = color - scene_mean_color    # [N, 3]  range ~ [-0.3, +0.3]

# Store both
scene['color']            = color_residuals    # replaces absolute RGB [0,1]
scene['mean_color_label'] = scene_mean_color   # supervision target for shape_embed
```

The assembled feature tensor now contains residuals in the color channels:

```
Before: [voxel_center(3), voxel_id(1), xyz(3),  color_absolute(3),  opacity(1), scale(3), quat(4)]
After:  [voxel_center(3), voxel_id(1), xyz(3),  color_residual(3),  opacity(1), scale(3), quat(4)]
```

Shape is unchanged: **`[B, 40000, 18]`**. The difference is that the 3 color channels now range over `[-0.3, +0.3]` (zero-centered residuals) instead of `[0, 1]` (absolute RGB).

---

### Implementation: Model Side

**`sal_perceiver_II_initialization.py`** — `MeanColorHead` is a small MLP attached to `shape_embed`:

```python
class MeanColorHead(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()          # output in [0, 1] — matches original RGB range
        )

    def forward(self, shape_embed_z):
        # shape_embed_z: [B, 32]  (the post-KL compressed shape token)
        return self.net(shape_embed_z)   # -> [B, 3]
```

During forward pass:

```
shape_embed_z [B, 32]  ->  MeanColorHead  ->  mean_color_pred [B, 3]
```

The prediction is stored as `self.last_mean_color_pred` on the model instance — this avoids changing the return signature of the encoder and keeps `asl_pl_module.py` untouched.

---

### Implementation: Loss Side

**`gs_can3tok_2.py`** — additional MSE loss on mean color prediction:

```python
# mean_color_pred: [B, 3]  — from MeanColorHead(shape_embed_z)
# mean_color_label: [B, 3] — from dataset preprocessing
color_pred_loss = F.mse_loss(mean_color_pred, mean_color_label)

# Total loss
L_total = L_recon + kl_weight * L_KL + beta * L_semantic + lambda_color * color_pred_loss
```

This loss creates a **direct gradient path** from the color prediction error back through `MeanColorHead` into `shape_embed_z`, and from there back through the KL bottleneck into the encoder's `shape_embed` token.

---

### Information Flow Summary

```
ENCODER
  40k Gaussians (residual colors) -> CrossAttentionEncoder
    -> shape_embed [B, 1, 384]   <- now has a real job
    -> latent_tokens [B, 511, 384]

VAE BOTTLENECK
  shape_embed -> mu_shape [B, 1, 32] -> z_shape [B, 32]
                                             |
                                     MeanColorHead
                                             |
                                   mean_color_pred [B, 3]
                                             |
                              MSE loss vs mean_color_label [B, 3]  ← gradient flows back here

  latent_tokens -> mu [B, 511, 32] -> z [B, 511*32]
                                             |
                              Decoder reconstructs color_residuals [B, 40k, 3]
                              (fine AC detail — not the mean)

RECONSTRUCTION
  final_color [B, 40k, 3] = mean_color_pred [B, 1, 3] + color_residual_pred [B, 40k, 3]
```

---

### Why It Works: Dimensional Analysis

| Component | Tensor | What it encodes |
|-----------|--------|-----------------|
| `shape_embed_z` | `[B, 32]` | Global scene identity: overall room tone, lighting character |
| `mean_color_pred` | `[B, 3]` | Scene-level DC color (e.g. warm yellow room vs cool grey room) |
| `mu` / `z` (latent) | `[B, 511, 32]` | Per-region AC color residuals + all geometry |
| `color_residuals` in features | `[B, 40000, 3]` | Per-Gaussian deviation from scene mean |

Before this change, `shape_embed_z` had to somehow encode global color via the reconstruction loss through 40,000 Gaussians — a vanishingly small gradient. Now it has a direct 3-dimensional prediction task that provides strong, clean gradients every step.

---

### Ablation Results

| Configuration | Validation L2 | ΔL2 |
|--------------|--------------|-----|
| Baseline (no color residual, no semantic) | 3.23 | — |
| Color residual only | 2.49 | **−23%** |
| Semantic InfoNCE only (β=0.3) | 3.30 | marginal |
| Color residual + Semantic (full model) | 2.05 | **−37%** |
| Extended training (15K epochs, lr=1e-5) | 1.58 | **−51%** |

The 23% jump from color residual alone confirms the gradient starvation diagnosis. Semantic InfoNCE adds an independent 18% improvement (2.49 → 2.05) because the two gradient paths target different parts of the network — `shape_embed` vs decoder hidden state.

---

##  Concept 2 — Semantic InfoNCE Supervision

Per-Gaussian contrastive learning using ScanNet72 category labels as supervision signal.

**`semantic_losses.py`** — `ScanNet72SemanticLoss`:

```
For each scene in batch:
  1. Extract semantic features from SemanticProjectionHead [40k, 32]
  2. Subsample 10k Gaussians (balanced across categories to prevent wall/floor dominance)
  3. Compute category prototypes: mean feature per ScanNet72 class
  4. InfoNCE loss with temperature=0.1:
       L = -log[ exp(sim(f_i, proto_c) / τ) / Σ_j exp(sim(f_i, proto_j) / τ) ]
```

Effect: Gaussians from the same semantic category are pulled together in the 32-dim feature space; Gaussians from different categories are pushed apart. The PCA visualization at epoch 1200 confirms this — ceiling Gaussians cluster in blue/cyan, furniture in red/pink, with clean category separation emerging without instance-level supervision.

---

## Training Configuration

```python
# Architecture
num_latents       = 256
embed_dim         = 32
width             = 384
encoder_layers    = 6
decoder_layers    = 12

# Optimization
learning_rate     = 1e-4
batch_size        = 64
kl_weight         = 1e-6    # 1e-5 causes KL to dominate at late epochs

# Color residual
color_residual    = True
lambda_color      = 1.0     # MSE weight on mean color prediction

# Semantic
semantic_mode     = 'hidden'
segment_loss_weight = 0.3
semantic_temperature = 0.1
semantic_subsample = 10000
```

---

## Dataset: SceneSplat-7K with ScanNet72 Labels

**Source:** SceneSplat (ICCV 2025 Oral, Li et al. 2025) — 7,916 indoor scenes from ScanNet, ScanNet++, Replica, Hypersim, 3RScan, ARKitScenes, Matterport3D.

Each scene directory:
```
scene_dir/
├── coord.npy      [N, 3]   Gaussian positions
├── color.npy      [N, 3]   RGB values (stored as residuals after preprocessing)
├── scale.npy      [N, 3]   Anisotropic scales, linear space (metres)
├── quat.npy       [N, 4]   Rotation quaternions
├── opacity.npy    [N]      Opacity values [0, 1]
├── segment.npy    [N]      ScanNet72 category labels (0–71)
└── instance.npy   [N]      Instance IDs (−1 = background)
```

**Normalization:** positions and scales normalized to fit a 10m radius sphere (linear scale multiplication, not log-space). Log-space scale loss was tested and abandoned — it causes scale collapse to <1cm due to domain mismatch.

**Sampling:** top 40k Gaussians by opacity, deterministic argsort (same selection every epoch).

---

## Code Structure

```
.
├── gs_can3tok_2.py                                      # Training loop, losses
├── gs_dataset_scenesplat.py                             # Dataset, DC/AC preprocessing
├── semantic_losses.py                                   # ScanNet72SemanticLoss (InfoNCE)
├── gs_ply_reconstructor.py                              # Output to .ply (SuperSplat/CloudCompare)
├── pca_feature_visualization.py                         # PCA coloring of semantic features
├── visualize_input_pca.py                               # Input scene PCA (no forward pass)
├── job_can3tok.sh                                       # SLURM job script
└── model/
    ├── configs/aligned_shape_latents/shapevae-256.yaml  # Architecture config
    └── michelangelo/models/tsal/
        ├── sal_perceiver_II_initialization.py           # Full model: Encoder, Decoder, Heads
        └── asl_pl_module.py                             # PyTorch Lightning wrapper
```

---

## Quick Start

```bash
# Full model: color residual + semantic InfoNCE
python gs_can3tok_2.py \
    --batch_size 64 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --kl_weight 1e-6 \
    --color_residual \
    --semantic_mode hidden \
    --segment_loss_weight 0.3 \
    --train_scenes 2000 \
    --eval_every 50

# Baseline: no color residual, no semantics
python gs_can3tok_2.py \
    --batch_size 64 \
    --num_epochs 700 \
    --lr 1e-4 \
    --kl_weight 1e-6 \
    --semantic_mode none \
    --train_scenes 300

# Visualize input scene PCA
python visualize_input_pca.py \
    --num_scenes 5 \
    --output_dir ./input_pca_vis
```