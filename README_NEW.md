# Can3Tok VAE — Semantic-Aware 3D Scene Tokenizer

A Perceiver-based Variational Autoencoder that encodes full indoor 3DGS scenes into a structured latent space, with DC/AC color decomposition, scene-level semantic supervision, and per-Gaussian InfoNCE contrastive learning. Built on SceneSplat-7K as the foundation for generative scene completion.

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

**`sal_perceiver_dist_changes.py`** — `MeanColorHead` is a small MLP attached to `shape_embed`:

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

## Concept 2 — Scene-Level Label Distribution Prediction (Move 1)

While Concept 1 gives `shape_embed` a color prediction task (3 numbers), Move 1 gives it a second, richer task: predict the **semantic composition** of the entire scene as a probability distribution over ScanNet72 categories.

### Motivation

After Concept 1, `shape_embed` is forced to encode 3 numbers (mean RGB). But a bedroom and a kitchen may have similar mean colors yet completely different semantic content — one is dominated by bed, pillow, and nightstand; the other by countertop, cabinet, and appliance. This scene-level semantic character is a global property that belongs in `shape_embed`, not in the per-Gaussian latents.

Move 1 gives `shape_embed` a second explicit task: predict **what fraction of the scene belongs to each of the 72 ScanNet categories**. This is the semantic fingerprint of the room.

---

### The Ground Truth: Scene Label Distribution

For each scene $b$, every Gaussian carries a ScanNet72 category label $y_i \in \{0, \ldots, 71\}$. The ground truth is the **empirical label distribution** — what fraction of Gaussians belong to each category:

$$p_s^{(b)}[k] = \frac{\sum_{i=1}^{N} \mathbf{1}[y_i^{(b)} = k]}{N_{\text{valid}}^{(b)}}$$

This is a proper probability distribution summing to 1. For a typical bedroom scene this might be:

| Label | Category | $p_s[k]$ |
|-------|----------|----------|
| 12    | floor    | 0.28     |
| 16    | wall     | 0.24     |
| 21    | ceiling  | 0.18     |
| 35    | bed      | 0.09     |
| ...   | ...      | ...      |

Two bedrooms will have similar $p_s$. A bedroom and a kitchen will have very different $p_s$. This distribution is the **semantic fingerprint** of the scene.

---

### Implementation: Dataset Side

**`gs_dataset_scenesplat.py`** — precomputed as a simple histogram in `__getitem__`, on CPU inside DataLoader workers:

```python
# Precompute ScanNet72 label distribution — zero GPU cost at training time
label_dist = np.zeros(72, dtype=np.float32)
valid_seg = segment[segment >= 0]
if len(valid_seg) > 0:
    for k in range(72):
        label_dist[k] = (valid_seg == k).sum()
    label_dist /= label_dist.sum()   # normalize to a probability distribution

return {
    ...
    'label_dist': label_dist,   # [72] float32, sums to 1
}
```


The label distribution is a fixed property of each scene that never changes between epochs. Computing it as a simple CPU histogram in a DataLoader worker and passing `[B, 72]` to the GPU (a trivial host-to-device copy) reduces the per-batch cost to essentially zero. The DataLoader workers run concurrently with GPU compute, so the cost is fully hidden.

| Implementation | Tensor allocated | Memory/batch | Epoch time |
|----------------|-----------------|-------------|------------|
| one_hot in training loop | `[64, 40000, 72]` float32 | 737 MB | ~400s |
| Histogram in dataset | `[64, 72]` float32 | 18 KB | ~65s |

---

### Implementation: Model Side

**`sal_perceiver_II_initialization.py`** — `SceneSemanticHead` is a 3-layer MLP with LayerNorm, attached to `shape_embed` independently of `MeanColorHead`:

```python
class SceneSemanticHead(nn.Module):
    def __init__(self, embed_dim=32, num_classes=72):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, shape_embed_z):
        # shape_embed_z: [B, 32]
        logits = self.net(shape_embed_z)          # [B, 72]
        return F.softmax(logits, dim=-1)           # [B, 72]  — valid probability distribution
```

The softmax ensures $\hat{p}_s$ sums to 1 and all values are positive, matching the structure of the ground truth $p_s$. Parameters: ~108K (vs MeanColorHead's 8K).

---

### Implementation: Loss Side

**`gs_can3tok_2.py`** — KL divergence between predicted and ground truth distributions:

```python
def scene_semantic_kl_loss(p_hat, p_s, eps=1e-8):
    """D_KL(p_s || p_hat) = Σ p_s * log(p_s / p_hat)"""
    p_hat_clamped = torch.clamp(p_hat, min=eps)
    kl_per_scene  = (p_s * (torch.log(p_s + eps) - torch.log(p_hat_clamped))).sum(dim=-1)
    return kl_per_scene.mean()

# In training loop — p_s already precomputed in dataset:
p_s   = batch_data['label_dist'].float().to(device)               # [B, 72]
p_hat = gs_autoencoder.shape_model.last_scene_semantic_pred        # [B, 72]
scene_semantic_loss = scene_semantic_kl_loss(p_hat, p_s)

# Full loss:
L_total = L_recon
        + kl_weight          * L_KL
        + lambda_color        * L_color_pred         # Concept 1: MeanColorHead
        + lambda_scene_sem    * L_scene_semantic      # Concept 2: SceneSemanticHead
        + beta                * L_infonce             # Concept 3: InfoNCE (optional)
```


**Properties:**
- $D_{\text{KL}} \geq 0$ always; $D_{\text{KL}} = 0$ iff $\hat{p}_s = p_s$ exactly
- Predicting near-zero for a dominant category causes a very large penalty (log term blows up)
- Predicting anything for a truly absent category has zero penalty ($p_s[k] \cdot \log(\cdot) = 0$)

---

### All Four Gradient Paths: Full Picture

After Concepts 1 and 2, the full model has **four independent gradient paths**:

```
PATH 1 — Reconstruction
  L_recon  ->  GS_decoder  ->  post_kl  ->  transformer  ->  mu / latent_tokens

PATH 2 — KL Regularisation
  L_KL  ->  mu, log_var  ->  encoder

PATH 3 — Mean Color (Concept 1)
  L_color_mse  ->  MeanColorHead  ->  shape_embed  ->  encoder token 0

PATH 4 — Scene Semantic Distribution (Concept 2)
  L_scene_kl  ->  SceneSemanticHead  ->  shape_embed  ->  encoder token 0
```

Paths 3 and 4 both target `shape_embed` but through separate heads, separate losses, and separate output spaces. They are fully independent — the network learns to use different dimensions of `shape_embed` for colour information and semantic information.

---

### What `shape_embed` Is Encoding

| Supervision source | Signal dimensionality | What shape_embed learns |
|-------------------|-----------------------|------------------------|
| Concept 1: MeanColorHead | 3 numbers (mean RGB) | Scene colour palette |
| Concept 2: SceneSemanticHead | 72 numbers (label distribution) | Dominant object categories, room type |
| Paths 1+2 (indirect) | Reconstruction quality | Residual global geometry |

After training with both heads, `shape_embed [B, 32]` encodes a structured global scene descriptor — colour palette, dominant category mix, and room type identity — all in 32 dimensions. This rich conditioning signal is what the downstream latent diffusion completion model will use to answer: *"What kind of scene am I completing?"*

---

### Information Flow: Concepts 1 + 2 Combined

```
ENCODER
  40k Gaussians -> CrossAttentionEncoder
    -> shape_embed [B, 384] -> compressed -> z_shape [B, 32]
                                                  |
                               ┌──────────────────┴──────────────────┐
                               ↓                                     ↓
                         MeanColorHead                    SceneSemanticHead
                               ↓                                     ↓
                    mean_color_pred [B, 3]          scene_dist_pred [B, 72]
                               ↓                                     ↓
                     MSE vs mean_color_gt              KL vs label_dist [B, 72]
                     (from dataset batch)              (precomputed histogram)
```

---

## Concept 3 — Per-Gaussian Semantic InfoNCE Supervision

Per-Gaussian contrastive learning using ScanNet72 category labels as supervision signal. Operates on the **decoder hidden state** — a completely separate gradient path from Concepts 1 and 2 which both target `shape_embed`.

**`semantic_losses.py`** — `ScanNet72SemanticLoss`:

```
For each scene in batch:
  1. Extract per-Gaussian features from SemanticProjectionHead [40k, 32]
  2. Subsample 10k Gaussians (balanced across categories — prevents wall/floor dominance)
  3. Compute category prototypes: mean feature per ScanNet72 class
  4. InfoNCE loss (temperature=0.1):
       L = -log[ exp(sim(f_i, proto_c) / τ) / Σ_j exp(sim(f_i, proto_j) / τ) ]
```

Effect: Gaussians from the same semantic category are pulled together in the 32-dim feature space; Gaussians from different categories are pushed apart. PCA visualization at epoch 1200 confirms this — ceiling Gaussians cluster in blue/cyan, furniture in red/pink, with clean category separation emerging without instance-level supervision.

**Key distinction from Concept 2:** InfoNCE operates on per-Gaussian features `[B, 40k, 32]` from the decoder hidden state. Concept 2 operates on the global scene token `shape_embed [B, 32]`. They supervise completely different parts of the network and are fully complementary.

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

# Concept 1: Color residual
color_residual          = True
lambda_color            = 1.0     # MSE weight on mean color prediction

# Concept 2: Scene semantic head (Move 1)
scene_semantic_head     = True
scene_semantic_weight   = 0.3     # KL weight on label distribution prediction

# Concept 3: Per-Gaussian InfoNCE (optional, adds ~50s/epoch)
semantic_mode           = 'hidden'
segment_loss_weight     = 0.3
semantic_temperature    = 0.1
semantic_subsample      = 10000
```

---


## Dataset: SceneSplat-7K with Labels

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
├── gs_can3tok_2.py                                      # Training loop, all losses
├── gs_dataset_scenesplat.py                             # Dataset, DC/AC preprocessing,
│                                                        #   label_dist precomputation
├── semantic_losses.py                                   # ScanNet72SemanticLoss (InfoNCE)
├── gs_ply_reconstructor.py                              # Output to .ply (SuperSplat/CloudCompare)
├── pca_feature_visualization.py                         # PCA coloring of semantic features
├── visualize_input_pca.py                               # Input scene PCA (no forward pass)
├── run_can3tok_move1.sh                                 # SLURM job script (Run C / Run D)
├── upload_logs_to_wandb.py                              # W&B log uploader
└── model/
    ├── configs/aligned_shape_latents/shapevae-256.yaml  # Architecture config
    └── michelangelo/models/tsal/
        ├── sal_perceiver_II_initialization.py           # Full model: Encoder, Decoder,
        │                                                #   MeanColorHead, SceneSemanticHead,
        │                                                #   SemanticProjectionHead
        └── asl_pl_module.py                             # PyTorch Lightning wrapper
```

---

## Quick Start

```bash
# Run C: colour residual + SceneSemanticHead only (isolates Move 1)
sbatch run_can3tok_move1.sh        # default — SEMANTIC_MODE=none, SCENE_SEMANTIC_HEAD=True

# Run D: colour residual + SceneSemanticHead + InfoNCE (full model)
# Edit run_can3tok_move1.sh: SEMANTIC_MODE=hidden, SEGMENT_WEIGHT=0.3
sbatch run_can3tok_move1.sh

# Run A: colour residual only (baseline for ablation)
python gs_can3tok_2.py \
    --color_residual --semantic_mode none \
    --batch_size 64 --num_epochs 1500 --lr 1e-4 --kl_weight 1e-6 \
    --train_scenes 2000 --eval_every 50

