# Can3Tok VAE — Semantic-Aware 3D Gaussian Scene Tokenizer

A Perceiver-based Variational Autoencoder that encodes full indoor 3DGS scenes into a structured, disentangled latent space. The model supports semantic supervision, spatial inductive bias, and a second-stage diffusion pipeline for scene generation.

Built on [SceneSplat-7K](https://arxiv.org/abs/2501.01895) (ICCV 2025 Oral) as the dataset foundation.

---

## Visualisation

The images below show the best reconstruction results achieved so far. Input and reconstruction are visually nearly identical at epoch 1950, confirming the VAE has learned a high-quality latent representation of indoor 3DGS scenes.

| Input Scene | Reconstructed (Epoch 1950) |
|---|---|
| `input_scene_000_0201_840151_0.ply` | `scene_000_epoch_1950_786.ply` |

---

## Architecture Overview

### 1. Input Representation

Each scene is sampled to **40,000 Gaussians** (top-k by opacity). Each Gaussian carries 14 parameters:

```
[x, y, z,  r, g, b,  opacity,  sx, sy, sz,  qw, qx, qy, qz]
   pos(3)   color(3)  opac(1)   scale(3)      quat(4)
```

The dataset assembles an **18-channel feature tensor** per Gaussian for the encoder:

```
cols  0:3   voxel_center   — coarse 40³ grid position
col   3     voxel_id       — encoder voxel index
cols  4:7   xyz            — absolute Gaussian position
cols  7:10  rgb            — color (or color residual if color_residual=True)
col   10    opacity
cols  11:14 scale
cols  14:18 quaternion
```

Input tensor: **`[B, 40000, 18]`**

---

### 2. Encoder — `CrossAttentionEncoder`

A Perceiver-style encoder with dual Fourier positional embeddings that separately encode coarse and fine spatial structure:

```
Input [B, 40000, 18]
  xyz_actual    → FourierEmbedder (num_freqs=8, input_dim=3)  → fine PE
  voxel_centers → FourierEmbedder (num_freqs=8, input_dim=3)  → coarse PE
  gaussian_params (opacity, scale, quat)  →  raw 8-dim features

  Concatenate [fine_PE | coarse_PE | gaussian_params]  → Input projection → width=384

  512 learned queries  →  Cross-attention over all 40k Gaussians
                       →  6× Self-attention transformer layers

Output: [B, 512, 384]   (512 latent tokens, each 384-dim)
  split: shape_embed [B, 384]   ← token 0, global scene descriptor
         geom_tokens [B, 511, 384] ← tokens 1-511, per-region geometry
```

The dual Fourier embedder gives the encoder a spectral multi-scale view of each Gaussian's position: coarse grid coordinates encode which room region it belongs to, fine xyz coordinates encode exact placement within that region.

---

### 3. VAE Bottleneck with Disentangled Latent Space

```
shape_embed [B, 384]          geom_tokens [B, 511, 384]
     |                               |
mu_s_proj_mean/var             pre_kl → kl_flat [B, 511×32]
     |                          kl_emb_proj_mean_g/var_g
     ↓                               ↓
mu_s [B, 512]   ──── concat ────  mu_g [B, 15872]
                        ↓
                 mu [B, 16384],  log_var [B, 16384]
                        ↓
              z = mu + σ·ε  (reparameterisation)
                        ↓
              Z = z.reshape(B, 512, 32)
```

The latent is explicitly split into two orthogonal subspaces:

- **`z_s` (semantic, dims 0–511):** derived from `shape_embed`. Encodes scene-level identity — dominant categories, color palette, spatial layout. 16 tokens when reshaped.
- **`z_g` (geometric, dims 512–16383):** derived from the 511 geometry tokens. Encodes per-region Gaussian arrangement. 496 tokens when reshaped.

This disentanglement is enforced by three complementary losses (cross-reconstruction, orthogonality, and semantic head supervision) described below.

---

### 4. Semantic Token Architecture (New — Inference-Clean Design)

**The core inference problem:** `shape_embed` only exists inside the VAE encoder. At second-stage diffusion inference, the DiT generates `Z [B, 512, 32]` directly, and there is no encoder to produce `shape_embed`. All prediction heads that take `shape_embed` as input break at inference.

**Solution:** Run all prediction heads on `z` token subsets extracted directly from the reshaped latent `Z`, *before* the decoder transformer. This makes the entire pipeline self-contained from `z` alone.

```
Z [B, 512, 32]
  ├── Z_color = Z[:, 0, :]            [B, 32]   — token 0
  │       └── MeanColorHead  → mean_color_pred  [B, 3]
  │
  ├── Z_sem   = Z[:, 1:16, :].flatten [B, 480]  — tokens 1–15
  │       ├── SceneSemanticHead → scene_label_dist  [B, 72]
  │       └── SceneLayoutHead  → category_centroids [B, 72, 3]
  │
  └── Z_geo   = Z[:, 16:, :]          [B, 496, 32] — tokens 16–511
          └── full geometric content
```

The gradient paths from each head supervision loss flow back through `z_s → reparameterisation → mu_s → encoder`, preserving unbiased gradients via the reparameterisation trick.

**Why token 0 for color and tokens 1–15 for semantic/layout:** Mean color is a 3-DOF scalar (single token sufficient). Scene category distribution and per-category centroids have higher information content (~288 DOF combined) and need the 15-token block (480 dims) to avoid an information bottleneck. Isolating the color gradient in its own token prevents it from competing with the semantic gradient in a shared representation.

**Second-stage inference pipeline (full, self-contained):**
```
1.  DiT samples Z [B, 512, 32]
2.  Z_color = Z[:, 0, :]
    Z_sem   = Z[:, 1:16, :].flatten(1)
3.  mean_color  = MeanColorHead(Z_color)       → [B, 3]
    layout_pred = SceneLayoutHead(Z_sem)        → [B, 72, 3]
4.  Set decoder conditioning: last_scene_layout_pred = layout_pred
5.  VAE_decode(Z) → color_residuals + abs_positions + scale + opacity + quat
6.  final_color = clamp(color_residuals + mean_color, 0, 1)
7.  Write PLY
```

No encoder. No `shape_embed`. No GT scaffold data.

---

### 5. Decoder

```
Z [B, 512, 32]
  → post_kl Linear(32 → 384)          [B, 512, 384]
  → (+ Fourier PE  OR  learnable PE)   [B, 512, 384]
  → (+ TokenCond B additive bias       [B, 512, 384]   if not AdaLN)
  → 12× Transformer self-attention (OR AdaLN-conditioned, see below)
  → Flatten  [B, 196608]
  → 8-layer MLP (W=1024)  →  raw [B, 40000, 14]
  → Activations:
      pos     → identity            (absolute positions ±10m)
      color   → identity            (residuals, ∈ [−0.5, +0.5])
      opacity → sigmoid             → (0, 1)
      scale   → exp                 → (0, +∞) metres
      quat    → L2-normalise        → unit quaternion
Output: [B, 40000, 14]  (absolute positions, color residuals)
```

---

### 6. Positional Encoding Options for Decoder Tokens

The 512 decoder tokens represent spatial regions (8×8×8 voxel grid). Two PE options:

#### Option A — Learnable PE (`decoder_pos_enc=True`)

```python
self.decoder_pos_emb = nn.Parameter(torch.zeros(512, width))
nn.init.trunc_normal_(self.decoder_pos_emb, std=0.02)
latents = latents + self.decoder_pos_emb.unsqueeze(0)
```

Simple and effective but learns spatial identity entirely from data. The 512 token positions are independent — token 0 at voxel `[0,0,0]` has no prior relationship with token 1 at `[0,0,1]`.

#### Option B — 3D Fourier PE (`decoder_fourier_pe=True`, recommended)

Precomputes 3D coordinates of all 512 voxels in the 8×8×8 grid, normalised to `[−1, 1]³`, and encodes them with the same FourierEmbedder used in the encoder:

```python
# Voxel (i, j, k) → normalised coordinate → Fourier features → Linear(→ width)
coords[t] = [(2i/(S−1))−1,  (2j/(S−1))−1,  (2k/(S−1))−1]  ∈ [−1,1]³
PE[t]     = Linear(FourierEmbedder(coords[t]))
```

Spatial proximity is encoded by construction: `‖PE[t] − PE[t′]‖ ∝ ‖coords[t] − coords[t′]‖`. The transformer is initialised with a local-spatial attention bias that matches the geometric structure of indoor scenes.

This is consistent with the encoder's dual Fourier embedder — both encoder and decoder use the same spectral basis for spatial information (spectral continuity).

Takes priority over learnable PE when both flags are True.

---

### 7. Decoder Conditioning via Scene Layout

#### Current Design — Additive Bias (TokenCond B)

A soft assignment matrix `W ∈ ℝ^{512×72}` maps each token to a weighted combination of the 72 predicted category centroids:

```python
W = softmax(token_cat_assign, dim=-1)              # [512, 72]  learnable
token_centroids = einsum('tk,bkd->btd', W, pred_c) # [B, 512, 3]
bias = TokenCondMLP(FourierEmbedder(token_centroids))  # [B, 512, 384]
latents = latents + bias                            # additive, before transformer
```

Provides semantic-spatial context to each token before self-attention. Limitation: applied only once before the 12-layer stack — the signal is diluted by each attention layer with no mechanism to re-inject it.

#### New Design — Per-Layer AdaLN-Zero (`token_cond_adaln=True`, recommended)

Replaces the once-before-stack additive bias with per-layer Adaptive Layer Normalisation inside each of the 12 transformer blocks. The same per-token semantic centroid signal (Fourier-encoded) is used, but now modulates every layer independently:

```
Standard LN:   γ · (h − μ)/σ + β         (fixed γ, β scalars)
AdaLN:         (1 + γ_c) ⊙ (h − μ)/σ + β_c    where [γ_c, β_c] = MLP(c)
AdaLN-Zero:    initialise MLP weights to zero → γ_c = 0, β_c = 0 at epoch 0
               → block = identity at init → safe to add to any checkpoint
```

For each of 12 layers, the conditioning MLP produces 6 vectors per token: `shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn`. The multiplicative scale component `(1 + γ_c)` can amplify or suppress individual feature dimensions based on the spatial-semantic context — this is not possible with additive bias alone.

Reference: DiT (Peebles & Xie, ICCV 2023) shows AdaLN-Zero reduces FID 10× over additive conditioning at identical parameter count and architecture depth.

Automatic fallback: if `last_scene_layout_pred` is None (layout head disabled or early training), the AdaLN transformer falls back to the standard transformer automatically.

---

## Auxiliary Supervision Heads

### Mean Color Head

```
input:  Z_color = z[:, :32]   [B, 32]   (semantic token heads mode)
     OR shape_embed            [B, 384]  (legacy mode)
head:   Linear(in → 64) → ReLU → Linear(64 → 3) → Sigmoid
output: mean_color_pred [B, 3]
loss:   MSE(mean_color_pred, GT_mean_color)
```

The GS_decoder predicts color *residuals* (deviations from the scene mean). Adding the predicted mean back recovers absolute colors at PLY save time. This solves shape_embed gradient starvation: without this head, the global scene token receives no gradient and collapses to near-zero.

### Scene Semantic Head

```
input:  Z_sem.flatten = z[:, 32:512]  [B, 480]  (semantic token heads mode)
     OR shape_embed                   [B, 384]   (legacy mode)
head:   Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear(→ 72) → Softmax
output: scene_label_dist [B, 72]
loss:   KL(GT_label_dist || predicted_dist)
```

Forces the semantic tokens to encode scene composition (bedroom vs. kitchen vs. corridor) as a probability distribution over ScanNet72 categories.

### Scene Layout Head

```
input:  Z_sem.flatten = z[:, 32:512]  [B, 480]  (semantic token heads mode)
     OR shape_embed                   [B, 384]   (legacy mode)
head:   Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear(→ 72×3)
output: category_centroids [B, 72, 3]  — per-category spatial centroid
loss:   masked MSE(predicted_centroids, GT_centroids)   (masked by category_valid)
```

Provides the TokenCond B conditioning signal. Forces the semantic tokens to encode where each semantic category is located in the scene, giving the decoder spatially grounded conditioning.

---

## Disentanglement Losses

Three losses enforce the `z_s / z_g` split:

### Cross-Reconstruction Loss

Swaps the semantic subspace between two scenes in the batch:

```python
z_s_B   = mu_s_shifted + σ_s_B · ε       # semantic from scene B
z_g_A   = mu_g + σ_g_A · ε               # geometry from scene A
z_cross = concat(z_s_B, z_g_A)           # mixed latent

# Decode and supervise geometric attributes only (position, opacity, scale, rotation)
UV_cross = decoder(z_cross)
L_cross_recon = ‖pos_pred − pos_gt‖ + ‖opacity_pred − opacity_gt‖ + ...
```

Geometric content of scene A must be recoverable from `z_g_A` regardless of which scene's semantic context it is paired with. Forces geometry-sufficiency of `z_g`.

### Orthogonality Loss

```python
p_s = F.normalize(mu_s[:, rand_idx_s], dim=0)   # [B, proj_dim]
p_g = F.normalize(mu_g[:, rand_idx_g], dim=0)   # [B, proj_dim]
L_ortho = ((p_s.T @ p_g) ** 2).mean()           # zero when mu_s ⊥ mu_g
```

Penalises linear correlation between the two subspaces. Random projection to 64 dims makes this computationally cheap.

### Semantic Head Losses

`L_color + L_semantic + L_layout` flowing through `z_s` tokens act as a structured regulariser, forcing the semantic subspace to organise into interpretable directions (color axis, category axis, layout axis) rather than arbitrary entangled features.

---

## Loss Function Summary

```
L_total = L_recon
        + kl_weight         × L_KL
        + color_weight      × L_color_mse
        + semantic_weight   × L_scene_kl
        + layout_weight     × L_layout_mse
        + cross_recon_weight× L_cross_recon
        + ortho_weight      × L_ortho
```

| Loss | Weight | Purpose |
|------|--------|---------|
| `L_recon` | 1.0 | Primary reconstruction of all 40k Gaussian attributes |
| `L_KL` | 1e-5 | VAE regularisation — keeps z near N(0,I) |
| `L_color_mse` | 1.0–3.0 | Mean color prediction from semantic token 0 |
| `L_scene_kl` | 0.3 | Scene label distribution from tokens 1–15 |
| `L_layout_mse` | 0.3 | Per-category centroid prediction from tokens 1–15 |
| `L_cross_recon` | 0.3 | Geometry sufficiency of z_g (disentanglement) |
| `L_ortho` | 0.1 | Linear independence of z_s and z_g |

---

## Gradient Path Summary

```
PATH 1 — Reconstruction (primary)
  L_recon → GS_decoder → post_kl → transformer → mu_g → kl_proj → encoder

PATH 2 — KL
  L_KL → mu, log_var → encoder

PATH 3 — Mean Color  (semantic tokens mode)
  L_color → MeanColorHead → z[:, :32] → mu_s → encoder

PATH 4 — Scene Semantic  (semantic tokens mode)
  L_scene_kl → SceneSemanticHead → z[:, 32:512] → mu_s → encoder

PATH 5 — Layout Centroids  (semantic tokens mode)
  L_layout → SceneLayoutHead → z[:, 32:512] → mu_s → encoder

PATH 6 — Cross-Reconstruction
  L_cross → decoder(z_cross) → gradient w.r.t. z_g (geometry sufficiency)

PATH 7 — Orthogonality
  L_ortho → mu_s, mu_g → both encoder branches
```

Paths 3, 4, 5 converge on `z_s` (the semantic subspace) through heads with orthogonal output spaces. Together they force `z_s` to encode the scene's color palette, semantic composition, and spatial layout simultaneously — three structurally independent pieces of information that together constitute the scene's semantic identity.

---

## Ablation Study

### Current Best Configuration

```
color_residual = True
scene_semantic_head = True
scene_layout_head = True
latent_disentangle = True,  semantic_dims = 512
token_cond = True,  token_cond_approach = B
decoder_fourier_pe = True
token_cond_adaln = True
semantic_token_heads = True
position_scaffold = False  (absolute position prediction)
kl_weight = 1e-5
```

**Validation L2 ≈ 0.79** at epoch 800, stable through epoch 1950.

### Completed Ablations

| Run | Config | Val L2 | Key Finding |
|-----|--------|--------|-------------|
| A | color_residual only | 1.43 | Baseline; shape_embed gradient starvation fixed |
| C | + scene_semantic_head | 1.80 | Alone hurts; needs color_residual as foundation |
| H | + disentangle + layout | 1.565 | Disentanglement beneficial but layout alone not enough |
| K | + position_layout_residual | ~1.0–1.2 | DC/AC position decomposition helps |
| P | + decoder_pos_enc (learnable) | 1.38 | PE helps tokens acquire spatial identity |
| Q | + predict_seg_labels | 1.54 | No benefit — categorical supervision insufficient without spatial decoder fix |
| R | + token_cond approach A | **0.589** | Largest single improvement — scaffold anchor Fourier bias into tokens |
| S | + token_cond approach B alone | unstable | KL explosion after epoch 1200 — circular optimisation |
| T | + token_cond approach AB | best visual | Both geometric + semantic spatial identity |
| T2 | T + trilinear anchor smoothing | 0.606 | Eliminated voxel boundary seam artifacts |
| **Current** | abs pos + Fourier PE + AdaLN + sem_tokens | **~0.79** | Clean training, best visualisation |

### Planned Ablations

The following systematic ablation is designed to isolate the contribution of each new component against a clean baseline. All runs use the same data split (300 train / 50 val), batch size 100, LR 1e-4, 2000 epochs.

| Run | `color_residual` | `scene_semantic` / `layout` | `latent_disentangle` | `decoder_fourier_pe` | `token_cond` (B) | `token_cond_adaln` | `semantic_token_heads` | Expected result |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Baseline** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | High L2, upper bound reference |
| **+PE** | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | Tests Fourier PE alone; tokens have spatial identity |
| **+Disentangle** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | Tests disentanglement + semantic heads (legacy path) |
| **+PE +Disent** | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | Fourier PE + disentanglement, no decoder conditioning |
| **+Cond (additive)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | Adds once-before-stack TokenCond B bias |
| **+Cond (AdaLN)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | AdaLN vs additive — key comparison |
| **Full** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full system with inference-clean semantic token heads |

**What each comparison measures:**

*Baseline → +PE:* Does spatial inductive bias in decoder PE (Fourier vs. none) matter even without conditioning?

*+PE → +PE +Disent:* Does disentanglement improve reconstruction quality when spatial PE is fixed?

*+PE +Disent → +Cond (additive):* Does TokenCond B conditioning help beyond PE alone?

*+Cond (additive) → +Cond (AdaLN):* Does per-layer modulation outperform once-before-stack bias? This is the key scientific comparison — DiT literature predicts yes, but the 3DGS setting is different.

*+Cond (AdaLN) → Full:* Does moving heads from shape_embed to z tokens hurt or help reconstruction? Hypothesis: no meaningful difference in L2, but enables the inference-clean pipeline.

### Job Script Configuration for Each Run

```bash
# Baseline
COLOR_RESIDUAL=False; SCENE_SEMANTIC_HEAD=False; LATENT_DISENTANGLE=False
DECODER_FOURIER_PE=False; DECODER_POS_ENC=False; TOKEN_COND=False
TOKEN_COND_ADALN=False; SEMANTIC_TOKEN_HEADS=False

# +PE only
DECODER_FOURIER_PE=True
(all others False)

# +Disentangle only
COLOR_RESIDUAL=True; SCENE_SEMANTIC_HEAD=True; SCENE_LAYOUT_HEAD=True
LATENT_DISENTANGLE=True; SEMANTIC_DIMS=512
(PE and conditioning False)

# +PE +Disent
COLOR_RESIDUAL=True; SCENE_SEMANTIC_HEAD=True; SCENE_LAYOUT_HEAD=True
LATENT_DISENTANGLE=True; DECODER_FOURIER_PE=True
(conditioning False)

# +Cond additive
# as above + TOKEN_COND=True; TOKEN_COND_APPROACH=B

# +Cond AdaLN  (key run)
# as above + TOKEN_COND_ADALN=True

# Full system
# all True + SEMANTIC_TOKEN_HEADS=True
```

---

## Training Configuration

```yaml
# Architecture (shapevae-256.yaml)
num_latents:       256        # → 512 tokens after +1 shape_embed prepend
embed_dim:         32         # token dimension in z
width:             384        # transformer hidden width
encoder_layers:    6
decoder_layers:    12
heads:             12
num_freqs:         8          # Fourier frequencies for positional embedders

# Training (gs_can3tok_2.py)
batch_size:        100        # per GPU; effective = 400 with 4× H100
learning_rate:     1e-4       # cosine decay to lr × 0.1
warmup_steps:      300
kl_weight:         1e-5
semantic_dims:     512        # z_s dimensions
cross_recon_weight: 0.3
ortho_weight:       0.1
layout_loss_weight: 0.3
scene_semantic_weight: 0.3
color_loss_weight:  3.0       # upweighted to prevent gray collapse
```

---

## Dataset

**SceneSplat-7K** — 7,916 indoor 3DGS scenes from ScanNet, ScanNet++, Replica, Hypersim, 3RScan, ARKitScenes, and Matterport3D. Each Gaussian is annotated with a ScanNet72 semantic label.

**Preprocessing:**
- Positions and scales normalised to a 10m radius canonical sphere (linear scale)
- Top-40k Gaussians sampled deterministically by opacity (same selection each epoch)
- Color: per-scene mean subtracted if `color_residual=True`, stored as residuals ∈ [−0.5, +0.5]
- Layout: per-category centroid positions computed per scene for SceneLayoutHead supervision

---

## Code Structure

```
.
├── gs_can3tok_2.py                           # Training loop, all loss functions, validation
├── gs_dataset_scenesplat.py                  # Dataset: preprocessing, scaffold, layout heads
├── semantic_losses.py                        # Per-Gaussian InfoNCE contrastive losses
├── gs_ply_reconstructor.py                   # Write decoder output to .ply (SuperSplat format)
├── pca_feature_visualization.py              # PCA coloring of decoder features
├── model/
│   ├── configs/aligned_shape_latents/
│   │   └── shapevae-256.yaml                 # Architecture hyperparameters
│   └── michelangelo/models/tsal/
│       ├── sal_perceiver_dist_changes.py     # Full model implementation:
│       │                                     #   CrossAttentionEncoder
│       │                                     #   AlignedShapeLatentPerceiver
│       │                                     #   MeanColorHead, SceneSemanticHead
│       │                                     #   SceneLayoutHead
│       │                                     #   FourierDecoderPE  (NEW)
│       │                                     #   AdaLNBlock, AdaLNTransformerDecoder (NEW)
│       │                                     #   AnchorPredFromTokens
│       │                                     #   GS_decoder (flat MLP)
│       └── asl_pl_module.py                  # PyTorch Lightning wrapper
└── job_scripts/
    ├── run_can3tok_scaffold.job              # SLURM job script with all ablation flags
    └── accelerate_config.yaml               # Accelerate DDP config (4× H100)
```

---

## Key Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--color_residual` | False | DC/AC color decomposition; MeanColorHead on z token 0 |
| `--scene_semantic_head` | False | Scene label distribution prediction |
| `--scene_layout_head` | False | Per-category centroid prediction; required for TokenCond B |
| `--latent_disentangle` | False | Split z into z_s (semantic) and z_g (geometric) subspaces |
| `--semantic_dims` | 512 | Dimensionality of z_s |
| `--cross_recon_weight` | 0.5 | Weight on cross-reconstruction disentanglement loss |
| `--ortho_weight` | 0.1 | Weight on orthogonality loss |
| `--decoder_pos_enc` | False | Learnable positional encoding on 512 decoder tokens |
| `--decoder_fourier_pe` | False | **NEW** 3D Fourier PE over 8³ voxel grid; overrides learnable PE |
| `--token_cond` | False | Inject spatial-semantic bias into decoder tokens |
| `--token_cond_approach` | A | A=scaffold anchors, B=category centroids, AB=both |
| `--token_cond_adaln` | False | **NEW** Per-layer AdaLN-Zero; requires token_cond + approach B |
| `--semantic_token_heads` | False | **NEW** Run heads on z tokens 0–15 instead of shape_embed |
| `--position_scaffold` | False | 8³ voxel scaffold for DC/AC position decomposition |
| `--kl_weight` | 1e-5 | KL regularisation weight (1e-6 insufficient, 1e-4 too strong) |

---

## Quick Start

```bash
# Full new architecture (all ablation flags on)
sbatch job_scripts/run_can3tok_scaffold.job
# Set in job script: DECODER_FOURIER_PE=True, TOKEN_COND_ADALN=True, SEMANTIC_TOKEN_HEADS=True

# Baseline (no improvements)
# Set: all new flags False, COLOR_RESIDUAL=False, SCENE_SEMANTIC_HEAD=False,
#      LATENT_DISENTANGLE=False, TOKEN_COND=False, DECODER_POS_ENC=False

# Fourier PE only
# Set: DECODER_FOURIER_PE=True, all others False

# Resume from checkpoint with AdaLN (safe due to zero-init)
# Set: RESUME_CHECKPOINT="/path/to/epoch_800.pth"
#      TOKEN_COND_ADALN=True
#      (AdaLN blocks start as identity, existing weights unaffected)
```