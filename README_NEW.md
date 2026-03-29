# Can3Tok VAE — Semantic-Aware 3D Scene Tokenizer

A Perceiver-based Variational Autoencoder that encodes full indoor 3DGS scenes into a structured latent space. The architecture combines DC/AC color and position decomposition, latent space disentanglement, scene-level semantic supervision, and spatial inductive bias in the decoder. Built on SceneSplat-7K as the foundation for generative scene completion.

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

**`shape_embed`** is the global scene token carrying overall scene identity. With standard training it collapses (no gradient), which motivated the DC/AC decomposition described below.

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

## Ablation Results Summary

| Run | Config | Val L2 |
|-----|--------|--------|
| A | color_residual only | 1.43 |
| C | + scene_semantic_head | 1.80 |
| H | + disentangle + layout | 1.565 |
| K | + position_layout_residual | ~1.0–1.2 |
| P | + decoder_pos_enc | 1.38 |
| Q | + predict_seg_labels | 1.54 (no benefit) |
| R | + token_cond approach A | **0.589** |
| S | + token_cond approach B | unstable (KL explosion) |
| T | + token_cond approach AB | best visual quality |
| T2 | T + trilinear anchor smoothing | 0.606 (no seam artifacts) |

---

## Concept 1 — DC/AC Color Residual Decomposition

### The Problem: `shape_embed` Gradient Starvation

`shape_embed` is intended as a global scene descriptor but receives almost no gradient during training because the 511 `latent_tokens` have sufficient capacity to explain all per-Gaussian variation without it. The result: `shape_embed` collapses to near-zero and encodes nothing.

### Solution: Two-Level Color Decomposition

Inspired by VQ-VAE-2's hierarchical latent structure:

- **Level 1 (DC — Global Color):** `shape_embed` → `MeanColorHead` → predicts the scene mean color `[B, 3]`
- **Level 2 (AC — Local Residuals):** `latent_tokens` / `mu` → encode per-Gaussian color deviations from the mean

### Implementation

**Dataset side (`gs_dataset_scenesplat.py`):**
```python
scene_mean_color = color.mean(axis=0)          # [3]  — DC component
color_residuals  = color - scene_mean_color    # [N, 3]  range ~ [-0.3, +0.3]
```

**Model side — `MeanColorHead`:**
```python
class MeanColorHead(nn.Module):
    def __init__(self, width=384):
        self.head = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid())    # output in [0, 1]

    def forward(self, shape_embed):
        return self.head(shape_embed)          # -> [B, 3]
```

**Loss side:**
```python
color_pred_loss = F.mse_loss(mean_color_pred, mean_color_gt)
L_total += lambda_color * color_pred_loss
```

This creates a direct gradient path: `MSE → MeanColorHead → shape_embed → encoder`.

---

## Concept 2 — Scene-Level Label Distribution Prediction

Gives `shape_embed` a second richer task beyond mean color: predict the **semantic composition** of the scene as a probability distribution over 72 ScanNet categories. A bedroom and a kitchen may have similar mean colors but very different semantic fingerprints.

### Ground Truth

For each scene, the empirical label distribution:

$$p_s^{(b)}[k] = \frac{\sum_{i=1}^{N} \mathbf{1}[y_i^{(b)} = k]}{N_{\text{valid}}^{(b)}}$$

### Implementation

**Dataset side:** a `[72]` histogram computed per scene, passed as `label_dist`.

**Model side — `SceneSemanticHead`:**
```python
class SceneSemanticHead(nn.Module):
    def __init__(self, width=384):
        self.head = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),   nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 72))

    def forward(self, shape_embed):
        return F.softmax(self.head(shape_embed), dim=-1)    # [B, 72]
```

**Loss side — KL divergence:**
```python
def scene_semantic_kl_loss(p_hat, p_s, eps=1e-8):
    """D_KL(p_s || p_hat) = Σ p_s * log(p_s / p_hat)"""
    p_hat_clamped = torch.clamp(p_hat, min=eps)
    return (p_s * (torch.log(p_s + eps) - torch.log(p_hat_clamped))).sum(dim=-1).mean()
```

---

## Concept 3 — Per-Gaussian Semantic InfoNCE Supervision

Per-Gaussian contrastive learning using ScanNet72 category labels. Operates on the **decoder hidden state** — a completely separate gradient path from Concepts 1 and 2.

```
For each batch:
  1. Extract per-Gaussian features from SemanticProjectionHead [B, 40k, 32]
  2. Subsample 10k Gaussians (balanced across categories)
  3. Compute category prototypes: mean feature per ScanNet72 class
  4. InfoNCE loss (temperature=0.1):
     L = -log[ exp(sim(f_i, proto_c) / τ) / Σ_j exp(sim(f_i, proto_j) / τ) ]
```

Effect: Gaussians from the same category cluster together in the 32-dim feature space; Gaussians from different categories are pushed apart.

---

## Concept 4 — Latent Disentanglement (mu_s / mu_g Split)

### The Problem

With standard training, `mu [B, 16384]` entangles semantic (what type of scene) and geometric (where things are) information. The decoder cannot separately access these two types of information, limiting its ability to generalise across scenes with the same layout but different semantic content.

### Solution: Split Latent into Semantic and Geometric Subspaces

```
mu [B, 16384]
  = concat(mu_s [B, D_s], mu_g [B, D_g])
    where D_s = 512 (semantic), D_g = 15872 (geometric)

mu_s  <- mu_s_proj_mean(shape_embed)    # from the global scene token
mu_g  <- kl_emb_proj_mean_g(kl_flat)   # from the 511 geometry tokens
```

### Three Complementary Losses

**1. Reconstruction loss** (standard): decoder sees the full `z = concat(z_s, z_g)`.

**2. Cross-reconstruction loss** enforces geometry-sufficiency of `mu_g`:
```python
# Swap semantic subspace between two scenes in the batch
z_s_swapped = mu_s_shifted + noise     # mu_s from scene B
z_g_current = mu_g + noise             # mu_g from scene A
z_cross = concat(z_s_swapped, z_g_current)

# Decode and compute reconstruction loss on geometric attributes only
UV_cross = decoder(z_cross)
L_cross_recon = ||pos_pred - pos_gt||² + ||opacity_pred - opacity_gt||² + ...
```

This forces `mu_g` to contain all geometry independently of which semantic context it is paired with. If the decoder can correctly place walls, floors, and furniture using `mu_g` from scene A even when paired with the semantic token from scene B, then `mu_g` is truly geometry-sufficient.

**3. Orthogonality loss** penalises linear correlation between `mu_s` and `mu_g`:
```python
def compute_orthogonality_loss(mu_s, mu_g, proj_dim=64):
    # Random projections for efficiency
    p_s = F.normalize(mu_s[:, idx_s], p=2, dim=0)
    p_g = F.normalize(mu_g[:, idx_g], p=2, dim=0)
    return ((p_s.T @ p_g) ** 2).mean()    # zero when mu_s ⊥ mu_g
```

### Total Disentanglement Loss

```
L_total += cross_recon_weight * L_cross_recon    # default 0.5
         + ortho_weight       * L_ortho           # default 0.1
```

---

## Concept 5 — Position DC/AC Decomposition

### The Problem: Position Has 20× Dynamic Range

The encoder compresses 40,000 positions spanning ±10m into the latent. The decoder must then reconstruct all positions from a single flat MLP output. Position loss converges slowly because the regression target has enormous dynamic range (±10m = 20m total span), while color converges easily because color residuals after Concept 1 span only ±0.3.

### Solution: Scene Layout Head + Category Centroids as DC

Inspired directly by Concept 1 (color residual), we decompose position into:

- **DC = per-category centroid** — the mean position of all Gaussians belonging to that ScanNet category in this scene (predicted by `SceneLayoutHead`)
- **AC = position residual** — the offset from the category centroid (~±0.5m)

```
absolute_pos[i] = category_centroid[label[i]] + position_residual[i]
```

The decoder only needs to predict offsets of ±0.5m instead of absolute positions of ±10m — a 20× reduction in dynamic range.

### SceneLayoutHead

```python
class SceneLayoutHead(nn.Module):
    """
    shape_embed -> [B, 72, 3] per-category spatial centroids.
    DC/AC for position — analogue of MeanColorHead for xyz.
    """
    def __init__(self, width=384):
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 72 * 3))

    def forward(self, shape_embed):
        return self.head(shape_embed).reshape(B, 72, 3)    # [B, 72, 3]
```

**Loss:**
```python
def compute_layout_loss(pred_centroids, gt_centroids, gt_valid):
    diff = (pred_centroids - gt_centroids) ** 2
    return (diff.mean(dim=-1) * gt_valid).sum() / (gt_valid.sum() + 1e-8)
```

**Dataset side:** the reconstruction target is swapped from absolute position to the residual:
```python
target[:, :, 0:3] = position_residuals    # coord - category_centroid, range ~±0.5m
```

**PLY save:** adds the centroid back to recover absolute position:
```python
absolute_pos = decoder_output_pos + dc_position    # dc_position[i] = centroid[label[i]]
```

---

## Concept 6 — Position Scaffold (8×8×8 Voxel Grid)

While Concept 5 uses semantic category centroids as the DC term, the scaffold approach uses a **spatial voxel grid** as the DC term. This is complementary and more stable at early training because it is purely geometric (no predicted centroids needed).

### Architecture

The 16m × 16m × 16m scene is divided into an **8×8×8 = 512 voxel grid**. Each voxel gets an anchor position = mean of all Gaussian positions within it. Each Gaussian is assigned to one voxel (hard assignment).

```python
scaffold_anchors:   [512, 3]    — voxel centroid positions
scaffold_token_ids: [40000]     — which voxel each Gaussian belongs to
position_offsets:   [40000, 3]  — coord - anchor  (the AC term)
```

The decoder predicts `position_offsets` (range ~±1m per voxel) instead of absolute positions (range ±10m).

### Trilinear Anchor Smoothing (Critical Fix)

The naive hard assignment creates **seam artifacts** at voxel boundaries. Two Gaussians 0.08m apart but on opposite sides of a voxel boundary have offset targets that differ by ~2m (the voxel width). The decoder must produce a discontinuous jump at every boundary — visible as grid-aligned artifacts in the output.

**Fix:** replace the hard voxel centroid lookup with trilinear interpolation over the 8 surrounding voxels:

```python
def trilinear_interpolate_anchors(coord, scaffold_anchors, scaffold_dims=8, domain_size=16.0):
    """
    For each Gaussian at position coord[i], compute a smoothly interpolated
    anchor from the 8 surrounding scaffold voxel centroids.

    smooth_anchor[i] = Σ_{corners} w_corner * anchor_corner
    where w_corner are trilinear weights (volume fractions, sum to 1).

    This makes the anchor a continuous function of position.
    Two Gaussians 0.08m apart get anchors that differ by only ~0.08m.
    """
    grid_coord = (coord + half_domain) / cell_size      # [N, 3] in [0, 8]
    i0 = np.floor(grid_coord).astype(np.int32)          # lower corner
    i1 = i0 + 1
    t  = (grid_coord - i0).astype(np.float32)           # fraction [0,1]

    tx = t[:, 0:1]; ty = t[:, 1:2]; tz = t[:, 2:3]

    smooth_anchor = (
        (1-tx)*(1-ty)*(1-tz) * a000 +
        (1-tx)*(1-ty)*   tz  * a001 +
        ...
        tx * ty * tz         * a111
    )
    return smooth_anchor    # [N, 3] — varies continuously across boundaries
```

This is the same fix used by Instant-NGP (Müller et al., SIGGRAPH 2022): without interpolation, grid-aligned discontinuities cause visible artifacts. With trilinear interpolation, the encoding is C0-continuous everywhere.

**Result:** voxel boundary seams eliminated. The 9× dynamic range reduction (absolute ±9m → smooth offset ±2.7m) makes position regression significantly easier.

**PLY save with smooth anchor:**
```python
# OLD (hard — seam artifacts):
all_preds[si, :, 0:3] += scaffold_anchors[scaffold_token_ids[si]]

# NEW (smooth — no seams):
all_preds[si, :, 0:3] += smooth_anchor[si]    # per-Gaussian, already interpolated
```

---

## Concept 7 — Spatial Inductive Bias in the Decoder

### The Root Problem

The decoder transformer treats 512 tokens as an **unordered set** with no spatial identity. Token 0 and token 400 look identical to self-attention unless their values differ. The subsequent flat MLP maps the flattened token sequence `[196608]` to 40,000 Gaussians using the same weights for all Gaussians — no spatial organisation, no permutation structure.

This is a fundamental architectural limitation: the decoder has no inductive bias to place Gaussians at spatially coherent locations.

The following ideas address this at three levels:

---

### Idea 0 — Decoder Positional Encoding

The cheapest fix. A learnable embedding `PE[512, width]` is added to the 512 decoder tokens **before** the self-attention transformer:

```python
self.decoder_pos_emb = nn.Parameter(torch.zeros(512, width))
nn.init.trunc_normal_(self.decoder_pos_emb, std=0.02)   # initialised near zero

# In decode():
latents = latents + self.decoder_pos_emb.unsqueeze(0)   # [B, 512, width]
```

Initialising near zero ensures this does not perturb existing weights when resuming from a checkpoint.

**Effect:** the transformer can now distinguish "token 0 is always the semantic region" from "token 400 is always a geometric region near the floor". Self-attention can learn structure-aware communication patterns.

**Ablation result (Run P vs K):** L2 improved from 1.0-1.2 to 1.38 when used alone, confirming tokens benefit from positional identity.

---

### Idea 1 — Segment Label Prediction (SegPredHead)

A lightweight per-Gaussian MLP that takes the decoder's 14-parameter Gaussian output and predicts which ScanNet category it belongs to:

```python
class SegPredHead(nn.Module):
    def __init__(self, in_dim=14, num_cats=72):
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, num_cats))    # [B, 40000, 72] raw logits

    def forward(self, gaussian_params):
        # gaussian_params: [B, 40000, 14]
        return self.head(gaussian_params.reshape(B*N, 14)).reshape(B, N, 72)
```

**Why this works:** the only way `SegPredHead` can correctly predict "this is a floor Gaussian" from the 14 Gaussian params is if the decoder has actually placed it at a floor-like position (low y-coordinate). The cross-entropy loss creates a backpropagation path that forces the decoder to produce **spatially discriminative** positions.

**Loss:**
```python
def compute_seg_pred_loss(seg_logits, segment_labels):
    # seg_logits:     [B, 40000, 72]  raw logits
    # segment_labels: [B, 40000]      gt labels (>= 0 valid, -1 unlabelled)
    valid = flat_labels >= 0
    return F.cross_entropy(flat_logits[valid], flat_labels[valid])
```

**At inference (no labels needed):** the predicted logits can soft-lookup the DC centroid:
```python
dc_i = Σ_k  softmax(seg_logits_i)[k] × pred_centroids[k]    # predicted DC position
absolute_pos_i = decoder_pos_i + dc_i
```

**Ablation result (Run Q):** no L2 improvement over baseline. The CE loss (SegPredCE dropped from 4.32 to 1.14, confirming learning), but the primary reconstruction loss did not benefit. The categorical supervision alone is insufficient without fixing the decoder's ability to produce spatially-resolved outputs per-Gaussian.

---

### Idea 2 — Token Centroid Conditioning

Injects spatial identity into each of the 512 decoder transformer tokens **before** self-attention by adding a Fourier-encoded spatial reference vector. After conditioning, the transformer can learn to exploit spatial structure in its self-attention patterns.

Applied before the transformer so the spatial context propagates through all 12 attention layers.

#### Approach A — Scaffold Anchor Conditioning

Each token receives the Fourier encoding of its corresponding voxel centroid:

```python
# scaffold_anchors: [B, 512, 3]
fourier_A  = self.fourier_embedder(scaffold_anchors)    # [B, 512, fourier_dim]
spatial_A  = self.token_cond_mlp_A(fourier_A)           # [B, 512, width]
latents    = latents + spatial_A
```

- Deterministic DC term, available from epoch 0
- Purely spatial (no semantic information)
- Requires `position_scaffold=True`

#### Approach B — Category Centroid Conditioning

A learnable soft assignment matrix `W [512, 72]` maps each token to a weighted combination of the 72 category centroids predicted by `SceneLayoutHead`:

```python
W = F.softmax(self.token_cat_assign, dim=-1)        # [512, 72] — learned
token_centroids = torch.einsum('tk,bkd->btd', W, pred_c)   # [B, 512, 3]
fourier_B  = self.fourier_embedder(token_centroids)
spatial_B  = self.token_cond_mlp_B(fourier_B)      # [B, 512, width]
latents    = latents + spatial_B
```

- Semantically grounded — tokens know which categories they represent
- `W` is scene-agnostic but adapts as `SceneLayoutHead` learns
- Requires `scene_layout_head=True`
- Unstable when used alone (Run S): KL explosion after epoch 1200

#### Approach AB — Both Additively

```python
latents = latents + spatial_A + spatial_B
```

**Best result**: visually cleanest reconstruction. Approach A provides stable geometric grounding from epoch 0; Approach B adds semantic organisation as `SceneLayoutHead` learns. Combined they give both geometric and semantic spatial identity to each token.

**Ablation results:** Run R (A only) L2=0.589; Run T (AB) best visual quality with comparable L2.

---

### Idea 3 — Spatial-Aware Per-Gaussian Decoder (Query Decoder)

Replaces the flat GS_decoder MLP with a per-Gaussian decoder that gives each Gaussian its own spatial context:

```python
class SpatialAwareDecoder(nn.Module):
    def forward(self, transformer_tokens, scaffold_anchors, scaffold_token_ids):
        # For each Gaussian i:
        # 1. Gather its token's features
        idx = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, D)
        token_feats = torch.gather(transformer_tokens, 1, idx)    # [B, N, D]

        # 2. Fourier-encode its voxel anchor
        anchor_i    = torch.gather(scaffold_anchors, 1, idx_3d)   # [B, N, 3]
        spatial_enc = self.fourier_embedder(anchor_i)              # [B, N, fourier_dim]

        # 3. Combine and decode
        combined = torch.cat([token_feats, spatial_enc], dim=-1)  # [B, N, D+fourier_dim]
        raw      = self.mlp(combined.reshape(B*N, -1))             # [B, N, 14]
```

Each Gaussian now knows (a) which spatial region it belongs to (from Fourier anchor encoding) and (b) what the transformer learned about that region (from the gathered token features). The MLP is ~350K parameters versus ~800M for the flat GS_decoder.

**Ablation result (Run U):** no L2 improvement even after 1500 epochs. The hard token assignment (`scaffold_token_ids` is non-differentiable) and information bottleneck (78 Gaussians share the same token features) prevent learning. The trilinear anchor smoothing fixed the position target discontinuity but not the feature discontinuity — adjacent Gaussians across a voxel boundary still receive completely different token features, making the MLP's task impossible.

**DeepSeek analysis** (agreed): Fix B (residual query decoder) is the correct next step — keep GS_decoder output as the base and add the query decoder as a small learned refinement. This ensures the model never regresses from baseline performance.

---

## Full Gradient Path Summary

After all concepts, the model has six independent gradient paths:

```
PATH 1 — Reconstruction (primary)
  L_recon -> GS_decoder -> post_kl -> transformer -> mu / latent_tokens

PATH 2 — KL Regularisation
  L_KL -> mu, log_var -> encoder

PATH 3 — Mean Color (Concept 1)
  L_color_mse -> MeanColorHead -> shape_embed -> encoder token 0

PATH 4 — Scene Semantic Distribution (Concept 2)
  L_scene_kl -> SceneSemanticHead -> shape_embed -> encoder token 0

PATH 5 — Layout Centroids (Concept 5)
  L_layout_mse -> SceneLayoutHead -> shape_embed -> encoder token 0

PATH 6 — Per-Gaussian InfoNCE (Concept 3, optional)
  L_infonce -> SemanticProjectionHead -> decoder hidden state
```

Paths 3, 4, 5 all converge on `shape_embed` through separate heads with separate output spaces. They force `shape_embed` to encode: (3) scene colour palette, (4) dominant category mix, (5) spatial layout of each category.

---

## Training Configuration

```python
# Architecture
num_latents       = 256
embed_dim         = 32
width             = 384
encoder_layers    = 6
decoder_layers    = 12

# Optimisation
learning_rate     = 1e-4
batch_size        = 64
kl_weight         = 1e-6    # 1e-5 causes KL to dominate at late epochs

# Concept 1: Color residual
color_residual          = True
mean_color_weight       = 1.0

# Concept 2: Scene semantic head
scene_semantic_head     = True
scene_semantic_weight   = 0.3

# Concept 3: Per-Gaussian InfoNCE (optional)
semantic_mode           = 'hidden'
segment_loss_weight     = 0.3
semantic_temperature    = 0.07
semantic_subsample      = 10000

# Concept 4: Latent disentanglement
latent_disentangle      = True
semantic_dims           = 512     # D_s; D_g = 16384 - 512 = 15872
cross_recon_weight      = 0.5
ortho_weight            = 0.1

# Concept 5: Position DC/AC (layout residual)
scene_layout_head       = True
layout_loss_weight      = 0.3
position_layout_residual = True   # mutually exclusive with position_scaffold

# Concept 6: Position scaffold + trilinear smoothing
position_scaffold       = True
anchor_loss_weight      = 0.3
# Trilinear smoothing: always active when position_scaffold=True
# No flag needed — gs_dataset_scenesplat.py computes smooth_anchor automatically

# Concept 7: Spatial inductive bias
decoder_pos_enc         = True    # Idea 0
predict_seg_labels      = False   # Idea 1 — no benefit in ablation
token_cond              = True
token_cond_approach     = 'AB'    # Idea 2 — best visual result
query_decoder           = False   # Idea 3 — pending residual fix
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

**Normalization:** positions and scales normalized to fit a 10m radius sphere (linear scale, not log-space — log-space causes scale collapse to <1cm due to domain mismatch).

**Sampling:** top 40,000 Gaussians by opacity, deterministic argsort (same selection every epoch).

---

## Batch Dictionary Keys

```python
{
    # Core features
    'features':           [B, 40000, 18],   # full feature tensor (abs xyz in cols 4:7)
    'segment_labels':     [B, 40000],        # ScanNet72 per-Gaussian labels
    'instance_labels':    [B, 40000],        # instance ids
    'mean_color':         [B, 3],            # scene mean color (for MeanColorHead supervision)
    'label_dist':         [B, 72],           # scene label distribution (for SceneSemanticHead)

    # Position scaffold (position_scaffold=True)
    'scaffold_anchors':   [B, 512, 3],       # voxel centroid positions (8x8x8 grid)
    'scaffold_token_ids': [B, 40000],        # hard voxel assignment per Gaussian
    'position_offsets':   [B, 40000, 3],     # coord - smooth_anchor  (TRILINEAR, the AC term)
    'smooth_anchor':      [B, 40000, 3],     # per-Gaussian trilinear anchor (for PLY save)

    # Position layout residual (position_layout_residual=True)
    'category_centroids': [B, 72, 3],        # per-category mean positions
    'category_valid':     [B, 72],           # which categories are present
    'dc_position':        [B, 40000, 3],     # category centroid per Gaussian
    'position_residuals': [B, 40000, 3],     # coord - dc_position  (the AC term)

    # JEPA Idea 1 (jepa_idea1=True)
    'voxel_label_dists':  [B, 512, 72],      # per-voxel label distribution
    'voxel_valid':        [B, 512],          # which voxels are occupied
}
```

---

## Code Structure

```
.
├── gs_can3tok_2.py                                      # Training loop, all losses
├── gs_dataset_scenesplat.py                             # Dataset, DC/AC preprocessing,
│                                                        #   trilinear anchor smoothing,
│                                                        #   label_dist precomputation
├── semantic_losses.py                                   # ScanNet72SemanticLoss (InfoNCE)
├── gs_ply_reconstructor.py                              # Output to .ply (SuperSplat/CloudCompare)
├── pca_feature_visualization.py                         # PCA coloring of semantic features
├── model/
│   ├── configs/aligned_shape_latents/shapevae-256.yaml  # Architecture config
│   └── michelangelo/models/tsal/
│       ├── sal_perceiver_dist_changes.py                # Full model: Encoder, Decoder,
│       │                                                #   MeanColorHead, SceneSemanticHead,
│       │                                                #   SceneLayoutHead, AnchorPositionHead,
│       │                                                #   SegPredHead, TokenCondMLP,
│       │                                                #   SpatialAwareDecoder,
│       │                                                #   decoder_pos_emb (PE),
│       │                                                #   latent disentanglement
│       └── asl_pl_module.py                             # PyTorch Lightning wrapper
│                                                        #   (updated to pass scaffold data)
└── job_scripts/
    └── run_can3tok_scaffold.job                         # SLURM job script (all ablations)
```

---

## Quick Start

```bash
# Run T2: full best config (token_cond AB + trilinear smoothing)
sbatch job_scripts/run_can3tok_scaffold.job
# Set in job script: TOKEN_COND=True, TOKEN_COND_APPROACH=AB, DECODER_POS_ENC=True
# Trilinear smoothing is always active when POSITION_SCAFFOLD=True

# Run R: scaffold + token_cond A only (best L2 = 0.589)
# Set: TOKEN_COND=True, TOKEN_COND_APPROACH=A

# Run K: position layout residual baseline (no scaffold)
# Set: POSITION_LAYOUT_RESIDUAL=True, POSITION_SCAFFOLD=False

# Run A: color residual only (baseline L2 = 1.43)
python gs_can3tok_2.py \
    --color_residual --semantic_mode none \
    --batch_size 64 --num_epochs 1500 --lr 1e-4 --kl_weight 1e-6 \
    --train_scenes 2000 --eval_every 50
```

---

## Key Design Decisions and

**Color DC/AC (Concept 1) was essential.** Without it, `shape_embed` receives no gradient and collapses. Every subsequent improvement builds on this foundation.

**Scaffold beats category residual for position.** Position layout residual (Concept 5) reduces dynamic range by 20× but requires accurate category centroids from `SceneLayoutHead`. The scaffold (Concept 6) with trilinear smoothing achieves a 9× reduction with no semantic dependency and no seam artifacts. Scaffold L2 (0.589) significantly beats layout residual L2 (~1.0).

**Token conditioning (Idea 2, Approach A) is the largest single improvement.** Injecting scaffold anchor Fourier encodings into decoder tokens before self-attention gives the transformer spatial structure to organise, reducing L2 from ~1.38 (PE only) to 0.589. This is the key architectural finding.

**Trilinear smoothing is necessary for visual quality.** Hard voxel assignment creates 2m jumps in position targets at boundaries, producing visible grid-aligned seams. Trilinear blending of 8 surrounding anchors makes the target continuous (C0-continuous), eliminating seams without any model change.

**Query decoder (Idea 3) did not learn.** The hard `gather` operation is non-differentiable across voxel boundaries and the information bottleneck (78 Gaussians per token) prevents the small MLP from producing diverse outputs. The residual query decoder (GS_decoder + small learned refinement) is the correct next step.

**Approach B (category centroid conditioning) is unstable alone** but beneficial when combined with Approach A. Alone, the dependency on predicted centroids creates a circular optimization problem that leads to KL explosion after ~1200 epochs.