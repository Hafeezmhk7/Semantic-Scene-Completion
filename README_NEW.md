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

## Full Architecture Diagram

```
════════════════════════════════════════════════════════════════════════════════════════
                           CAN3TOK VAE — FULL PIPELINE
════════════════════════════════════════════════════════════════════════════════════════

INPUT
┌────────────────────────────────────────────────────────────────────┐
│  40,000 Gaussians × 18 features                [B, 40000, 18]      │
│                                                                    │
│  cols 0:3   voxel_center  (coarse 40³ grid position)               │
│  col  3     voxel_id      (encoder voxel index)                    │
│  cols 4:7   xyz           (absolute Gaussian position)             │
│  cols 7:10  rgb           (color or color residual)                │
│  col  10    opacity                                                │
│  cols 11:14 scale                                                  │
│  cols 14:18 quaternion                                             │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
════════════════════════════════════════════════════════════════════════
  ENCODER — CrossAttentionEncoder
════════════════════════════════════════════════════════════════════════
                               │
        ┌──────────────────────┼────────────────────┐
        │                      │                    │
        ▼                      ▼                    ▼
  voxel_centers          xyz_actual          gaussian_params
  [B, 40000, 3]         [B, 40000, 3]     (opacity,scale,quat)
        │                      │              [B, 40000, 8]
        ▼                      ▼
  FourierEmbedder        FourierEmbedder
  (coarse PE,            (fine PE,
   8 freqs, π-inc)        8 freqs, π-inc)
  out_dim = 51           out_dim = 51
        │                      │
        └──────────┬───────────┘
                   │ concat [51 | 51 | 8] = 110 dims
                   ▼
             Linear(110 → 384)              [B, 40000, 384]
                   │
                   │  512 learned queries   [512, 384]  (init: voxel grid + noise)
                   ▼
          Cross-Attention                   [B, 512, 384]
          (queries attend over 40k pts)
                   │
                   ▼
          6× Self-Attention Transformer     [B, 512, 384]
          (width=384, heads=12)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   token 0                tokens 1–511
   shape_embed             geom_tokens
   [B, 384]                [B, 511, 384]
   (global scene           (per-region
    descriptor)             geometry)

════════════════════════════════════════════════════════════════════════
  VAE BOTTLENECK — Disentangled Latent Space
════════════════════════════════════════════════════════════════════════

   shape_embed             geom_tokens
   [B, 384]                [B, 511, 384]
        │                       │
        ▼                       ▼
   mu_s_proj_mean/var      pre_kl Linear(384 → 64)
   Linear(384 → 512)       [B, 511, 64]
        │                       │  flatten
        │                  kl_flat [B, 511×32 = 16352]
        ▼                       │
   mu_s  [B, 512]               ▼
   lv_s  [B, 512]       kl_emb_proj_mean_g/var_g
                         Linear(16352 → 15872)
                                │
                                ▼
                         mu_g  [B, 15872]
                         lv_g  [B, 15872]
        │                       │
        └──────── concat ───────┘
                      │
                      ▼
               mu  [B, 16384]
               lv  [B, 16384]
                      │  reparameterisation: z = mu + σ·ε
                      ▼
               z   [B, 16384]
                      │  reshape
                      ▼
               Z   [B, 512, 32]
                      │
                      ├── z_s = Z[:, 0:16,  :]   [B, 16,  32]   ← semantic tokens
                      └── z_g = Z[:, 16:512, :]  [B, 496, 32]   ← geometry tokens

════════════════════════════════════════════════════════════════════════
  PREDICTION HEADS (semantic_token_heads=True — inference-clean)
════════════════════════════════════════════════════════════════════════

   z_s = Z[:, 0:16, :]  [B, 16, 32]
   │
   ├── Z_color = Z[:, 0, :]              [B, 32]
   │   └── MeanColorHead                                    LOSS: MSE
   │       Linear(32→64) → ReLU → Linear(64→3) → Sigmoid
   │       → mean_color_pred              [B, 3]
   │
   └── Z_sem = Z[:, 1:16, :].flatten     [B, 480]
       ├── SceneSemanticHead                                LOSS: KL
       │   Linear(480→128) → LN → ReLU → ×2 → Linear(→72) → Softmax
       │   → scene_label_dist             [B, 72]
       │
       └── SceneLayoutHead                                  LOSS: masked MSE
           Linear(480→512) → LN → ReLU → Linear(→256) → LN → ReLU
           → Linear(→72×3)
           → category_centroids           [B, 72, 3]

════════════════════════════════════════════════════════════════════════
  DECODER — NEW DESIGN: z_s Cross-Attention Conditioning
  (decoder_zs_cross_attn=True, the main new idea)
════════════════════════════════════════════════════════════════════════

   z_g [B, 496, 32]                     z_s [B, 16, 32]
        │                                     │
        ▼                                     ▼
  post_kl_g                             post_kl_s
  Linear(32 → 384)                      Linear(32 → 384)
  [B, 496, 384]                         [B, 16, 384]
        │                                     │
        ▼                                     │
  + FourierPE                                 │
    (8³ grid, last 496 voxels)                │
    [B, 496, 384]                             │
        │                                     │
        ▼         (K and V in every layer)    │
  ┌─────────────────────────────────────────┐ │
  │  ZSCondTransformerDecoder  ×12 layers   │ │
  │  ┌─────────────────────────────────┐    │ │
  │  │ Self-Attention(H_g, H_g, H_g)  │    │ │
  │  │   z_g attends to z_g           │    │ │
  │  │                                │    │ │
  │  │ Cross-Attention(Q=H_g,         │◄───┼─┘
  │  │               K=H_s, V=H_s)   │    │
  │  │   z_g reads from z_s           │    │
  │  │                                │    │
  │  │ FFN(H)                         │    │
  │  └─────────────────────────────────┘    │
  └─────────────────────────────────────────┘
        │
        ▼
  H_out [B, 496, 384]
        │  flatten
        ▼
  [B, 496 × 384] = [B, 190,464]
        │
        ▼
  GS_decoder (8-layer MLP, W=1024)
  Layer 0:  Linear(190464 → 1024) + LN + ReLU
  Layers 1–7: Linear(1024 → 1024) + LN + ReLU
  Output:   Linear(1024 → 40000×14)
        │
        ▼
  raw [B, 40000, 14]  →  activations:
  ├── pos     [B, 40000, 3]   identity         (absolute, ±10m)
  ├── color   [B, 40000, 3]   identity         (residuals ∈[−0.5,+0.5])
  ├── opacity [B, 40000, 1]   sigmoid          → (0, 1)
  ├── scale   [B, 40000, 3]   exp              → (0, +∞) m
  └── quat    [B, 40000, 4]   L2-normalise     → unit quaternion
        │
        ▼
  OUTPUT: [B, 40000, 14]

═══════════════════════════════════════════════════════════════════════
  WHY CROSS-ATTENTION, NOT ADALN
═══════════════════════════════════════════════════════════════════════

  AdaLN injects z_s into every affine computation in the decoder.
  Result (Run 1): swap ratio >400× — geometry became completely
  structurally dependent on z_s semantics, destroying disentanglement.

  Cross-attention is a soft gate:
    attention weights can → 0 if z_g is geometry-sufficient.
    The decoder consults z_s when helpful; it is never forced to.
  Result: swap ratio ≈1× — geometry preserved under semantic swap.

  Gradient flow under new design:
    L_recon → GS_decoder → ZSCondDecoder(self-attn path) → post_kl_g → z_g
    L_recon → GS_decoder → ZSCondDecoder(cross-attn weights) → post_kl_s → z_s

═══════════════════════════════════════════════════════════════════════
  LEGACY DECODER (decoder_zs_cross_attn=False, backward compatible)
═══════════════════════════════════════════════════════════════════════

  Z [B, 512, 32]
    → post_kl Linear(32 → 384)           [B, 512, 384]
    → (+ FourierPE over 512 voxels  OR  learnable PE [512, 384])
    → (+ TokenCond B additive bias — once before stack)
    → 12× Self-Attention Transformer  OR  AdaLN-conditioned
    → flatten [B, 196,608]
    → GS_decoder (same 8-layer MLP, larger input)
    → [B, 40000, 14]

════════════════════════════════════════════════════════════════════════
  GRADIENT PATH SUMMARY
════════════════════════════════════════════════════════════════════════

  PATH 1 — Reconstruction (primary, new design):
    L_recon → GS_decoder → ZSCondDecoder → post_kl_g → z_g → encoder

  PATH 2 — KL:
    L_KL → mu, log_var → encoder

  PATH 3 — Mean Color:
    L_color → MeanColorHead → Z[:, 0, :] → mu_s[:, :32] → encoder

  PATH 4 — Scene Semantic:
    L_sem_kl → SceneSemanticHead → Z[:, 1:16, :].flatten → mu_s[32:512] → encoder

  PATH 5 — Layout Centroids:
    L_layout → SceneLayoutHead → Z[:, 1:16, :].flatten → mu_s[32:512] → encoder

  PATH 6 — Cross-Attn (new design, replaces L_cross_recon as primary enforcer):
    L_recon flows through cross-attn weights → post_kl_s → z_s → encoder

  PATH 7 — Cross-Reconstruction (geometry sufficiency):
    z_cross = [z_s_B | z_g_A] → decode() → L_cross_recon
    Gradient reaches z_g — forces geometry-sufficiency of z_g.

  PATH 8 — Scene z_s InfoNCE:
    L_z_s_nce → SemanticTokenInfoNCEHead → z[:, :512] → mu_s → encoder

════════════════════════════════════════════════════════════════════════
  SECOND-STAGE INFERENCE PIPELINE (self-contained, no encoder needed)
════════════════════════════════════════════════════════════════════════

  1. Small DiT samples  z_s [B, 16, 32]   (conditioned on text/class)
  2. Main DiT samples   z_g [B, 496, 32]  (cross-attn conditioned on z_s)
  3. Assemble:          Z = concat(z_s, z_g) → [B, 512, 32]
  4. Run heads:
       mean_color  = MeanColorHead(Z[:, 0, :])          → [B, 3]
       layout_pred = SceneLayoutHead(Z[:, 1:16, :].flat) → [B, 72, 3]
  5. decode(Z):   z_g → decoder, z_s → cross-attn K/V
  6. final_color = clamp(color_residuals + mean_color, 0, 1)
  7. Write PLY

  No encoder. No shape_embed. No GT scaffold data.
```

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
  xyz_actual    → FourierEmbedder (num_freqs=8, input_dim=3)  → fine PE    [B, 40000, 51]
  voxel_centers → FourierEmbedder (num_freqs=8, input_dim=3)  → coarse PE  [B, 40000, 51]
  gaussian_params (opacity, scale, quat)                      → raw 8-dim  [B, 40000, 8]

  Concatenate [fine_PE | coarse_PE | gaussian_params]  → Linear(110 → 384) [B, 40000, 384]

  512 learned queries  →  Cross-attention over all 40k Gaussians   [B, 512, 384]
                       →  6× Self-attention transformer layers      [B, 512, 384]

Output: [B, 512, 384]
  split: shape_embed [B, 384]      ← token 0, global scene descriptor
         geom_tokens [B, 511, 384] ← tokens 1–511, per-region geometry
```

---

### 3. VAE Bottleneck with Disentangled Latent Space

```
shape_embed [B, 384]          geom_tokens [B, 511, 384]
     │                               │
mu_s_proj_mean/var             pre_kl → kl_flat [B, 511×32 = 16352]
Linear(384 → 512)              kl_emb_proj_mean_g/var_g
     │                         Linear(16352 → 15872)
     ▼                               ▼
mu_s [B, 512]   ──── concat ────  mu_g [B, 15872]
                        ▼
                 mu [B, 16384],  log_var [B, 16384]
                        ▼
              z = mu + σ·ε     (reparameterisation)
                        ▼
              Z = z.reshape(B, 512, 32)   [B, 512, 32]
                        │
              ├── z_s = Z[:, 0:16,  :]   [B, 16,  32]   ← 16 semantic tokens
              └── z_g = Z[:, 16:512, :]  [B, 496, 32]   ← 496 geometry tokens
```

The latent is explicitly split into two orthogonal subspaces:

- **`z_s` (semantic, tokens 0–15, dims 0–511):** derived from `shape_embed`. Encodes scene-level identity — dominant categories, color palette, spatial layout. 16 tokens when reshaped.
- **`z_g` (geometric, tokens 16–511, dims 512–16383):** derived from the 511 geometry tokens. Encodes per-region Gaussian arrangement. 496 tokens when reshaped.

---

### 4. Semantic Token Architecture (Inference-Clean Design)

All prediction heads operate on `z` token subsets extracted from the reshaped latent `Z` directly, making the entire pipeline self-contained from `z` alone — no encoder or `shape_embed` needed at inference.

```
Z [B, 512, 32]
  ├── Z_color = Z[:, 0, :]            [B, 32]   — token 0
  │       └── MeanColorHead  → mean_color_pred  [B, 3]
  │
  ├── Z_sem   = Z[:, 1:16, :].flatten [B, 480]  — tokens 1–15
  │       ├── SceneSemanticHead → scene_label_dist   [B, 72]
  │       └── SceneLayoutHead  → category_centroids  [B, 72, 3]
  │
  └── Z_geo   = Z[:, 16:, :]          [B, 496, 32] — tokens 16–511
          └── full geometric content → decoder
```

---

### 5. Decoder — New z_s Cross-Attention Design

The main architectural innovation. z_g is the **only** decoder input sequence; z_s conditions every transformer layer via cross-attention.

```
z_g [B, 496, 32]  →  post_kl_g Linear(32→384)  →  H_g [B, 496, 384]
                                                 →  + FourierPE (voxels 16–511)
z_s [B, 16,  32]  →  post_kl_s Linear(32→384)  →  H_s [B, 16,  384]  (K and V only)

For each of 12 transformer layers:
  H = LayerNorm(H_g)
  H = H_g + Self-Attention(H, H, H)            ← z_g attends to z_g
  H = H  + Cross-Attention(Q=H, K=H_s, V=H_s)  ← z_g reads from z_s
  H = H  + FFN(LayerNorm(H))

H_out [B, 496, 384]
  → flatten  [B, 190464]
  → GS_decoder 8-layer MLP (W=1024)
  → raw [B, 40000, 14]
  → activations → [B, 40000, 14]
```

**Why cross-attention instead of AdaLN:** Run 1 with AdaLN showed swap ratio >400× — geometry became structurally dependent on z_s, destroying disentanglement. Cross-attention is a soft gate: attention weights can approach zero if z_g is geometry-sufficient. The decoder consults z_s when helpful; it is never forced to depend on it for geometry.

---

### 6. Positional Encoding for Decoder Tokens

#### 3D Fourier PE (`decoder_fourier_pe=True`, recommended)

Precomputes 3D coordinates of voxels in the 8×8×8 grid, normalised to `[−1, 1]³`:

```python
coords[t] = [(2i/(S−1))−1,  (2j/(S−1))−1,  (2k/(S−1))−1]  ∈ [−1,1]³
PE[t]     = Linear(FourierEmbedder(coords[t]))   # shared embedder with encoder
```

With new decoder design, PE applies to the 496 z_g voxels (voxels 16–511 of the grid). Spatial proximity encoded by construction: adjacent voxels have similar PE vectors.

#### Learnable PE (`decoder_pos_enc=True`)

```python
self.decoder_pos_emb = nn.Parameter(torch.zeros(512, width))
```

Simpler but learns spatial identity entirely from data with no geometric prior.

---

## Disentanglement Losses

Three losses enforce the `z_s / z_g` split:

### Cross-Reconstruction Loss

Swaps the semantic subspace between two scenes in the batch. With the new decoder design, `z_cross = [z_s_B | z_g_A]` is reshaped to `[B, 512, 32]` and passed to `decode()`, which internally splits it: z_s from scene B goes into cross-attention conditioning, z_g from scene A goes into the decoder sequence.

```python
z_s_B   = mu_s_shifted + σ_s_B · ε       # semantic from scene B
z_g_A   = mu_g + σ_g_A · ε               # geometry from scene A
z_cross = concat(z_s_B, z_g_A)           # → [B, 16384] → reshape → [B, 512, 32]

UV_cross = decoder(z_cross)               # z_s_B → cross-attn K/V; z_g_A → sequence
L_cross_recon = ‖pos_pred − pos_gt‖ + ‖opacity_pred − opacity_gt‖ + ...
```

### Orthogonality Loss

```python
p_s = F.normalize(mu_s[:, rand_idx_s], dim=0)   # [B, 64]
p_g = F.normalize(mu_g[:, rand_idx_g], dim=0)   # [B, 64]
L_ortho = ((p_s.T @ p_g) ** 2).mean()
```

### Scene-Level z_s InfoNCE (optional, `z_s_infonce_weight > 0`)

```python
# SemanticTokenInfoNCEHead: z_s [B,512] → L2-norm [B,128]  (no LN between layers)
# Positive pairs: scenes with cosine_sim(label_dist_i, label_dist_j) > delta
# Formulation: SupCon generalisation with per-anchor normalised soft weights
z_s_proj = SemanticTokenInfoNCEHead(z[:, :512])   # [B, 128]
w_ij = clamp(cos_sim(label_dist_i, label_dist_j) - delta, min=0)
L_z_s_nce = -sum_j norm_w_ij * log P(j|i)
```

---

## Loss Function Summary

```
L_total = L_recon
        + kl_weight           × L_KL
        + color_weight        × L_color_mse
        + semantic_weight     × L_scene_kl
        + layout_weight       × L_layout_mse
        + cross_recon_weight  × L_cross_recon
        + ortho_weight        × L_ortho
        + z_s_infonce_weight  × L_z_s_nce       [optional]
        + seg_loss_weight     × L_per_gaussian   [optional, ablation]
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
| `L_z_s_nce` | 0.0 | Scene-level semantic clustering in z_s (optional) |

---

## Ablation Study

### Current Architecture State

```
decoder_zs_cross_attn = True      ← MAIN NEW IDEA
color_residual = True
latent_disentangle = True,  semantic_dims = 512
decoder_fourier_pe = True
semantic_token_heads = True
cross_recon_weight = 0.3
ortho_weight = 0.1
kl_weight = 1e-5
```

### Completed Ablations

| Run | Config | Val L2 | Swap ratio | Key finding |
|-----|--------|--------|-----------|-------------|
| A | color_residual only | 1.43 | 4× | shape_embed starvation fixed |
| C | + scene_semantic_head alone | 1.80 | — | Alone hurts — needs color_residual foundation |
| H | + disentangle + layout | 1.565 | ~1× | Disentanglement beneficial |
| K | + position_layout_residual | ~1.0–1.2 | — | DC/AC position decomposition helps |
| P | + decoder_pos_enc (learnable) | 1.38 | — | PE gives tokens spatial identity |
| R | + token_cond approach A | **0.589** | — | Largest single jump — scaffold anchor Fourier bias |
| S | + token_cond approach B alone | unstable | — | KL explosion after epoch 1200 |
| T | + token_cond approach AB | best visual | — | Both geometric + semantic spatial identity |
| T2 | T + trilinear anchor smoothing | 0.606 | — | Eliminated voxel boundary seam artifacts |
| Old best | all + AdaLN + sem_tokens | ~0.79 | >400× | AdaLN causes geometry–semantic coupling |
| **New (ZS-CA)** | **z_g→decoder, z_s→cross-attn** | **TBD** | ~1× (expected) | Structural separation without swap penalty |

### Planned Ablations

| Run | `color_residual` | `disent + layout` | `fourier_pe` | `decoder_zs_cross_attn` | `sem_tok_heads` | `z_s_infonce` | Expected |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Baseline** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Upper bound reference |
| **+PE** | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | Fourier PE alone |
| **+Disent** | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | Disentanglement, legacy decoder |
| **+PE +Disent** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | Fourier PE + disentanglement |
| **+ZS-CA** | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | Key run — new decoder design |
| **+SemTok** | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | Inference-clean pipeline |
| **+NCE** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full system |

---

## Training Configuration

```yaml
# Architecture (shapevae-256.yaml)
num_latents:       256        # encoder queries → 512 tokens after +1 shape_embed
embed_dim:         32         # token dimension in z; z is always [B, 512, 32]
width:             384        # transformer hidden width
encoder_layers:    6
decoder_layers:    12
heads:             12
num_freqs:         8          # Fourier frequencies

# Training (gs_can3tok_2.py)
batch_size:        100        # per GPU; effective = 200 with 2× H100
learning_rate:     1e-4       # cosine decay to lr × 0.1
warmup_steps:      300
kl_weight:         1e-5
semantic_dims:     512        # z_s spans first 512 dims = 16 tokens × 32
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
├── semantic_losses.py                        # Per-Gaussian InfoNCE + scene z_s InfoNCE
├── gs_ply_reconstructor.py                   # Write decoder output to .ply (SuperSplat format)
├── pca_feature_visualization.py              # PCA coloring; z_s space PLY visualisation
├── model/
│   ├── configs/aligned_shape_latents/
│   │   └── shapevae-256.yaml                 # Architecture hyperparameters
│   └── michelangelo/models/tsal/
│       ├── sal_perceiver_dist_changes.py     # Full model:
│       │                                     #   CrossAttentionEncoder
│       │                                     #   AlignedShapeLatentPerceiver
│       │                                     #   ZSCondTransformerBlock/Decoder  ← NEW
│       │                                     #   MeanColorHead, SceneSemanticHead
│       │                                     #   SceneLayoutHead
│       │                                     #   SemanticTokenInfoNCEHead        ← NEW
│       │                                     #   FourierDecoderPE (496 or 512)
│       │                                     #   AdaLNBlock, AdaLNTransformerDecoder (legacy)
│       │                                     #   GS_decoder (configurable num_tokens)
│       └── asl_pl_module.py                  # PyTorch Lightning wrapper
└── job_scripts/
    ├── run_can3tok_scaffold.job              # SLURM job script
    └── accelerate_config.yaml               # Accelerate DDP config
```

---

## Key Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--decoder_zs_cross_attn` | **False** | **NEW MAIN IDEA.** z_g only in decoder sequence; z_s conditions via cross-attn K/V in every layer. Requires `latent_disentangle`. |
| `--color_residual` | False | DC/AC color decomposition; MeanColorHead on z token 0 |
| `--scene_semantic_head` | False | Scene label distribution prediction from tokens 1–15 |
| `--scene_layout_head` | False | Per-category centroid prediction from tokens 1–15 |
| `--latent_disentangle` | False | Split z into z_s (semantic, tokens 0–15) and z_g (geometric, tokens 16–511) |
| `--semantic_dims` | 512 | Dimensionality of z_s; z_s token count = semantic_dims // embed_dim = 16 |
| `--cross_recon_weight` | 0.3 | Cross-reconstruction loss weight (geometry sufficiency of z_g) |
| `--ortho_weight` | 0.1 | Orthogonality loss weight (linear independence of z_s and z_g) |
| `--decoder_fourier_pe` | False | 3D Fourier PE; with ZS-CA applies to 496 z_g voxels (16–511) |
| `--decoder_pos_enc` | False | Learnable PE; overridden by `decoder_fourier_pe` |
| `--semantic_token_heads` | False | Run heads on z tokens 0–15 instead of shape_embed (inference-clean) |
| `--z_s_infonce_weight` | 0.0 | Scene-level z_s InfoNCE weight; 0=disabled, 0.1=recommended start |
| `--z_s_infonce_temperature` | 0.07 | InfoNCE temperature |
| `--z_s_infonce_delta` | 0.4 | Min label_dist cosine similarity for positive pairs |
| `--token_cond` | False | Legacy: inject spatial-semantic bias into decoder tokens (auto-disabled with ZS-CA) |
| `--token_cond_approach` | B | A=scaffold anchors, B=category centroids, AB=both |
| `--token_cond_adaln` | False | Legacy: per-layer AdaLN-Zero conditioning (auto-disabled with ZS-CA) |
| `--semantic_mode` | none | Per-Gaussian InfoNCE mode: `hidden`, `geometric`, `dist` (ablation only) |
| `--kl_weight` | 1e-5 | KL regularisation weight |

---

## Quick Start

```bash
# New architecture (main new idea — z_s cross-attention conditioning)
# Set in job script:
DECODER_ZS_CROSS_ATTN=True
LATENT_DISENTANGLE=True
DECODER_FOURIER_PE=True
COLOR_RESIDUAL=True
SEMANTIC_TOKEN_HEADS=True

# Enable z_s InfoNCE after reconstruction stabilises (~200 epochs)
Z_S_INFONCE_WEIGHT=0.1

# Baseline (no improvements)
DECODER_ZS_CROSS_ATTN=False; COLOR_RESIDUAL=False; LATENT_DISENTANGLE=False
DECODER_FOURIER_PE=False; SEMANTIC_TOKEN_HEADS=False

# Resume from old checkpoint into new design (strict=False, new components init fresh)
RESUME_CHECKPOINT="/path/to/epoch_800.pth"
DECODER_ZS_CROSS_ATTN=True
```

---

## Diagnostic Output Format

Every training epoch prints all active loss components:

```
Epoch NNNN | Loss=X | Recon=X | KL=X | ColorPred=X | SceneSem=X | Layout=X |
            CrossRecon=X | Ortho=X | Anchor=X | SegPred=X | ScalePen=X |
            Z_sNCE=X | Z_sNPos=X | PgNCE=X | LR=X
  Pos=X | Col=X | Opa=X | Scl=X | Rot=X
```

**Z_sNCE** — scene-level z_s InfoNCE loss (new design).
**Z_sNPos** — average number of positive pairs per anchor per batch. If this is 0, reduce `Z_S_INFONCE_DELTA`.
**PgNCE** — per-Gaussian InfoNCE loss (legacy, ablation only).