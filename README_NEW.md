# Can3Tok VAE — Semantic-Aware 3D Gaussian Scene Tokenizer

A Perceiver-based Variational Autoencoder that encodes full indoor 3DGS scenes
into a structured, disentangled latent space with explicit global layout and local
geometry subspaces. Designed as Stage 1 of a two-stage hierarchical scene generation
pipeline.

Built on [SceneSplat-7K](https://arxiv.org/abs/2501.01895) (ICCV 2025 Oral).

---

## Visualisation

| Input Scene | Reconstructed (Epoch 1950) |
|---|---|
| `input_scene_000_0201_840151_0.ply` | `scene_000_epoch_1950_786.ply` |

Input and reconstruction are visually nearly identical at epoch 1950, confirming
the VAE has learned a high-quality latent representation of indoor 3DGS scenes.

---

## Table of Contents

1. [Full Architecture Diagram](#1-full-architecture-diagram)
2. [Decoder Strategies](#2-decoder-strategies)
3. [Structured Layout Token Supervision](#3-structured-layout-token-supervision)
4. [Semantic Supervision — InfoNCE Variants](#4-semantic-supervision--infonce-variants)
5. [PCA Visualisations](#5-pca-visualisations)
6. [Loss Function Summary](#6-loss-function-summary)
7. [Second-Stage Generation Pipeline](#7-second-stage-generation-pipeline)
8. [Training Configuration](#8-training-configuration)
9. [Dataset](#9-dataset)
10. [Ablation Study](#10-ablation-study)
11. [Key Flags Reference](#11-key-flags-reference)
12. [Experiment Grid](#12-experiment-grid)
13. [Diagnostic Output](#13-diagnostic-output)
14. [Code Structure](#14-code-structure)

---

## 1. Full Architecture Diagram

```
══════════════════════════════════════════════════════════════════════════════
                        CAN3TOK VAE — FULL PIPELINE
══════════════════════════════════════════════════════════════════════════════

INPUT
┌──────────────────────────────────────────────────────────────┐
│  40,000 Gaussians × 18 features          [B, 40000, 18]      │
│  cols 0:3   voxel_center  (coarse 40³ grid position)         │
│  col  3     voxel_id      (encoder voxel index)              │
│  cols 4:7   xyz           (absolute Gaussian position)       │
│  cols 7:10  rgb           (color residual if color_residual) │
│  col  10    opacity                                          │
│  cols 11:14 scale                                            │
│  cols 14:18 quaternion                                       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
══════════════════════════════════════════════════════════════════
  ENCODER — CrossAttentionEncoder
══════════════════════════════════════════════════════════════════

  voxel_centers         xyz_actual          gaussian_params
  [B,40000,3]           [B,40000,3]         (opac,scale,quat)
       │                     │              [B,40000,8]
       ▼                     ▼
  FourierEmbedder       FourierEmbedder
  (8 freqs, coarse)     (8 freqs, fine)
  [B,40000,51]          [B,40000,51]
       │                     │
       └──────── concat ─────┘  [51|51|8] = 110 dims
                      │
                 Linear(110→384)             [B, 40000, 384]
                      │
       512 learned queries ──────────────┐
                      │                  ▼
              Cross-Attention         [B, 512, 384]
                      │
              6× Self-Attention       [B, 512, 384]
                      │
          ┌───────────┴──────────────┐
          ▼                          ▼
    shape_embed [B,384]         geom_tokens [B,511,384]
    (token 0, global)           (tokens 1-511, per-region)

══════════════════════════════════════════════════════════════════
  VAE BOTTLENECK — Disentangled Latent Space
══════════════════════════════════════════════════════════════════

  shape_embed [B,384]           geom_tokens [B,511,384]
       │                               │
  mu_s_proj_mean/var             pre_kl + flatten
  Linear(384→512)                Linear(16352→15872)
       ▼                               ▼
  mu_s [B,512]  ──── concat ────  mu_g [B,15872]
                        ▼
                  mu [B,16384],  lv [B,16384]
                        │  reparameterisation
                        ▼
                  Z [B,512,32]
                        │
          ┌─────────────┴──────────────────┐
          ▼                                ▼
  z_s = Z[:,0:16,:]  [B,16,32]    z_g = Z[:,16:,:]  [B,496,32]
  16 semantic/layout tokens        496 geometry tokens
  KL-regularised → N(0,I)         KL-regularised → N(0,I)

══════════════════════════════════════════════════════════════════
  SEMANTIC TOKEN HEADS
  semantic_token_heads=True → inference-clean (no encoder needed)
══════════════════════════════════════════════════════════════════

  WITHOUT structured_layout_tokens:
    Token 0     [B,32]  → MeanColorHead   → mean_color [B,3]
    Tokens 1-15 [B,480] → SemanticHead    → label_dist [B,72]   } interference
    Tokens 1-15 [B,480] → LayoutHead      → centroids  [B,72,3] } same floats

  WITH structured_layout_tokens=True:
    Token 0     [B,32]  → MeanColorHead   → mean_color [B,3]    (exclusive)
    Tokens 1-8  [B,256] → SemanticHead    → label_dist [B,72]   (exclusive)
    Tokens 9-15 [B,224] → LayoutHead      → centroids  [B,72,3] (exclusive)
```

---

## 2. Decoder Strategies

```
══════════════════════════════════════════════════════════════════
  STRATEGY A — 16+496 in Sequence  [BEST PERFORMANCE]
  latent_disentangle=True
══════════════════════════════════════════════════════════════════

  Z [B,512,32] = [z_layout(0:16) | z_geo(16:512)]
       │
  post_kl Linear(32→384) → [B,512,384]
  + FourierPE
       │
  12× Self-Attention (layout and geometry mix freely)
       │
  flatten → GS_decoder 8-layer MLP → [B,40000,14]

  WHY BEST: richest mixing — z_geo tokens can attend to z_layout tokens
  at every transformer layer. No bottleneck between global layout and
  local geometry.
  SECOND STAGE: both z_layout and z_geo KL-regularised → DiT-generatable.


══════════════════════════════════════════════════════════════════
  STRATEGY B1 — 512 Geometry + Cross-Attention Conditioning
  decoder_layout_cross_attn=True
══════════════════════════════════════════════════════════════════

  Z [B,512,32] = full geometry
  z_layout [B,16,32] from Layout16Projector(shape_embed) ← SEPARATE

  z_geo → post_kl → H_g [B,512,384]     z_layout → post_kl_layout → H_lay [B,16,384]
                                                                      (K and V only)
  ┌─────────────────────────────────────────────┐
  │  12× ZSCondTransformerDecoder               │
  │    Self-Attention(H_g, H_g, H_g)            │
  │    Cross-Attention(Q=H_g, K=H_lay, V=H_lay) │◄── H_lay per layer
  │    FFN                                      │
  └─────────────────────────────────────────────┘
  → GS_decoder → [B,40000,14]


══════════════════════════════════════════════════════════════════
  STRATEGY B2 — 512 Geometry + Additive Bias Conditioning
  decoder_layout_additive=True
══════════════════════════════════════════════════════════════════

  Same z_layout as B1. LayoutAdditiveConditioner:
  flatten(z_layout)→MLP→[B,384] broadcast bias added once before stack.


══════════════════════════════════════════════════════════════════
  STRATEGY C — Baseline (original Can3Tok)
══════════════════════════════════════════════════════════════════

  512 geometry tokens, standard self-attention, no layout conditioning.
  Limitation: captures local geometry, not global layout structure.
```

---

## 3. Structured Layout Token Supervision

```
WITHOUT (default):
  Tokens 1-15 [B,480] → SceneSemanticHead  \  same 480 floats
  Tokens 1-15 [B,480] → SceneLayoutHead    /  gradient interference

WITH structured_layout_tokens=True:
  Tokens 1-8  [B,256] → SceneSemanticHead  ← exclusive (8×32)
  Tokens 9-15 [B,224] → SceneLayoutHead    ← exclusive (7×32)
  Token 0     [B,32]  → MeanColorHead      ← always exclusive
```

Each head's gradient reaches **only** its own token range. Implemented
correctly for both Strategy A (dedicated `scene_*_module` heads with
dims 256/224) and Strategy B (dedicated `lay_*_head` heads with same dims).

---

## 4. Semantic Supervision — InfoNCE Variants

All five types can coexist. Select which to enable per experiment.

```
TYPE 1 — Soft-Pair Scene InfoNCE  (--z_s_infonce_weight)
  flatten(z_s [B,512]) → SemanticTokenInfoNCEHead → z_s_proj [B,128]
  Positive pairs: cos_sim(label_dist_i, label_dist_j) > delta (soft weights)
  SupCon formulation. One point per scene.
  Vis: z_s_space_epoch_NNN.ply

TYPE 2 — Per-Token Prototype InfoNCE  (--zs_token_infonce_weight)
  Raw tokens [B,16,32] → B×16 points labelled by argmax(label_dist[b])
  Cross-batch prototype NCE — same mechanism as per-Gaussian.
  Vis: zs_tokens_epoch_NNN.ply

TYPE 3 — Flatten Prototype InfoNCE  (--zs_layout_infonce_weight)
  Strategy A: routes last_z_s_infonce_proj → last_z_layout_proj
  Strategy B: flatten(z_layout)→SemanticTokenInfoNCEHead→[B,128]
  Hard dominant-category prototypes. One point per scene.
  Vis: zs_layout_epoch_NNN.ply

TYPE 4 — Pool InfoNCE, mirrors decoder hidden  (--zs_pool_infonce_weight)
  tokens [B,16,32] → mean_pool → [B,32]
    → Linear(32→1024) → [B,1024]   ← SAME dim as decoder hidden state
    → Lin(1024→512)+LN+ReLU
    → Lin(512→256)+LN+ReLU
    → Lin(256→16×32) → [B,16,32] L2-norm
    → compute_semantic_loss(...)    ← EXACT same call as decoder NCE
  Labels: argmax(label_dist) broadcast to all 16 positions.
  Vis: zs_pool_epoch_NNN.ply  (PCA of [B,1024] hidden states)

TYPE 5 — Per-Gaussian Decoder InfoNCE  (--semantic_mode hidden)
  decoder hidden [B,1024] → SemanticProjectionHead → [B,40000,32]
  → compute_semantic_loss(actual per-Gaussian ScanNet72 labels,
                          subsample=10000, strategy=balanced)
  Vis: scene{i}_semantic_infonce.ply

────────────────────────────────────────────────────────
  TYPE 4 MIRRORS TYPE 5 STRUCTURALLY:

  Type 5: hidden [B,1024]    →MLP→ [B,40000,32] → NCE (actual labels)
  Type 4: pool   [B,1024]    →MLP→ [B,16,32]    → NCE (dominant-cat labels)
                 ↑ same bottleneck dim, same MLP structure, same NCE call
────────────────────────────────────────────────────────
```

---

## 5. PCA Visualisations

Written every `pca_vis_freq` epochs. Open in SuperSplat (splat scale ~0.05m).

| PLY file | Trigger | Points | Colors |
|---|---|---|---|
| `scene{i}_input.ply` | always | 40k | Gaussian color |
| `scene{i}_recon.ply` | always | 40k | Gaussian color |
| `scene{i}_semantic_infonce.ply` | `semantic_mode=hidden` | 40k | ScanNet72 category |
| `z_s_space_epoch_NNN.ply` | `z_s_infonce_weight>0` | B | Dominant category |
| `zs_tokens_epoch_NNN.ply` | `zs_token_infonce_weight>0` | B×16 | Dominant category |
| `zs_layout_epoch_NNN.ply` | `zs_layout_infonce_weight>0` | B (A) / B×16 (B) | Dominant category |
| `zs_pool_epoch_NNN.ply` | `zs_pool_infonce_weight>0` | B | Dominant category |

All category colors use the same ScanNet72 palette — PLYs are directly
comparable side by side to assess relative clustering quality.

---

## 6. Loss Function Summary

```
L_total = L_recon
        + kl_weight               × L_KL
        + mean_color_weight       × L_color_mse
        + scene_semantic_weight   × L_scene_kl
        + layout_loss_weight      × L_layout_mse
        + cross_recon_weight      × L_cross_recon   [if latent_disentangle]
        + ortho_weight            × L_ortho         [if latent_disentangle]
        + z_s_infonce_weight      × L_z_s_nce       [Type 1, optional]
        + zs_token_infonce_weight × L_zs_tok_nce    [Type 2, optional]
        + zs_layout_infonce_weight× L_zs_layout_nce [Type 3, optional]
        + zs_pool_infonce_weight  × L_zs_pool_nce   [Type 4, optional]
        + segment_loss_weight     × L_per_gaussian   [Type 5, optional]
```

---

## 7. Second-Stage Generation Pipeline

Requires Strategy A (both z_layout and z_geo are KL-regularised).

```
══════════════════════════════════════════════════════════════════
  SCENE GENERATION (text → full scene)
══════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────┐
  │  STAGE 1 — Layout DiT  (~6 transformer layers)           │
  │                                                          │
  │  noise [B,16,32] ~ N(0,I)                               │
  │    + CLIP(text) or scene-class token                     │
  │  Flow matching: learn P(z_layout | text/class)           │
  │  Train target: Z[:,0:16,:] from VAE                      │
  │                                                          │
  │  Output: z_layout [B,16,32]                              │
  │    encodes: scene type, dominant categories,             │
  │             color palette, spatial centroids             │
  └──────────────────────┬───────────────────────────────────┘
                         │ z_layout
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  STAGE 2 — Geometry DiT  (~16-28 transformer layers)     │
  │                                                          │
  │  concat(z_layout, noisy_z_geo) [B,512,32]                │
  │    ↑ identical structure to VAE decoder input            │
  │  Flow matching: learn P(z_geo | z_layout)                │
  │  Train target: Z[:,16:,:] from VAE                       │
  │                                                          │
  │  Architecture: same self-attention as VAE decoder        │
  │  Init: can load VAE decoder transformer weights          │
  │                                                          │
  │  Output: z_geo [B,496,32]                                │
  └──────────────────────┬───────────────────────────────────┘
                         │
  ┌──────────────────────────────────────────────────────────┐
  │  DECODE (VAE decoder, frozen)                            │
  │                                                          │
  │  Z = concat(z_layout, z_geo) [B,512,32]                  │
  │    → same VAE decoder as training                        │
  │    → mean_color from MeanColorHead(Z[:,0,:])             │
  │    → final_color = residuals + mean_color                │
  │    → PLY (40k Gaussians)                                 │
  │                                                          │
  │  No encoder. No shape_embed. No GT annotations.          │
  └──────────────────────────────────────────────────────────┘


══════════════════════════════════════════════════════════════════
  SCENE COMPLETION (partial scan → complete scene)
══════════════════════════════════════════════════════════════════

  1. encoder(partial Gaussians) → z_layout (stable at 30%+ coverage)
                                + z_geo_partial

  2. Construct z_geo_noisy:
       Observed voxels   → z_geo from encoder (fixed)
       Unobserved voxels → Gaussian noise

  3. Stage 2 DiT inpainting:
       [z_layout | z_geo_noisy] → denoise unobserved z_geo tokens
       Observed tokens fixed throughout denoising.
       → z_geo_complete

  4. Z = concat(z_layout, z_geo_complete) → VAE decoder → full scene

  WHY z_layout IS STABLE FROM PARTIAL OBSERVATIONS:
    shape_embed cross-attends globally over all observed Gaussians.
    Even 30% scene coverage preserves dominant categories, approximate
    centroids, and color palette — enough to condition geometry generation.


══════════════════════════════════════════════════════════════════
  COMPARISON WITH SEEN2SCENE (Meng et al., arXiv 2026)
══════════════════════════════════════════════════════════════════

  Seen2Scene: TSDF voxel grids + GT 3D bounding box layout annotations
              + ControlNet fine-tuning for completion.

  Can3Tok:    3DGS Gaussians + per-Gaussian ScanNet72 labels only.
              z_layout learned end-to-end — no box annotations.
              Completion via DiT inpainting — no separate fine-tuning.
              Richer layout: 16 continuous tokens vs sparse box list.
```

---

## 8. Training Configuration

```yaml
# Architecture (shapevae-256.yaml)
num_latents:       256    # encoder queries; 512 tokens total
embed_dim:         32     # token dim in z; z always [B,512,32]
width:             384    # transformer hidden width
encoder_layers:    6
decoder_layers:    12
heads:             12
num_freqs:         8      # Fourier frequencies

# Training (gs_can3tok_2.py)
batch_size:        64     # per GPU
learning_rate:     1e-4   # cosine decay to lr × lr_min_ratio
lr_min_ratio:      0.1
warmup_steps:      300
kl_weight:         1e-5
semantic_dims:     512    # z_s spans first 512 dims = 16 tokens × 32
cross_recon_weight: 0.3
ortho_weight:       0.1
layout_loss_weight: 0.3
scene_semantic_weight: 0.3
```

---

## 9. Dataset

**SceneSplat-7K** — 7,916 indoor 3DGS scenes from ScanNet, ScanNet++, Replica,
Hypersim, 3RScan, ARKitScenes, Matterport3D. Per-Gaussian ScanNet72 labels.

Preprocessing:
- Positions normalised to 10m radius canonical sphere (linear scale)
- Top-40k Gaussians sampled by opacity (deterministic per epoch)
- Color: per-scene mean subtracted → residuals ∈ [−0.5, +0.5] if `color_residual=True`
- Layout: per-category centroid computed per scene for SceneLayoutHead

---

## 10. Ablation Study

### Completed Runs

| Run | Config | Val L2 | Key finding |
|---|---|---|---|
| A | color_residual only | 1.43 | shape_embed starvation fixed |
| H | + disentangle + layout | 1.565 | Disentanglement beneficial |
| P | + decoder_pos_enc | 1.38 | Spatial token identity |
| R | + token_cond approach A | **0.589** | Largest single gain |
| Old best | all + AdaLN | ~0.79 | AdaLN caused geometry-semantic coupling |

### Decoder Strategy × Structured Split Grid (7 runs)

| Exp | Strategy | Structured | Val L2 |
|---|---|:---:|---|
| 1 | A | ✗ | TBD |
| 2 | A | ✓ | TBD |
| 3 | B1 | ✗ | TBD |
| 4 | B1 | ✓ | TBD |
| 5 | B2 | ✗ | TBD |
| 6 | B2 | ✓ | TBD |
| 7 | C baseline | ✗ | TBD |

### InfoNCE Ablations (on best strategy)

| Run | Type 1 | Type 2 | Type 3 | Type 4 | Type 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| Ref | ✗ | ✗ | ✗ | ✗ | ✗ |
| Pool | ✗ | ✗ | ✗ | 0.1 | ✗ |
| Pool+Dec | ✗ | ✗ | ✗ | 0.1 | 0.1 |
| Soft | 0.1 | ✗ | ✗ | ✗ | ✗ |
| Tok | ✗ | 0.1 | ✗ | ✗ | ✗ |
| All3 | 0.1 | 0.1 | 0.1 | ✗ | ✗ |
| Full | 0.1 | ✗ | 0.1 | 0.1 | 0.1 |

---

## 11. Key Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--latent_disentangle` | False | Split Z into z_s (tokens 0-15) and z_g (tokens 16-511) |
| `--semantic_dims` | 512 | z_s dims; token count = 512/32 = 16 |
| `--decoder_layout_cross_attn` | False | Strategy B1: 512 geom + z_layout as cross-attn K/V |
| `--decoder_layout_additive` | False | Strategy B2: 512 geom + z_layout as additive bias |
| `--structured_layout_tokens` | False | Exclusive split: tokens 1-8 → semantic, 9-15 → layout |
| `--color_residual` | False | DC/AC color; MeanColorHead on z_s token 0 |
| `--semantic_token_heads` | False | Heads on z tokens (inference-clean; requires latent_disentangle) |
| `--scene_semantic_head` | False | KL loss: label distribution prediction |
| `--scene_layout_head` | False | MSE loss: category centroid prediction |
| `--cross_recon_weight` | 0.3 | Cross-recon: geometry sufficiency of z_g |
| `--ortho_weight` | 0.1 | Orthogonality: z_s ⊥ z_g |
| `--decoder_fourier_pe` | False | 3D Fourier PE from voxel grid (recommended) |
| `--z_s_infonce_weight` | 0.0 | Type 1: soft-pair scene InfoNCE |
| `--z_s_infonce_delta` | 0.4 | Min label_dist cosine sim for positives |
| `--zs_token_infonce_weight` | 0.0 | Type 2: per-token prototype NCE |
| `--zs_layout_infonce_weight` | 0.0 | Type 3: flatten prototype NCE |
| `--zs_pool_infonce_weight` | 0.0 | Type 4: pool→1024→NCE (decoder mirror) |
| `--semantic_mode` | none | Type 5: `hidden`/`geometric`/`dist` |
| `--segment_loss_weight` | 0.0 | Type 5 weight |
| `--semantic_subsample` | 2000 | Type 5 subsample count |
| `--sampling_strategy` | balanced | Type 5 subsample: `balanced` or `random` |
| `--pca_vis_freq` | 500 | Write PCA PLYs every N epochs |

---

## 12. Experiment Grid

```bash
# Strategy A, no structured
LATENT_DISENTANGLE=True; SEMANTIC_TOKEN_HEADS=True
SCENE_SEMANTIC_HEAD=True; SCENE_LAYOUT_HEAD=True; COLOR_RESIDUAL=True
STRUCTURED_LAYOUT_TOKENS=False; DECODER_LAYOUT_CROSS_ATTN=False
DECODER_LAYOUT_ADDITIVE=False; CROSS_RECON_WEIGHT=0.3; ORTHO_WEIGHT=0.1

# Strategy A, structured  (only STRUCTURED_LAYOUT_TOKENS changes)
STRUCTURED_LAYOUT_TOKENS=True

# Strategy B1, no structured
LATENT_DISENTANGLE=False; SEMANTIC_TOKEN_HEADS=False
SCENE_SEMANTIC_HEAD=True; SCENE_LAYOUT_HEAD=True; COLOR_RESIDUAL=True
STRUCTURED_LAYOUT_TOKENS=False; DECODER_LAYOUT_CROSS_ATTN=True
DECODER_LAYOUT_ADDITIVE=False; CROSS_RECON_WEIGHT=0.0; ORTHO_WEIGHT=0.0

# Strategy B1, structured  (only STRUCTURED_LAYOUT_TOKENS changes)
STRUCTURED_LAYOUT_TOKENS=True

# Strategy B2, no structured
DECODER_LAYOUT_CROSS_ATTN=False; DECODER_LAYOUT_ADDITIVE=True

# Strategy B2, structured
STRUCTURED_LAYOUT_TOKENS=True

# Strategy C baseline
LATENT_DISENTANGLE=False; SEMANTIC_TOKEN_HEADS=False
SCENE_SEMANTIC_HEAD=False; SCENE_LAYOUT_HEAD=False; COLOR_RESIDUAL=True
STRUCTURED_LAYOUT_TOKENS=False; DECODER_LAYOUT_CROSS_ATTN=False
DECODER_LAYOUT_ADDITIVE=False; CROSS_RECON_WEIGHT=0.0; ORTHO_WEIGHT=0.0
```

---

## 13. Diagnostic Output

```
Epoch NNNN | Loss=X | Recon=X | KL=X | ColorPred=X | SceneSem=X |
            Layout=X | CrossRecon=X | Ortho=X | Anchor=X |
            Z_sNCE=X | Z_sNPos=X |       ← Type 1
            ZsTokNCE=X | ZsTokNCats=X |  ← Type 2
            LayNCE=X | LayNCats=X |       ← Type 3
            PoolNCE=X | PoolNCats=X |     ← Type 4
            PgNCE=X |                     ← Type 5
            LR=X
  Pos=X | Col=X | Opa=X | Scl=X | Rot=X
```

Key diagnostics:
- `*NCats` — distinct dominant categories in batch. Need ≥2 for NCE. If =1: increase batch size.
- `Z_sNPos` — avg positive pairs per anchor (Type 1). If 0: reduce `z_s_infonce_delta`.
- `CrossRecon` — should converge toward same order as `Recon` by epoch 500.
- `Ortho` — decreases over training as z_s and z_g specialise.

---

## 14. Code Structure

```
.
├── gs_can3tok_2.py                     # Training loop, all losses, eval,
│                                       #   PCA collection, PLY writing
├── gs_dataset_scenesplat.py            # Dataset, preprocessing, scaffold
├── semantic_losses.py                  # compute_semantic_loss         (Type 5)
│                                       # compute_scene_infonce_loss    (Type 1)
│                                       # compute_zs_token_infonce_loss (Type 2)
│                                       # compute_zs_layout_infonce_loss(Type 3)
├── pca_feature_visualization.py        # visualize_semantic_features   (Type 5)
│                                       # visualize_z_s_space           (Types 1,3,4)
│                                       # visualize_zs_tokens           (Types 2,4)
├── gs_ply_reconstructor.py             # Write PLY in SuperSplat format
├── model/configs/aligned_shape_latents/
│   └── shapevae-256.yaml               # Architecture hyperparameters
└── model/michelangelo/models/tsal/
    └── sal_perceiver_dist_changes.py   # Full model
        │                               # CrossAttentionEncoder
        │                               # AlignedShapeLatentPerceiver
        │                               #   Strategy A/B1/B2/C decoder
        │                               #   Layout16Projector          (B)
        │                               #   LayoutAdditiveConditioner  (B2)
        │                               #   ZSTokenPoolProjectHead     (Type 4)
        │                               #   MeanColorHead
        │                               #   SceneSemanticHead (256/480)
        │                               #   SceneLayoutHead   (224/480)
        │                               #   SemanticTokenInfoNCEHead   (Types 1,3)
        │                               #   ZSCondTransformerDecoder   (B1)
        │                               #   FourierDecoderPE
        │                               #   GS_decoder
        └── job_scripts/
            ├── run_can3tok_scaffold.job       # SLURM — all flags documented
            └── accelerate_config.yaml         # Accelerate DDP config
```