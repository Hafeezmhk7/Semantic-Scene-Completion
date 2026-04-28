# Can3Tok VAE — Semantic-Aware 3D Gaussian Scene Tokenizer

A Perceiver-based Variational Autoencoder encoding full indoor 3DGS scenes into a
structured, disentangled latent space. Designed as Stage 1 of a two-stage hierarchical
scene generation pipeline.

Built on [SceneSplat-7K](https://arxiv.org/abs/2501.01895).

---

## Table of Contents

1. [Architecture at a Glance](#1-architecture-at-a-glance)
2. [Shared Encoder Pipeline](#2-shared-encoder-pipeline)
3. [Strategy A — Disentangled VAE (without Structured Split)](#3-strategy-a--disentangled-vae-without-structured-split)
4. [Strategy A — Disentangled VAE (with Structured Split)](#4-strategy-a--disentangled-vae-with-structured-split)
5. [Strategy B1 — Cross-Attention Conditioning](#5-strategy-b1--cross-attention-conditioning)
6. [Strategy D — Hybrid (A Bottleneck + B1 Decoder)](#6-strategy-d--hybrid-a-bottleneck--b1-decoder)
7. [Strategy C — Baseline](#7-strategy-c--baseline)
8. [InfoNCE Loss Variants](#8-infonce-loss-variants)
9. [Training Configuration](#9-training-configuration)
10. [Loss Function Summary](#10-loss-function-summary)
11. [Second-Stage Generation Pipeline](#11-second-stage-generation-pipeline)
12. [Dataset](#12-dataset)
13. [Ablation Results](#13-ablation-results)
14. [Key Flags Reference](#14-key-flags-reference)
15. [Code Structure](#15-code-structure)

---

## 1. Architecture at a Glance

| Strategy | Z content | Decoder conditioning | Disentangle losses | Partial obs @40% |
|---|---|---|---|:---:|
| **C** — Baseline | 512 geometry tokens | None | None | N/A |
| **A** — Disentangled | 16 layout + 496 geometry | Self-attention (all 512 together) | L_cross + L_ortho | 0.20–0.37 |
| **B1** — Cross-Attn | 512 geometry tokens | Cross-attention K,V per layer | None (structural) | **0.761** |
| **B2** — Additive | 512 geometry tokens | Broadcast additive bias | None | −0.06 (failed) |
| **D** — Hybrid | 16 layout + 496 geometry | Cross-attention K,V per layer | L_cross + L_ortho | ~0.37–0.76 (expected) |

All strategies share the same encoder. The decoder strategy is selected by flags.

> **Original Can3Tok vs this work.** The published Can3Tok (ICCV 2025) uses a single-stream
> Strategy C baseline: 256 encoder geom queries → projected to 512 decoder tokens, standard
> self-attention, no z_s / z_g split. shape_embed feeds only auxiliary heads — it is **not**
> part of the generative latent Z. Our contribution is (1) routing shape_embed into a dedicated
> z_s subspace of Z via `mu_s_proj_mean` (Strategy A), (2) enforcing disentanglement with
> L_cross + L_ortho + InfoNCE, and (3) the cross-attention conditioning decoder (Strategy B1)
> which extracts z_layout deterministically from shape_embed without entering Z.

---

## 2. Shared Encoder Pipeline

All three strategies use this encoder identically.

```mermaid
flowchart TD
    IN["<b>Raw 3DGS Scene</b><br/>N Gaussians (all attributes)"]

    SAMPLE["<b>Top-40k Sampling</b><br/>sort by opacity ↓<br/>pad with repeats if N &lt; 40,000"]

    NORM["<b>normalize_to_canonical_sphere</b><br/>center = coord.mean(axis=0)<br/>coord = (coord − center) × scale_factor<br/>scale_factor = 10.0 / (max_dist × 1.1)<br/>scale = scale × scale_factor"]

    FEAT["<b>Encoder Input Features</b><br/>[B, 40000, 18]<br/>─────────────────────────────────────<br/>cols 0:3  voxel_centers  ← FNV hash on 40³ grid<br/>col  3    voxel_id       ← FNV bucket index<br/>cols 4:7  xyz            ← normalised absolute position<br/>cols 7:10 rgb            ← ÷255 (or residual if color_residual)<br/>col  10   opacity<br/>cols 11:14 scale         ← × scale_factor<br/>cols 14:18 quaternion"]

    FE1["<b>FourierEmbedder (coarse)</b><br/>input: voxel_centers [B, 40000, 3]<br/>8 freqs + π  →  [B, 40000, 51]"]

    FE2["<b>FourierEmbedder (fine)</b><br/>input: xyz [B, 40000, 3]<br/>8 freqs + π  →  [B, 40000, 51]"]

    GP["<b>Gaussian Params</b><br/>[B, 40000, 8]<br/>opacity · scale×3 · quat×4"]

    CAT["<b>Concat</b><br/>[51 | 51 | 8] = 110 dims<br/>[B, 40000, 110]"]

    PROJ["<b>input_proj  Linear(110 → 384)</b><br/>[B, 40000, 384]"]

    QRY["<b>256 + 1 = 257 learned queries</b><br/>[257, 384]  ← random init, trained<br/>1 global query (→ shape_embed)<br/>256 geometry queries (→ geom_tokens)"]

    CA["<b>Cross-Attention</b><br/>queries [257, 384]  attend to  data [40000, 384]<br/>→  [B, 257, 384]"]

    SA["<b>6 × Self-Attention layers</b><br/>on latent queries only<br/>[B, 257, 384]  →  [B, 257, 384]"]

    SPLIT["<b>Split token 0 vs tokens 1:256</b>"]

    SE["<b>shape_embed</b><br/>[B, 384]<br/>query 0 — global scene summary<br/>─────────────────────────────────<br/>Strategy A: routed into z_s via mu_s_proj<br/>Strategy B1: routed into z_layout via Layout16Projector<br/>Strategy C: feeds auxiliary heads ONLY — NOT in Z"]

    GT["<b>geom_tokens</b><br/>[B, 256, 384]<br/>queries 1–256 — per-region geometry<br/>─────────────────────────────────<br/>All strategies: goes into VAE bottleneck<br/>256 encoder queries → upsampled to 496 or 512 latent token positions"]

    IN --> SAMPLE --> NORM --> FEAT
    FEAT --> FE1
    FEAT --> FE2
    FEAT --> GP
    FE1 --> CAT
    FE2 --> CAT
    GP  --> CAT
    CAT --> PROJ --> CA
    QRY --> CA
    CA  --> SA  --> SPLIT
    SPLIT --> SE
    SPLIT --> GT
```

---

## 3. Strategy A — Disentangled VAE (without Structured Split)

**Flags:** `--latent_disentangle --semantic_token_heads`

The VAE bottleneck splits into **z_s** (16 semantic tokens, 512 dims) and **z_g**
(496 geometry tokens, 15,872 dims). Both are KL-regularised. All 512 tokens are
concatenated and passed through the decoder together via self-attention.

```mermaid
flowchart TD
    SE["shape_embed  [B, 384]"]
    GT["geom_tokens  [B, 256, 384]"]

    subgraph BOTTLENECK["VAE BOTTLENECK — Disentangled Split"]
        direction TB

        MUS_PROJ["<b>mu_s_proj_mean / var</b><br/>Linear(384 → 512)<br/>─────────────────<br/>μ_s  [B, 512]<br/>σ²_s [B, 512]"]

        PREKL["<b>pre_kl  Linear(384 → 64)</b><br/>applied to geom_tokens [B, 256, 384]<br/>→ moments [B, 256, 64]<br/>→ sample → kl_embed [B, 256, 32]<br/>→ flatten → [B, 8192]"]

        MUG_PROJ["<b>kl_emb_proj_mean_g / var_g</b><br/>Linear(8192 → 15872)<br/>input: 256 geom tokens × 32 dims = 8192<br/>output: geom_dims = 64×64×4 − 512 = 15872<br/>─────────────────────────────────<br/>μ_g  [B, 15872]<br/>σ²_g [B, 15872]<br/>↑ 256 encoder queries upsampled to 496 latent positions"]

        CAT_MU["<b>Concat</b><br/>μ = [μ_s | μ_g]      [B, 16384]<br/>σ² = [σ²_s | σ²_g]  [B, 16384]"]

        REPARAM["<b>Reparameterisation</b><br/>z = μ + σ · ε,   ε ~ N(0, I)<br/>z  [B, 16384]"]

        RESHAPE["<b>Reshape</b><br/>Z  [B, 512, 32]"]

        ZS["<b>z_s = Z[:, 0:16, :]</b><br/>[B, 16, 32]<br/>16 semantic tokens<br/>semantic_dims=512 → 512÷32=16 tokens<br/>KL → N(0, I)"]

        ZG["<b>z_g = Z[:, 16:, :]</b><br/>[B, 496, 32]<br/>496 geometry tokens<br/>KL → N(0, I)"]
    end

    subgraph HEADS_A["AUXILIARY HEADS on z_s  (semantic_token_heads=True, NO structured split)"]
        direction TB
        H0["<b>MeanColorHead</b>  ← token 0 only<br/>z[:, 0:32]  [B, 32]<br/>Linear(32→64)→ReLU→Linear(64→3)→Sigmoid<br/>→  mean_color  [B, 3]<br/><i>L_color = MSE(pred, GT mean RGB)</i>"]

        H1["<b>SceneSemanticHead</b>  ← tokens 1–15 shared<br/>z[:, 32:512]  [B, 480]<br/>Linear(480→128)→LN→ReLU→Linear(128→72)→Softmax<br/>→  label_dist_pred  [B, 72]<br/><i>L_sem = KL(pred ∥ GT dist)</i>"]

        H2["<b>SceneLayoutHead</b>  ← tokens 1–15 shared<br/>z[:, 32:512]  [B, 480]  ← SAME 480 floats as H1<br/>Linear(480→512)→LN→ReLU→Linear(512→256)→LN→ReLU→Linear(256→216)<br/>→  category_centroids  [B, 72, 3]<br/><i>L_layout = masked MSE(pred, GT centroids)</i>"]

        WARN["⚠️ H1 and H2 receive identical 480 floats<br/>→ gradient interference between semantic and layout<br/>→ use --structured_layout_tokens to eliminate this"]
    end

    subgraph DISENT["DISENTANGLEMENT LOSSES"]
        direction TB
        CROSS["<b>Cross-Reconstruction Loss</b><br/>z_cross = [z_s_B | z_g_A]  (roll z_s by 1 within batch)<br/>UV_cross = decode(z_cross)<br/><i>L_cross = L2(UV_cross positions, GT_A positions)</i><br/>Forces z_g to carry geometry without relying on z_s"]
        ORTH["<b>Orthogonality Loss</b><br/>p_s = random_proj(μ_s, 64 dims)<br/>p_g = random_proj(μ_g, 64 dims)<br/><i>L_ortho = mean((p_s · p_g)²)</i><br/>Penalises alignment between μ_s and μ_g directions"]
    end

    subgraph INFONCE_A["INFONCE HEADS on z_s  (optional — select one or combine)"]
        direction TB
        LAYNCE["<b>LayNCE  (--zs_layout_infonce_weight)</b><br/>flatten(z_s) [B, 512]<br/>→ SemanticTokenInfoNCEHead: Linear(512→256)→ReLU→Linear(256→128)<br/>→ L2-norm  [B, 128]<br/>→ cross-batch dominant-category prototype InfoNCE<br/><i>Direct gradient to z_s tokens</i>"]
        POOLNCE["<b>PoolNCE  (--zs_pool_infonce_weight)</b><br/>z_s tokens [B, 16, 32]<br/>→ mean_pool → [B, 32]<br/>→ Linear(32→1024) → [B, 1024]<br/>→ MLP → [B, 16, 32] L2-norm<br/>→ compute_semantic_loss (same call as decoder pgNCE)<br/><i>Direct gradient to z_s tokens. Mirrors decoder InfoNCE at [B,1024]</i>"]
        PGNCE["<b>pgNCE  (--semantic_mode hidden)</b><br/>decoder hidden [B, 1024]<br/>→ SemanticProjectionHead MLP<br/>→ [B, 40000, 32] L2-norm<br/>→ InfoNCE on per-Gaussian ScanNet72 labels<br/><i>Indirect gradient to z_s (through self-attention)</i>"]
    end

    subgraph DECODER_A["DECODER  (all 512 tokens together — richest conditioning)"]
        direction TB
        MERGE["<b>Concat z_s + z_g → full Z</b><br/>[B, 16, 32] | [B, 496, 32]  →  [B, 512, 32]"]
        POSTKL["<b>post_kl  Linear(32 → 384)</b><br/>applied to all 512 tokens<br/>[B, 512, 32]  →  [B, 512, 384]"]
        FOURPE["<b>FourierDecoderPE  (--decoder_fourier_pe)</b><br/>8×8×8 scaffold grid → Fourier PE<br/>[B, 512, 384]  added to token embeddings"]
        TRANS["<b>Transformer  12 × Self-Attention layers</b><br/>z_s and z_g tokens attend to each other freely<br/>[B, 512, 384]  →  [B, 512, 384]"]
        FLAT["<b>Flatten</b><br/>[B, 512, 384]  →  [B, 196608]"]
        GSMLP["<b>GS_decoder MLP  (D=3, W=1024)</b><br/>Linear(196608→1024)→LN→ReLU × 2<br/>Linear(1024→560000)  where 560000 = 40000×14<br/>[B, 196608]  →  [B, 40000, 14]"]
        ACTS["<b>Output Activations</b><br/>pos   [B,40000,3]  ← raw (metres)<br/>color [B,40000,3]  ← raw residual (+ mean_color at vis time)<br/>opacity[B,40000,1] ← sigmoid → (0, 1)<br/>scale  [B,40000,3] ← exp → metres<br/>quat   [B,40000,4] ← L2-norm → unit quaternion"]
        OUT_A["<b>Reconstructed Gaussians  [B, 40000, 14]</b>"]
    end

    SE --> MUS_PROJ
    GT --> PREKL --> MUG_PROJ
    MUS_PROJ --> CAT_MU
    MUG_PROJ --> CAT_MU
    CAT_MU --> REPARAM --> RESHAPE
    RESHAPE --> ZS & ZG
    ZS --> H0 & H1 & H2
    H1 -.->|"same 480 dims"| H2
    H1 & H2 --> WARN
    ZS --> CROSS
    ZG --> CROSS
    MUS_PROJ --> ORTH
    MUG_PROJ --> ORTH
    ZS --> LAYNCE & POOLNCE
    ZS --> MERGE
    ZG --> MERGE
    MERGE --> POSTKL --> FOURPE --> TRANS --> FLAT --> GSMLP --> ACTS --> OUT_A
    TRANS -.->|"hidden state B×1024"| PGNCE
```

---

## 4. Strategy A — Disentangled VAE (with Structured Split)

**Flags:** `--latent_disentangle --semantic_token_heads --structured_layout_tokens`

Everything is identical to §3 except the auxiliary heads. Each head now receives
an **exclusive** token range — gradients from one head cannot interfere with another.

```mermaid
flowchart TD
    ZS["z_s = Z[:, 0:16, :]  [B, 16, 32]<br/>16 semantic tokens"]

    subgraph SPLIT_BOX["STRUCTURED TOKEN SPLIT  (--structured_layout_tokens=True)"]
        direction LR
        T0["<b>Token 0</b><br/>z[:, 0:32]<br/>[B, 32]"]
        T18["<b>Tokens 1 – 8</b><br/>z[:, 32 : 32+8×32]<br/>z[:, 32:288]  [B, 256]<br/>8 tokens × 32 dims"]
        T915["<b>Tokens 9 – 15</b><br/>z[:, 288:512]  [B, 224]<br/>7 tokens × 32 dims"]
    end

    subgraph HEADS_STRUCT["AUXILIARY HEADS — Exclusive Gradient Ranges"]
        direction TB
        H0S["<b>MeanColorHead</b><br/>Linear(32→64)→ReLU→Linear(64→3)→Sigmoid<br/>→  mean_color  [B, 3]<br/><i>Gradient: token 0 ONLY</i>"]
        H1S["<b>SceneSemanticHead</b><br/>Linear(256→128)→LN→ReLU→Linear(128→72)→Softmax<br/>→  label_dist_pred  [B, 72]<br/><i>Gradient: tokens 1–8 ONLY</i>"]
        H2S["<b>SceneLayoutHead</b><br/>Linear(224→512)→LN→ReLU→Linear(512→256)→LN→ReLU→Linear(256→216)<br/>→  category_centroids  [B, 72, 3]<br/><i>Gradient: tokens 9–15 ONLY</i>"]
        OK["✅ Zero cross-head gradient interference<br/>Each head shapes its own exclusive token range"]
    end

    ZS --> T0 & T18 & T915
    T0   -->|"B x 32"| H0S
    T18  -->|"B x 256"| H1S
    T915 -->|"B x 224"| H2S
    H0S & H1S & H2S --> OK
```

**Token assignment table:**

| Tokens | Dims | Head | Loss | Input to head |
|--------|------|------|------|---|
| Token 0 | [B, 32] | `MeanColorHead` | `L_color` = MSE vs GT mean RGB | `z[:, 0:32]` |
| Tokens 1–8 | [B, 256] | `SceneSemanticHead` | `L_sem` = KL vs GT label dist | `z[:, 32:288]` |
| Tokens 9–15 | [B, 224] | `SceneLayoutHead` | `L_layout` = masked MSE vs GT centroids | `z[:, 288:512]` |

**Without** `--structured_layout_tokens` (Strategy A §3): SceneSemanticHead **and**
SceneLayoutHead both receive `z[:, 32:512]` (same 480 floats). The gradient from one
head modifies the exact same parameters as the other — they interfere. The structured
split eliminates this completely.

The encoder, bottleneck, disentanglement losses, InfoNCE heads, and decoder are
**identical** to §3. Only the head input ranges change.

---

## 5. Strategy B1 — Cross-Attention Conditioning

**Flags:** `--decoder_layout_cross_attn`  (without `--latent_disentangle`)

Z contains **512 pure geometry tokens**. A separate deterministic
`Layout16Projector` maps `shape_embed → z_layout [B, 16, 32]` without VAE sampling.
This z_layout conditions every decoder transformer layer as Keys and Values.

Because `Layout16Projector` is a deterministic MLP, encoding the same scene twice
produces **identical** z_layout (cosine similarity = 1.000000). This enables stable
scene completion from partial observations (cosine sim 0.761 at 40% coverage).

```mermaid
flowchart TD
    SE_B["shape_embed  [B, 384]"]
    GT_B["geom_tokens  [B, 512, 384]"]

    subgraph BOT_B["VAE BOTTLENECK — Geometry Only (no split)"]
        direction TB
        PREKL_B["<b>pre_kl  Linear(384 → 64)</b><br/>geom_tokens [B, 256, 384]<br/>→ moments [B, 256, 64]<br/>→ sample → kl_embed [B, 256, 32]<br/>→ flatten → [B, 8192]"]
        MU_B["<b>kl_emb_proj_mean / var</b><br/>Linear(8192 → 16384)<br/>256 geom tokens upsampled to 512 latent positions<br/>μ [B, 16384]   σ² [B, 16384]"]
        REPARAM_B["<b>Reparameterisation</b><br/>z = μ + σ · ε<br/>z [B, 16384]  →  Z [B, 512, 32]<br/>512 pure geometry tokens<br/>KL-regularised → N(0, I)"]
    end

    subgraph PROJECTOR["LAYOUT16PROJECTOR — Deterministic (no VAE sampling)"]
        direction TB
        L16["<b>Layout16Projector(shape_embed)</b><br/>Linear(384→256)→LN→ReLU→Linear(256→512)<br/>reshape → [B, 16, 32]<br/>──────────────────────────────────────<br/><b>z_layout  [B, 16, 32]</b><br/>DETERMINISTIC: same scene → cos_sim = 1.000000<br/>Not part of Z · Not KL-regularised<br/>Separate from the geometry latent"]
    end

    subgraph HEADS_B["AUXILIARY HEADS on z_layout tokens"]
        direction LR
        HC_B["<b>lay_color_head</b><br/>z_layout[:, 0, :]  [B, 32]<br/>→ mean_color  [B, 3]"]
        HS_B["<b>lay_semantic_head</b><br/>z_layout[:, 1:, :].flatten  [B, 480]<br/>or tokens 1–8 if structured  [B, 256]<br/>→ label_dist_pred  [B, 72]"]
        HL_B["<b>lay_layout_head</b><br/>z_layout[:, 1:, :].flatten  [B, 480]<br/>or tokens 9–15 if structured  [B, 224]<br/>→ category_centroids  [B, 72, 3]"]
    end

    subgraph INFONCE_B["INFONCE HEADS on z_layout (optional)"]
        direction TB
        LAYNCE_B["<b>z_layout_infonce_head  (--zs_layout_infonce_weight)</b><br/>flatten(z_layout) [B, 512]<br/>→ SemanticTokenInfoNCEHead: Linear(512→256)→ReLU→Linear(256→128)<br/>→ L2-norm  [B, 128]<br/>→ prototype InfoNCE"]
        POOLNCE_B["<b>z_layout_pool_head  (--zs_pool_infonce_weight)</b><br/>z_layout [B, 16, 32]<br/>→ mean_pool → [B, 32] → Linear(32→1024) → [B, 1024]<br/>→ MLP → [B, 16, 32] L2-norm<br/>→ compute_semantic_loss"]
    end

    subgraph DECODER_B1["DECODER — ZSCondTransformerDecoder (B1 cross-attention)"]
        direction TB

        POSTKL_GEO["<b>post_kl  Linear(32 → 384)</b><br/>Z [B, 512, 32]  →  H_g [B, 512, 384]<br/>(standard post_kl, applied to geometry tokens)"]

        POSTKL_LAY["<b>post_kl_layout  Linear(32 → 384)</b><br/>z_layout [B, 16, 32]  →  H_lay [B, 16, 384]"]

        FOURPE_B["<b>FourierDecoderPE  (--decoder_fourier_pe)</b><br/>8×8×8 scaffold PE  →  added to H_g  [B, 512, 384]"]

        LAYER["<b>12 × ZSCondTransformerBlock</b><br/>────────────────────────────────────────────────<br/>① norm_sa(H_g)<br/>② Self-Attention: Q = K = V = H_g<br/>   H_g = H_g + self_attn_out       ← geometry attends to itself<br/>③ norm_ca(H_g)<br/>④ Cross-Attention: Q = H_g, K = H_lay, V = H_lay<br/>   H_g = H_g + cross_attn_out     ← geometry reads from z_layout<br/>   H_lay is injected at EVERY layer but never modified<br/>⑤ FFN: H_g = H_g + FFN(norm_ff(H_g))<br/>────────────────────────────────────────────────<br/>output: H_g [B, 512, 384]"]

        FLAT_B["<b>LayerNorm + Flatten</b><br/>[B, 512, 384]  →  [B, 196608]"]

        GSMLP_B["<b>GS_decoder_B MLP  (D=3, W=1024)</b><br/>Linear(196608→1024)→LN→ReLU × 2 → Linear(1024→560000)<br/>[B, 196608]  →  [B, 40000, 14]"]

        OUT_B["<b>Reconstructed Gaussians  [B, 40000, 14]</b>"]
    end

    SE_B --> L16
    SE_B --> HEADS_B
    GT_B --> PREKL_B --> MU_B --> REPARAM_B

    L16 --> HC_B & HS_B & HL_B
    L16 --> LAYNCE_B & POOLNCE_B
    L16 --> POSTKL_LAY

    REPARAM_B --> POSTKL_GEO --> FOURPE_B
    POSTKL_LAY -->|"H_lay B×16×384 — K and V per layer"| LAYER
    FOURPE_B --> LAYER
    LAYER --> FLAT_B --> GSMLP_B --> OUT_B
```

**Key difference between Strategy A and B1:**

| Property | Strategy A | Strategy B1 |
|---|---|---|
| z_layout source | VAE posterior (stochastic) | Layout16Projector (deterministic) |
| z_layout inside Z? | Yes — Z[:,0:16,:] | No — separate tensor |
| KL on z_layout | Yes (part of z_s) | No |
| Decoder z_layout role | One of 512 self-attention tokens | K, V in cross-attention every layer |
| Same-scene stability | cos_sim ≈ 0.97 (sampling noise) | cos_sim = 1.000000 exactly |
| Partial obs @40% | 0.20–0.37 | **0.761** |

---

## 6. Strategy D — Hybrid (A Bottleneck + B1 Decoder)

**Flags:** `--latent_disentangle --decoder_zs_cross_attn --semantic_token_heads --structured_layout_tokens`

Strategy D combines the two best properties across all strategies:

- **From Strategy A:** both z_s and z_g pass through the VAE bottleneck (KL-regularised toward
  N(0,I)). The full Z [B, 512, 32] has a generative path — a single Stage-2 DiT can generate
  both z_s and z_g from scratch for unconditional scene generation.
- **From Strategy B1:** z_g tokens [B, 496, 32] are the only decoder sequence input; z_s tokens
  [B, 16, 32] condition every decoder layer as cross-attention K, V via `ZSCondTransformerDecoder`.
  The architectural separation is harder than the soft self-attention mixing in Strategy A.

**Why the original run failed, and why it should work now:**
An early experiment with `decoder_zs_cross_attn=True` produced near-zero disentanglement.
The cause: z_s received gradient only through the cross-attention path during reconstruction —
indirect and too weak without explicit semantic targets. z_s degenerated to near-constant noise
and cross-attention weights collapsed toward zero.

The fix is LayNCE and PoolNCE, which give **direct** gradients to all 16 z_s tokens every step,
independent of the decoder. These force z_s to organise by scene type before reconstruction
sees it. EncDiff (NeurIPS 2024 Spotlight) provides theoretical support: cross-attention between
concept tokens and geometry is itself a strong disentanglement inductive bias, provided the
concept tokens carry meaningful signal.

```mermaid
flowchart TD
    SE["shape_embed  [B, 384]"]
    GT["geom_tokens  [B, 256, 384]"]

    subgraph BOT_H["VAE BOTTLENECK — Disentangled (same as Strategy A)"]
        direction TB
        MUS_H["<b>mu_s_proj_mean / var</b><br/>Linear(384 → 512)<br/>μ_s [B, 512]  σ²_s [B, 512]"]
        PREKL_H["<b>pre_kl  Linear(384 → 64)</b><br/>geom_tokens [B, 256, 384]<br/>→ flatten [B, 8192]<br/>→ kl_emb_proj_mean_g: Linear(8192 → 15872)"]
        CAT_H["<b>Concat and reparameterise</b><br/>z_s [B, 512]  +  z_g [B, 15872]<br/>→ Z [B, 512, 32]<br/>Both KL-regularised → N(0,I)"]
        ZS_H["<b>z_s = Z[:, 0:16, :]</b><br/>[B, 16, 32]  ← semantic tokens<br/>KL-regularised → generative path exists"]
        ZG_H["<b>z_g = Z[:, 16:, :]</b><br/>[B, 496, 32]  ← geometry tokens<br/>KL-regularised → generative path exists"]
    end

    subgraph DEC_H["DECODER — Cross-Attention Conditioning (same as Strategy B1)"]
        direction TB
        POSTKL_GEO_H["<b>post_kl_g  Linear(32 → 384)</b><br/>z_g [B, 496, 32]  →  H_g [B, 496, 384]<br/>ONLY geometry tokens in decoder sequence"]
        POSTKL_S_H["<b>post_kl_s  Linear(32 → 384)</b><br/>z_s [B, 16, 32]  →  H_s [B, 16, 384]"]
        FOURPE_H["<b>FourierDecoderPE</b><br/>496-token scaffold PE added to H_g"]
        LAYER_H["<b>12 × ZSCondTransformerBlock</b><br/>Self-Attention: H_g attends to H_g only<br/>Cross-Attention: Q=H_g, K=H_s, V=H_s<br/>→ geometry reads z_s at EVERY layer<br/>H_g [B, 496, 384]"]
        GSD_H["<b>GS_decoder MLP</b><br/>[B, 496 × 384]  →  [B, 40000, 14]"]
        OUT_H["<b>Reconstructed Gaussians  [B, 40000, 14]</b>"]
    end

    subgraph COMP_H["COMPARISON WITH STRATEGIES A AND B1"]
        direction TB
        R1["Strategy A: z_s in self-attention → soft mixing → stochastic partial obs (0.365 @40%)"]
        R2["Strategy B1: z_layout deterministic → perfect stability → cos_sim=1.000000"]
        R3["Hybrid: z_s stochastic but cross-attention → harder architectural separation<br/>Expected: better disentanglement than A, better generation than B1"]
    end

    SE --> MUS_H
    GT --> PREKL_H
    MUS_H --> CAT_H
    PREKL_H --> CAT_H
    CAT_H --> ZS_H & ZG_H
    ZS_H --> POSTKL_S_H
    ZG_H --> POSTKL_GEO_H
    POSTKL_GEO_H --> FOURPE_H
    POSTKL_S_H -->|"H_s B×16×384 — K and V per layer"| LAYER_H
    FOURPE_H --> LAYER_H
    LAYER_H --> GSD_H --> OUT_H
```

**Key difference from Strategy A:**

| Property | Strategy A | Strategy B1 | Hybrid |
|---|---|---|---|
| z_s source | VAE posterior | Layout16Projector | VAE posterior |
| z_s in Z? | Yes (tokens 0–15) | No (separate) | Yes (tokens 0–15) |
| KL on z_s | Yes | No | Yes |
| Decoder sequence | all 512 tokens | 512 geo tokens | 496 geo tokens only |
| z_s decoder role | One of 512 self-attn tokens | K, V in cross-attn | K, V in cross-attn |
| Generation possible? | Yes (Z is full) | Only conditioned | Yes (Z is full) |
| Partial obs expected | ~0.37 @40% | 0.761 @40% | Between A and B1 |

---

## 7. Strategy C — Baseline

**Flags:** none (all strategy flags `False`)

512 pure geometry tokens. No disentanglement. No layout conditioning. Used as the
control experiment to prove Strategy A's disentanglement is real.

```mermaid
flowchart TD
    SE_C["shape_embed  [B, 384]"]
    GT_C["geom_tokens  [B, 512, 384]"]

    subgraph BOT_C["VAE BOTTLENECK — No Split"]
        direction TB
        PREKL_C["<b>pre_kl  Linear(384 → 64)</b><br/>geom_tokens [B, 256, 384]<br/>→ kl_embed [B, 256, 32]<br/>→ flatten  [B, 8192]"]
        MU_C["<b>kl_emb_proj_mean / var</b><br/>Linear(8192 → 16384)<br/>256 geom tokens upsampled to 512 latent positions<br/>μ [B, 16384]   σ² [B, 16384]"]
        Z_C["<b>z = μ + σ · ε  →  Z [B, 512, 32]</b><br/>512 undifferentiated tokens<br/>No semantic / geometry split<br/>KL-regularised → N(0, I)"]
    end

    subgraph HEAD_C["AUXILIARY HEAD  (shape_embed, legacy path)"]
        HC_C["<b>MeanColorHead</b><br/>shape_embed [B, 384]<br/>→ mean_color [B, 3]<br/><i>L_color = MSE(pred, GT mean RGB)</i><br/>shape_embed is NOT part of Z in Strategy C<br/>The MeanColorHead gradient reaches Z only<br/>indirectly through decoder self-attention"]
    end

    subgraph DEC_C["DECODER — Standard Self-Attention"]
        direction TB
        POSTKL_C["<b>post_kl  Linear(32 → 384)</b><br/>Z [B, 512, 32]  →  H [B, 512, 384]"]
        FOURPE_C["<b>FourierDecoderPE</b><br/>Fourier PE added to H  [B, 512, 384]"]
        TRANS_C["<b>Transformer  12 × Self-Attention layers</b><br/>No layout conditioning  ·  No z_s / z_g split<br/>[B, 512, 384]  →  [B, 512, 384]"]
        FLAT_C["<b>Flatten</b><br/>[B, 512, 384]  →  [B, 196608]"]
        GSD_C["<b>GS_decoder MLP</b><br/>[B, 196608]  →  [B, 40000, 14]"]
        OUT_C["<b>Reconstructed Gaussians  [B, 40000, 14]</b>"]
    end

    subgraph CTRL["WHY STRATEGY C IS THE CONTROL EXPERIMENT"]
        direction TB
        NOTE1["<b>Swap experiment on Strategy C:</b><br/>Take first 16 token positions from scene B, last 496 from scene A<br/>Z_swap = [Z_B[:, 0:16, :] | Z_A[:, 16:, :]]<br/>Decode Z_swap → measure geo L2 vs original A"]
        NOTE2["Result: geometry disrupted by 6.5–53% of cross-scene baseline<br/>Colour still transfers (MeanColorHead on token 0)<br/>──────────────────────────────────────────────<br/>Proves: Strategy A's 0.08–0.27% swap ratio is REAL disentanglement<br/>Without disentanglement losses, the same token swap<br/>causes major geometry disruption"]
    end

    SE_C --> HEAD_C
    GT_C --> PREKL_C --> MU_C --> Z_C
    Z_C --> POSTKL_C --> FOURPE_C --> TRANS_C --> FLAT_C --> GSD_C --> OUT_C
```

---

## 8. InfoNCE Loss Variants

All variants use **cross-batch dominant-category prototype InfoNCE**:
`dom_cat(b) = argmax(label_dist[b])`. They differ in *what* is supervised and whether
the gradient path to z_s/z_layout is direct or indirect.

```mermaid
flowchart LR
    subgraph MECH["SHARED MECHANISM (all variants)"]
        direction TB
        M1["dom_cat = argmax(label_dist)  →  one integer per scene"]
        M2["pool embeddings across batch by dom_cat<br/>prototype_k = L2_norm(mean(embeds[label==k]))"]
        M3["sim[i,k] = embed_i · prototype_k / τ"]
        M4["cross_entropy loss: push embed_i toward its prototype"]
        M1 --> M2 --> M3 --> M4
    end

    subgraph LAYNCE_V["LayNCE  (--zs_layout_infonce_weight)"]
        direction TB
        LN1["flatten(z_s or z_layout)  [B, 512]"]
        LN2["SemanticTokenInfoNCEHead<br/>Linear(512→256)→ReLU→Linear(256→128)<br/>L2-norm  →  [B, 128]"]
        LN3["1 point per scene<br/><b>DIRECT gradient to z_s tokens</b>"]
        LN1 --> LN2 --> LN3
    end

    subgraph POOLNCE_V["PoolNCE  (--zs_pool_infonce_weight)"]
        direction TB
        PN1["z_s tokens  [B, 16, 32]"]
        PN2["mean_pool  →  [B, 32]<br/>Linear(32→1024)  →  [B, 1024]<br/>← same dim as decoder hidden!<br/>MLP(1024→512→256→16×32)<br/>L2-norm  →  [B, 16, 32]"]
        PN3["16 points per scene<br/><b>DIRECT gradient to z_s tokens</b><br/>Mirrors decoder InfoNCE architecture exactly"]
        PN1 --> PN2 --> PN3
    end

    subgraph PGNCE_V["pgNCE  (--semantic_mode hidden)"]
        direction TB
        PG1["decoder hidden  [B, 1024]"]
        PG2["SemanticProjectionHead MLP<br/>Linear(1024→512)→LN→ReLU<br/>Linear(512→256)→LN→ReLU<br/>Linear(256→40000×32)<br/>L2-norm  →  [B, 40000, 32]"]
        PG3["40,000 points per scene<br/>uses actual per-Gaussian ScanNet72 labels<br/><b>INDIRECT gradient</b> to z_s (through self-attention)"]
        PG1 --> PG2 --> PG3
    end

    subgraph POOLPG["Pool+pgNCE  (both above active)"]
        direction TB
        PP1["PoolNCE provides DIRECT gradient<br/>to z_s tokens → regularises z_s subspace"]
        PP2["pgNCE provides fine-grained<br/>per-Gaussian semantic supervision"]
        PP3["Combined: fixes pgNCE-alone coupling issue<br/>swap ratio: 4.5% (pgNCE alone) → 0.08% (Pool+pgNCE)"]
        PP1 --> PP3
        PP2 --> PP3
    end

    PG3 -.->|"indirect gradient"| LN3
    PG3 -.->|"indirect gradient"| PN3
```

**Comparison table:**

| Variant | Flag | Points/scene | Labels | Gradient to z_s | Visualisation PLY |
|---|---|:---:|---|:---:|---|
| LayNCE | `--zs_layout_infonce_weight` | 1 | dom_cat | **Direct** | `zs_layout_epoch_NNN.ply` |
| PoolNCE | `--zs_pool_infonce_weight` | 16 | dom_cat | **Direct** | `zs_pool_epoch_NNN.ply` |
| pgNCE | `--semantic_mode hidden` | 40,000 | per-Gaussian | Indirect | `scene{i}_semantic_infonce.ply` |
| Pool+pgNCE | both above | 40,016 | both | **Direct + Indirect** | both above |

---

## 9. Training Configuration

```yaml
# Architecture (shapevae-256.yaml)
# ─────────────────────────────────────────────────────────────────────────
# IMPORTANT: num_latents=256 refers to the number of GEOMETRY encoder queries.
# The encoder has 1 + 256 = 257 total queries:
#   query 0 → shape_embed [B, 384]   (global scene summary)
#   queries 1–256 → geom_tokens [B, 256, 384]   (per-region geometry)
# The 512-token decoder latent Z is produced by upsampling:
#   Strategy A:  shape_embed → mu_s_proj → 16 tokens (z_s)
#                geom_tokens → Linear(8192→15872) → 496 tokens (z_g)
#                concat → Z [B, 512, 32]
#   Strategy B1: geom_tokens → Linear(8192→16384) → 512 tokens
#                shape_embed → Layout16Projector → z_layout [B, 16, 32]  (separate)
#   Strategy C:  geom_tokens → Linear(8192→16384) → 512 tokens
#   Strategy D:  shape_embed → mu_s_proj → 16 tokens (z_s)  [same as A]
#                geom_tokens → Linear(8192→15872) → 496 tokens (z_g)  [same as A]
#                decoder: z_g in sequence + z_s as cross-attn K/V per layer
#                flags: --latent_disentangle --decoder_zs_cross_attn
# ─────────────────────────────────────────────────────────────────────────
num_latents:       256    # geometry encoder queries (total encoder = 257 with shape_embed)
embed_dim:         32     # token dimension in Z; Z always [B, 512, 32]
width:             384    # transformer hidden width throughout
encoder_layers:    6      # CrossAttentionEncoder self-attention layers
decoder_layers:    12     # transformer decoder layers
heads:             12     # attention heads
num_freqs:         8      # Fourier embedding frequencies

# Training (gs_can3tok_2.py)
batch_size:        64     # per GPU × 2 H100s = 128 effective
learning_rate:     1e-4   # cosine decay to lr × lr_min_ratio
lr_min_ratio:      0.1    # floor = 1e-5
warmup_steps:      300
kl_weight:         1e-5   # deliberately small — allows large σ for better reconstruction
semantic_dims:     512    # z_s spans first 512 dims of z = 16 tokens × 32 dims
cross_recon_weight: 0.3
ortho_weight:       0.1
layout_loss_weight: 0.3
scene_semantic_weight: 0.3
```

---

## 10. Loss Function Summary

```
L_total = L_recon                              (pos + color + opacity + scale + quat)
        + kl_weight              × L_KL        (= 1e-5 · keeps latent near N(0,I))
        + mean_color_weight      × L_color      (MeanColorHead MSE · weight = 1.0)
        + scene_semantic_weight  × L_sem_kl     (SceneSemanticHead KL · weight = 0.3)
        + layout_loss_weight     × L_layout_mse (SceneLayoutHead masked MSE · weight = 0.3)
        + cross_recon_weight     × L_cross      (geometry sufficiency · weight = 0.3)
        + ortho_weight           × L_ortho      (μ_s ⊥ μ_g · weight = 0.1)
        + zs_layout_infonce_weight × L_layNCE   [LayNCE,    optional]
        + zs_pool_infonce_weight   × L_poolNCE  [PoolNCE,   optional]
        + segment_loss_weight      × L_pgNCE    [pgNCE,     optional]
```

**Gradient paths to z_s:**

| Loss | Path | Strength |
|---|---|---|
| L_recon | via decoder self-attention (indirect) | Dominant — reconstruction signal overwhelms others |
| L_KL | directly on μ_s, σ²_s | Weak (weight 1e-5) |
| L_color | MeanColorHead → z[:,0,:] → z_s | Strong for token 0 only |
| L_sem | SceneSemanticHead → z[:,32:] → z_s | Tokens 1–15 (or 1–8 if structured) |
| L_layout | SceneLayoutHead → z[:,288:] → z_s | Tokens 9–15 (or 1–15 if unstructured) |
| L_cross | via decode(z_cross) which uses z_g — gradient isolates z_g | Strengthens z_g geometry independence |
| L_ortho | directly on μ_s and μ_g projections | Pushes subspaces to be orthogonal |
| L_layNCE | SemanticTokenInfoNCEHead → flatten(z_s) | Direct, all 16 tokens |
| L_poolNCE | ZSTokenPoolProjectHead → mean_pool(z_s) | Direct, pooled signal |
| L_pgNCE | SemanticProjectionHead → decoder hidden → z_s | Indirect through all transformer layers |

---

## 11. Second-Stage Generation Pipeline

Stage 2 learns to transform Gaussian noise into the latent Z that the frozen Stage 1
decoder can decode into a coherent 3DGS scene. The key constraint: every tensor the
decoder depends on must have a generative path from noise.

| | Strategy A | Strategy B1 | Strategy D |
|---|---|---|---|
| Z fully KL-regularised? | Yes | Z yes, z_layout **no** | Yes |
| Unconditional generation | Yes | **No** | Yes |
| Scene completion | Yes, moderate | Yes, **excellent** | Yes, moderate |
| Stage 2a (layout tokens) | Layout DiT on z_s [B,16,32] | N/A | Layout DiT on z_s [B,16,32] |
| Stage 2b (geometry tokens) | Flat DiT, z_s concatenated | Inpainting DiT, z_layout fixed | Cross-attn DiT, z_s as K,V |
| Architecture mirrors VAE decoder? | Partial | Strong | **Strong** |

---

### Strategy A — Stage 2

```mermaid
flowchart TD
    subgraph GEN["STRATEGY A — TEXT-CONDITIONED GENERATION"]
        direction TB
        TEXT["Text / scene-class token"]
        N1["noise [B, 16, 32]  ~  N(0, I)"]

        D1["<b>Stage 1 — Layout DiT  (~6 transformer layers)</b><br/>Flow matching: N(0,I) → P(z_layout | text)<br/>Train target: Z[:, 0:16, :]  from Can3Tok VAE<br/>Text conditioning via cross-attention<br/>─────────────────────────────────<br/>Output: z_layout [B, 16, 32]<br/>encodes: scene type · dominant categories<br/>         colour palette · spatial centroids"]

        ZL_GEN["z_layout  [B, 16, 32]"]
        N2["noise [B, 496, 32]  ~  N(0, I)"]

        D2["<b>Stage 2 — Geometry DiT  (~16–28 transformer layers)</b><br/>Input: concat(z_layout, noisy_z_geo) [B, 512, 32]<br/>Flow matching: N(0,I) → P(z_geo | z_layout)<br/>Train target: Z[:, 16:, :]  from Can3Tok VAE<br/>Same Z structure as VAE decoder input<br/>─────────────────────────────────<br/>Output: z_geo [B, 496, 32]"]

        ZG_GEN["z_geo  [B, 496, 32]"]

        DECODE["<b>VAE Decoder  (frozen Can3Tok)</b><br/>Z = concat(z_layout, z_geo)  [B, 512, 32]<br/>post_kl → FourierPE → Transformer → GS_decoder<br/>mean_color = MeanColorHead(Z[:, 0, :])<br/>final_color = color_residuals + mean_color<br/>─────────────────────────────────<br/>40,000 Gaussians — no encoder · no GT needed"]

        PLY["<b>Generated Scene PLY</b><br/>[B, 40000, 14]"]

        TEXT --> D1
        N1   --> D1
        D1   --> ZL_GEN --> D2
        N2   --> D2
        D2   --> ZG_GEN
        ZL_GEN --> DECODE
        ZG_GEN --> DECODE
        DECODE --> PLY
    end

    subgraph COMP["SCENE COMPLETION  (partial scan → full scene)"]
        direction TB
        PARTIAL["Partial scan  (30–80% of Gaussians)"]
        ENC_P["<b>Can3Tok Encoder</b><br/>→ z_layout  (stable from 30%+ coverage)<br/>→ z_geo_partial  [B, 496, 32]"]
        MASK["<b>Construct noisy z_geo</b><br/>Observed voxels:    z_geo from encoder  (held fixed)<br/>Unobserved voxels:  Gaussian noise"]
        INPAINT["<b>Stage 2 DiT — Inpainting</b><br/>Denoise only the unobserved z_geo tokens<br/>Observed tokens fixed throughout denoising"]
        FULL["<b>VAE Decoder  (frozen)</b><br/>Z = concat(z_layout, z_geo_complete)<br/>→ full scene PLY"]
        PARTIAL --> ENC_P --> MASK --> INPAINT --> FULL
    end
```

---

### Strategy B1 — Stage 2 (Scene Completion Only)

Because z_layout is deterministic (Layout16Projector, not KL-regularised), there is no
distribution P(z_layout) to sample from. Stage 2 for B1 is therefore **completion only**,
never unconditional generation.

```mermaid
flowchart TD
    subgraph B1_COMP["STRATEGY B1 — SCENE COMPLETION"]
        direction TB
        PARTIAL_B["Partial scan (30–80% of Gaussians)"]
        ENC_B["<b>Can3Tok Encoder</b><br/>shape_embed → Layout16Projector → z_layout [B,16,32]<br/>DETERMINISTIC: cos_sim = 1.000000 even at 40% coverage<br/>geom_tokens → Z [B, 512, 32]  (partial geometry)"]
        MASK_B["<b>Construct noisy Z_geo</b><br/>Observed voxels: Z from encoder (held fixed)<br/>Unobserved voxels: Gaussian noise"]
        INPAINT_B["<b>Completion DiT</b><br/>Architecture: ZSCondTransformerDecoder (mirrors VAE decoder exactly)<br/>z_layout [B,16,32] as cross-attn K/V — same as VAE decoder<br/>Only unobserved Z tokens denoised; observed tokens fixed<br/>Loss: flow matching on unobserved tokens only"]
        FULL_B["<b>VAE Decoder (frozen B1)</b><br/>ZSCondTransformerDecoder(Z_complete, z_layout) → scene"]
        NOTE_B["<b>Why B1 is ideal for completion:</b><br/>z_layout stable from 30% partial scan (0.761 @40%)<br/>No sampling noise — same partial scan always produces<br/>identical z_layout conditioning"]
        PARTIAL_B --> ENC_B --> MASK_B --> INPAINT_B --> FULL_B
        NOTE_B -.-> INPAINT_B
    end
```

---

### Strategy D — Stage 2 (Generation + Completion)

Same two-stage structure as Strategy A. The difference is Stage 2b uses cross-attention
conditioning — a structural mirror of the VAE decoder.

```mermaid
flowchart TD
    subgraph D_GEN["STRATEGY D — HIERARCHICAL GENERATION"]
        direction TB
        TEXT_D["Text / scene-class token"]
        N1_D["noise [B, 16, 32]  ~  N(0, I)"]

        D1_D["<b>Stage 2a — Layout DiT  (~6 layers)</b><br/>Identical to Strategy A Stage 2a<br/>Flow matching: N(0,I) → P(z_s | text)<br/>Train target: Z[:, 0:16, :] from frozen Stage 1 encoder<br/>─────────────────────────────────<br/>Output: z_s [B, 16, 32]<br/>Encodes: scene type, colour, spatial layout"]

        ZS_D["z_s  [B, 16, 32]"]
        N2_D["noise [B, 496, 32]  ~  N(0, I)"]

        D2_D["<b>Stage 2b — Geometry DiT  (~16–24 layers)</b><br/>Architecture: ZSCondTransformerBlock (mirrors VAE decoder)<br/>  z_g [B, 496, 32] = denoising sequence<br/>  z_s [B, 16, 32]  = cross-attn K/V at every layer<br/>Flow matching: N(0,I) → P(z_g | z_s)<br/>Train target: Z[:, 16:, :] from frozen Stage 1 encoder<br/>─────────────────────────────────<br/>Output: z_g [B, 496, 32]"]

        ZG_D["z_g  [B, 496, 32]"]

        DEC_D["<b>VAE Decoder (frozen Strategy D)</b><br/>Z = concat(z_s, z_g)  [B, 512, 32]<br/>post_kl_s(z_s) → H_s [B, 16, 384]  (cross-attn K/V)<br/>post_kl_g(z_g) → H_g [B, 496, 384] (sequence)<br/>ZSCondTransformerDecoder(H_g, H_s) → GS_decoder_new → scene"]

        PLY_D["<b>Generated Scene PLY</b><br/>[B, 40000, 14]"]

        TEXT_D --> D1_D
        N1_D   --> D1_D
        D1_D   --> ZS_D --> D2_D
        N2_D   --> D2_D
        D2_D   --> ZG_D
        ZS_D   --> DEC_D
        ZG_D   --> DEC_D
        DEC_D  --> PLY_D
    end

    subgraph D_COMP["STRATEGY D — SCENE COMPLETION"]
        direction TB
        PARTIAL_D["Partial scan (30–80% of Gaussians)"]
        ENC_D["<b>Can3Tok Encoder</b><br/>z_s [B,16,32] from VAE posterior (stochastic — sampling noise)<br/>z_g [B,496,32] from partial geometry"]
        MASK_D["<b>Construct noisy z_g</b><br/>Observed voxels: z_g from encoder (held fixed)<br/>Unobserved voxels: Gaussian noise"]
        INPAINT_D["<b>Stage 2b DiT — Inpainting</b><br/>z_s from partial encoder (fixed conditioning)<br/>Denoise only unobserved z_g tokens"]
        FULL_D["<b>VAE Decoder (frozen)</b><br/>concat(z_s, z_g_complete) → scene"]
        PARTIAL_D --> ENC_D --> MASK_D --> INPAINT_D --> FULL_D
    end
```

**Key difference from Strategy A in Stage 2b:** Strategy A concatenates z_s into the DiT
sequence (all 512 tokens together). Strategy D uses z_s as cross-attention K,V, which mirrors
the VAE decoder architecture exactly. This structural consistency means the DiT is learning
to invert the same conditioning structure the VAE decoder uses — the representations are
already aligned in the right space.

---

## 12. Dataset

**SceneSplat-7K** — 7,916 indoor 3DGS scenes from ScanNet, ScanNet++, Replica,
Hypersim, 3RScan, ARKitScenes, and Matterport3D. Per-Gaussian ScanNet72 semantic labels.

Preprocessing pipeline (implemented in `gs_dataset_scenesplat.py`):
- Positions normalised to 10 m radius canonical sphere using `normalize_to_canonical_sphere`
- Top-40,000 Gaussians sampled deterministically by opacity per epoch
- Color: per-scene mean subtracted → residuals ∈ [−0.5, +0.5] when `color_residual=True`
- Encoder voxelisation: FNV hash on 40³ grid → `voxel_centers [N,3]` and `voxel_id [N]`
- Layout: per-category centroid computed per scene for `SceneLayoutHead`

---

## 13. Ablation Results

### Architecture Evaluation (9 models, 4 experiments)

| Model | Config | Exp1 ratio\_lay | Exp2 %P1 | Exp3 @40% | Exp4 valid% |
|---|---|:---:|:---:|:---:|:---:|
| C — Baseline | color\_res + FourierPE | 1.048 | 6.5% | −0.05 | 100% |
| A — Unstructured | + latent\_disent | 0.985 | 0.27% | 0.257 | 94% |
| A — Structured | + structured\_split | 0.968 | 0.23% | 0.204 | 81% |
| A + LayNCE | + layout InfoNCE | 0.967 | **0.14%** | 0.249 | 95% |
| A + PoolNCE | + pool InfoNCE | 1.027 | 0.18% | 0.278 | 91% |
| A + pgNCE | + decoder InfoNCE | 1.036 | 4.5% | 0.272 | 86% |
| A + Pool+pgNCE | + pool+decoder | 0.979 | **0.08%** | **0.365** | 95% |
| **B1** | cross-attn conditioning | **1.240** | 0.34% | **0.761** | 97% |
| B2 | additive bias | 1.046 | 97% | −0.06 | 99.5% |

- **Exp1 ratio\_lay**: inter/intra cosine dist in z\_layout — higher = better category clustering
- **Exp2 %P1**: swap geometry L2 as % of cross-scene baseline — lower = better disentanglement
- **Exp3 @40%**: z\_layout cosine similarity from 40% partial scan — higher = better completion
- **Exp4 valid%**: fraction of N(0,I) samples decoding to valid geometry

### Semantic Feature Quality (pgNCE vs Pool+pgNCE)

| Metric | pgNCE | Pool+pgNCE |
|---|:---:|:---:|
| Fisher Ratio (per-Gaussian) | **0.440** | 0.411 |
| Mean Intra Distance | 0.491 | **0.453** |
| Mean Silhouette | **−0.056** | −0.066 |
| Linear Probe Accuracy | 26.01%±0.30 | **26.07%±0.96** (tied) |
| μ_s Fisher (scene-level) | **1.987** | 1.538 |
| pool\_hidden Fisher | 1.751 | 0.994 |

---

## 14. Key Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--latent_disentangle` | False | Split Z into z\_s (tokens 0–15) and z\_g (tokens 16–511) |
| `--semantic_dims` | 512 | z\_s dims; token count = 512÷32 = 16 tokens |
| `--decoder_layout_cross_attn` | False | Strategy B1: 512 geom + z\_layout as cross-attn K,V per layer |
| `--decoder_layout_additive` | False | Strategy B2: 512 geom + z\_layout as broadcast additive bias |
| `--decoder_zs_cross_attn` | False | Strategy D: z\_g [B,496,32] in sequence + z\_s [B,16,32] as cross-attn K,V; requires `--latent_disentangle` |
| `--structured_layout_tokens` | False | Exclusive split: tokens 1–8 → semantic, 9–15 → layout |
| `--color_residual` | False | DC/AC color; MeanColorHead on z\_s token 0 |
| `--semantic_token_heads` | False | Heads on z tokens (inference-clean; requires latent\_disentangle) |
| `--scene_semantic_head` | False | KL loss: label distribution prediction head |
| `--scene_layout_head` | False | MSE loss: category centroid prediction head |
| `--cross_recon_weight` | 0.3 | Cross-recon: geometry sufficiency of z\_g |
| `--ortho_weight` | 0.1 | Orthogonality: μ\_s ⊥ μ\_g |
| `--decoder_fourier_pe` | False | 3D Fourier PE from 8³ scaffold grid (recommended) |
| `--zs_layout_infonce_weight` | 0.0 | LayNCE: flatten(z\_s)→MLP→[B,128]→prototype NCE |
| `--zs_pool_infonce_weight` | 0.0 | PoolNCE: pool→1024→NCE (mirrors decoder hidden InfoNCE) |
| `--semantic_mode` | none | pgNCE: `hidden` enables per-Gaussian decoder InfoNCE |
| `--segment_loss_weight` | 0.0 | pgNCE weight |
| `--pca_vis_freq` | 500 | Write PCA PLYs every N epochs |

---

## 15. Code Structure

```
.
├── gs_can3tok_2.py                     # Training loop · all losses · eval · PLY
├── gs_dataset_scenesplat.py            # Dataset · preprocessing · voxelisation
├── semantic_losses.py                  # All 5 InfoNCE loss functions
├── pca_feature_visualization.py        # PCA PLY writers for all visualisation types
├── gs_ply_reconstructor.py             # Write PLY in SuperSplat format
├── model/configs/aligned_shape_latents/
│   └── shapevae-256.yaml               # Architecture hyperparameters
└── model/michelangelo/models/tsal/
    └── sal_perceiver_dist_changes.py   # Full model
        ├── CrossAttentionEncoder       # Shared encoder (all strategies)
        ├── AlignedShapeLatentPerceiver # Main model — strategy dispatcher
        │
        ├── Strategy A components
        │   ├── MeanColorHead              [B,32]   → RGB [B,3]
        │   ├── SceneSemanticHead          [B,256/480] → dist [B,72]
        │   ├── SceneLayoutHead            [B,224/480] → centroids [B,72,3]
        │   ├── SemanticTokenInfoNCEHead   flatten(z_s)→[B,128]  (LayNCE)
        │   └── ZSTokenPoolProjectHead     pool(z_s)→[B,1024]→[B,16,32]  (PoolNCE)
        │
        ├── Strategy D components  (shares bottleneck with A, decoder with B1)
        │   ├── post_kl_g                  z_g [B,496,32] → [B,496,384]
        │   ├── post_kl_s                  z_s [B,16,32]  → [B,16,384]
        │   ├── ZSCondTransformerDecoder   12× self_attn(z_g) + cross_attn(z_g,z_s)
        │   └── GS_decoder_new             flat MLP 496×384=190464 → [B,40000,14]
        │
        ├── Strategy B1 components
        │   ├── Layout16Projector          shape_embed→z_layout [B,16,32]
        │   ├── ZSCondTransformerBlock     self_attn(z_g) + cross_attn(z_g,z_layout)
        │   └── ZSCondTransformerDecoder   12×ZSCondTransformerBlock
        │
        ├── Strategy B2 components
        │   └── LayoutAdditiveConditioner  flatten(z_layout)→[B,384] broadcast bias
        │
        ├── Shared decoder components
        │   ├── FourierDecoderPE           8³ scaffold → Fourier PE [B,512,384]
        │   ├── GS_decoder                 flat MLP → [B,40000,14]
        │   └── SemanticProjectionHead     hidden→[B,40000,32]  (pgNCE)
        │
        └── job_scripts/
            ├── run_can3tok_scaffold.job   SLURM · all flags documented
            └── accelerate_config.yaml     2× H100 DDP config
```