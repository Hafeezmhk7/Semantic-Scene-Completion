# Semantic-Aware Can3Tok: 3D Scene Generation with Semantic Scene Completion

## Implementation Differences from Original Can3Tok

### 1. Dataset Format and Normalization

**Can3Tok Original:**
- Data Format: Raw 3DGS PLY files
- Scale Storage: Log-space (stores `log(scale_meters)`)
- Canonical Normalization:
```python
  # Positions
  positions_norm = (positions - center) * scale_factor

  # Scales (log-space arithmetic)
  scales_norm = scales + log(scale_factor)
  # Because: log(scale_new) = log(scale_old x factor)
  #                          = log(scale_old) + log(factor)
```

**Our Implementation:**
- Data Format: SceneSplat preprocessed .npy files
- Scale Storage: Linear-space (stores `scale_meters` after exp())
- Canonical Normalization:
```python
  # Positions (same as Can3Tok)
  positions_norm = (positions - center) * scale_factor

  # Scales (linear-space arithmetic)
  scales_norm = scales * scale_factor
  # Direct multiplication because scales are already in meters
```

Note: Log-space scale loss was tested and abandoned. It caused immediate
scale collapse to <1cm because the loss domain mismatches the linear-space
targets. Standard L2 on linear-space scales with softplus activation is
the correct formulation for our dataset.

**Location:** `normalize_to_canonical_sphere()` in `gs_dataset_scenesplat.py`

### 2. Model Output
- 14 parameters including RGB colors (xyz, rgb, opacity, scale, quaternion)

### 3. Decoder Architecture
- Can3Tok: Raw outputs without explicit activations
- Our Implementation: Added activation functions in final layer:
  - Colors: sigmoid -> [0, 1]
  - Opacity: sigmoid -> [0, 1]
  - Scales: softplus -> (0, +inf)
  - Quaternions: L2 normalization -> ||q|| = 1.0
- Location: `GS_decoder` class in `sal_perceiver_II_initialization.py`

### 4. Encoder Structure
- Fixed Issue: Previously missing voxel center Fourier embeddings
- Current: Dual Fourier embeddings (voxel centers + actual xyz positions)
- Location: `CrossAttentionEncoder` in `sal_perceiver_II_initialization.py`

---
## File Structure and Usage

### Training Files

**Main Training Script:**
- `gs_can3tok_2.py` - Primary training script with all loss functions and training loop

**Model Architecture:**
- `model/michelangelo/models/tsal/sal_perceiver_II_initialization.py` - Contains:
  - `AlignedShapeLatentPerceiver` - Main VAE model with 3 semantic modes
  - `CrossAttentionEncoder` - Encoder with dual Fourier embeddings
  - `GS_decoder` - Decoder with activation functions (outputs 14 params)
  - `SemanticProjectionHead` classes - For semantic feature extraction

**Dataset Loading:**
- `gs_dataset_scenesplat.py` - Loads SceneSplat scenes with ScanNet72 labels
  - Handles importance sampling (top 40k Gaussians by opacity, deterministic)
  - Voxelization for positional encoding
  - Canonical sphere normalization

**Loss Functions:**
- `semantic_losses.py` - Contains:
  - `ScanNet72SemanticLoss` - InfoNCE contrastive loss for semantic learning
  - Handles segment-level supervision

**PLY Reconstruction:**
- `gs_ply_reconstructor.py` - Converts model outputs to 3DGS PLY format
  - RGB to SH coefficient conversion: `f_dc = (RGB - 0.5) / C0`
  - Inverts activations: logit(opacity), log(scale)
  - Normalizes quaternions

**Visualization:**
- `pca_feature_visualization.py` - PCA coloring of per-Gaussian semantic features
  - Writes PLY files viewable in CloudCompare
  - No Open3D dependency, uses plyfile only
- `visualize_input_pca.py` - Standalone script to generate PCA visualizations
  of raw input scenes (no model forward pass needed)
  - Outputs per scene: position PCA, original colors, scale PCA,
    opacity grayscale, all-params PCA

**Configuration:**
- `model/configs/aligned_shape_latents/shapevae-256.yaml` - VAE architecture config

**Job Submission:**
- `job_can3tok.sh` - SLURM job script for cluster training

---

## Training Pipeline Flow

### 1. Data Loading (gs_dataset_scenesplat.py)
```
For each scene:
1. Load .npy files (coord, color, opacity, scale, quat, segment, instance)
2. Sample top 40k Gaussians by opacity (deterministic argsort, same every epoch)
3. Apply canonical sphere normalization:
   - Center positions at origin
   - Scale to fit in 10m radius sphere
   - Scale proportionally applies to scales (linear space)
4. Voxelize: assign each Gaussian to 40x40x40 grid (resolution=0.4m)
5. Assemble 18-channel features:
   [voxel_centers(3), voxel_id(1), xyz(3), color(3), opacity(1), scale(3), quat(4)]
6. Return: features, segment_labels, instance_labels
```

### 2. Forward Pass (sal_perceiver.py)
```
Input: [B, 40k, 18] features

ENCODER:
1. Extract voxel_centers and xyz from features
2. Apply dual Fourier embeddings:
   - voxel_centers -> Fourier -> coarse spatial encoding
   - xyz -> Fourier -> fine spatial encoding
3. Concatenate [xyz_fourier, voxel_fourier, gaussian_params]
4. Input projection -> transformer width
5. Cross-attention: queries attend to Gaussian features
6. Self-attention: refine latent tokens
   Output: [B, 512, 384] latent tokens

VAE BOTTLENECK:
1. Split: shape_embed [B, 384] + latents [B, 511, 384]
2. Project to mean mu and log_var
3. Sample: z = mu + sigma * epsilon
   Output: [B, 512, 32] compressed latents

DECODER:
1. Post-KL projection: [B, 512, 32] -> [B, 512, 384]
2. Transformer self-attention
3. Flatten: [B, 512*384]
4. 8-layer MLP decoder with activations:
   - Raw output: [B, 40k, 14]
   - Apply sigmoid(colors), sigmoid(opacity), softplus(scales), normalize(quat)
   Output: [B, 40k, 14] activated Gaussians

SEMANTIC HEAD (if enabled):
1. Extract features from decoder hidden state or Gaussian params
2. Project to [B, 40k, 32] per-Gaussian features
3. L2 normalize features
   Output: [B, 40k, 32] semantic features
```

### 3. Loss Computation (gs_can3tok_2.py)
```
RECONSTRUCTION LOSS:
L_recon = ||Gaussians_pred - Gaussians_target||_2 / batch_size
Standard L2 across all 14 parameters. Scale loss computed in linear space.

KL DIVERGENCE:
L_KL = KL(q(z|x) || N(0,I)) weighted by kl_weight (1e-6)
KL loss reaches 50k-100k in magnitude but effective gradient contribution
is kl_weight * KL = 0.05-0.10, which is manageable.

SEMANTIC LOSS (if semantic_mode != 'none'):
1. For each scene, extract valid Gaussians with segment labels
2. Subsample 10k Gaussians for efficiency
3. Create category prototypes by averaging features per category
4. Compute InfoNCE loss with temperature=0.1
L_segment = CrossEntropy(similarity_matrix, target_categories)
Weighted by segment_loss_weight (default 0.3)

TOTAL LOSS:
L_total = L_recon + kl_weight * L_KL + segment_loss_weight * L_segment
```

### 4. Training Loop
```
For each epoch:
  For each batch:
    1. Load batch of scenes with features and labels
    2. Forward pass through model
    3. Compute losses
    4. Backward pass and optimizer step
    5. Log metrics (L2 error, individual parameter losses)

  Every eval_every epochs (default 50):
    - Run validation
    - Save checkpoint
    - Reconstruct scenes to PLY files (if epoch % recon_ply_freq == 0)
    - Generate PCA visualizations (if semantic enabled and epoch % pca_vis_freq == 0)
    - Track best model by validation L2 error
```

Note: PCA visualization only writes files when both eval_every and
pca_vis_freq conditions are satisfied simultaneously. Set pca_vis_freq
equal to eval_every to get PCA at every validation checkpoint.

### 5. PLY Reconstruction (gs_ply_reconstructor.py)
```
Input: [B, 40k, 14] model predictions (post-activation)

For each scene:
1. Extract parameters:
   - positions [40k, 3]
   - colors [40k, 3] in [0,1]
   - opacity [40k, 1] in [0,1]
   - scales [40k, 3] in (0, inf)
   - quaternions [40k, 4] normalized

2. Convert to PLY format:
   - colors: f_dc = (RGB - 0.5) / C0  (SH coefficient conversion)
   - opacity: logit(opacity) = log(p / (1-p))  (inverse sigmoid)
   - scales: log(scale)  (inverse softplus)
   - quaternions: already normalized

3. Write to .ply file
4. View in supersplat.at
```

---

## Current Training Configuration

### Validated Hyperparameters
```python
# Architecture
num_latents = 256
embed_dim = 32
width = 384
num_encoder_layers = 6
num_decoder_layers = 12

# Optimization
learning_rate = 1e-4
batch_size = 64
epochs = 700
kl_weight = 1e-6           # Critical: 1e-5 causes KL to dominate at late epochs

# Loss
scale_norm_mode = 'linear'  # Do not use 'log': causes immediate scale collapse
segment_loss_weight = 0.3   # For semantic runs
semantic_temperature = 0.1
semantic_subsample = 10000

# Dataset
sampling_method = 'opacity'    # Deterministic top-40k by opacity
canonical_normalization = True  # 10m radius sphere
train_scenes = 300
val_scenes = 30
eval_every = 50
```

### Semantic Modes
```bash
--semantic_mode none        # No semantic learning (pure VAE)
--semantic_mode geometric   # Per-splat projection from 14-dim Gaussian params
--semantic_mode hidden      # Projection from 1024-dim decoder hidden state
--semantic_mode attention   # Cross-attention between positions and transformer tokens
```

---

## Running Training

### Basic Training (No Semantics)
```bash
python gs_can3tok_2.py \
    --batch_size 64 \
    --num_epochs 700 \
    --lr 1e-4 \
    --kl_weight 1e-6 \
    --scale_norm_mode linear \
    --semantic_mode none \
    --train_scenes 300 \
    --eval_every 50
```

### Training with Semantic Learning
```bash
python gs_can3tok_2.py \
    --batch_size 64 \
    --num_epochs 700 \
    --lr 1e-4 \
    --kl_weight 1e-6 \
    --scale_norm_mode linear \
    --semantic_mode geometric \
    --segment_loss_weight 0.3 \
    --semantic_subsample 10000 \
    --train_scenes 300 \
    --eval_every 50
```

### Generate Input Scene PCA Visualizations
```bash
python visualize_input_pca.py \
    --num_scenes 5 \
    --output_dir ./input_pca_vis \
    --sampling_method opacity \
    --scale_norm_mode linear
```

Outputs 5 PLY files per scene for comparison with model PCA in CloudCompare:
position PCA, original colors, scale PCA, opacity grayscale, all-params PCA.

---

## Evaluation Metrics

### Reconstruction Quality

1. **L2 Error** (primary metric)
```python
L2_error = ||G_pred - G_true||_2 / batch_size
```

2. Individual parameter losses tracked separately for position, color,
   opacity, scale, and rotation to diagnose training dynamics.

3. Scale distribution tracked across bins (<5cm, 5-10cm, 10-20cm, >20cm)
   as a proxy for geometric representation quality.

---

## Dataset: SceneSplat with ScanNet72 Labels

### Dataset Structure

Each scene directory contains:
```
scene_dir/
├── coord.npy          # [N, 3] Gaussian positions
├── color.npy          # [N, 3] RGB values
├── scale.npy          # [N, 3] Anisotropic scales (linear space, metres)
├── quat.npy           # [N, 4] Rotation quaternions
├── opacity.npy        # [N] Opacity values [0,1]
├── segment.npy        # [N] ScanNet72 category labels (0-71)
└── instance.npy       # [N] Instance IDs (-1 = background)
```

### ScanNet72 Categories

- Total categories: 72 (indices 0-71)
- Missing categories: [13, 53, 61] (never appear in dataset, handled in loss)
- Most frequent: Wall (0), Floor (1), Ceiling (23), Cabinet (2), Window (8)

---

## Output Files

### Checkpoints
```
checkpoints/RGB_job_<jobid>_<semantic_mode>/
├── epoch_050.pth              # Model checkpoint (every 50 epochs)
├── best_model.pth             # Best model by validation L2
├── final.pth                  # Final epoch model
├── reconstructed_gaussians/
│   ├── epoch_050/
│   │   ├── scene_000_epoch_050.ply
│   │   └── scene_001_epoch_050.ply
│   └── epoch_100/
│       └── ...
└── pca_visualizations/        # Only for semantic runs
    ├── epoch_050/
    │   ├── scene_000_semantic_pca.ply
    │   └── scene_000_position_pca.ply
    └── epoch_100/
        └── ...
```

### Logs
```
logs/
└── can3tok_<jobid>.out        # Training output with all metrics
```

## Code Structure
```
.
├── gs_can3tok_2.py                     # Main training script
├── gs_dataset_scenesplat.py            # Dataset loader
├── semantic_losses.py                  # Contrastive loss functions
├── gs_ply_reconstructor.py             # PLY file writer
├── pca_feature_visualization.py        # PCA coloring utilities
├── visualize_input_pca.py              # Input scene PCA visualization
├── job_can3tok.sh                      # SLURM job script
└── model/
    ├── configs/aligned_shape_latents/
    │   └── shapevae-256.yaml           # Model configuration
    └── michelangelo/models/tsal/
        ├── sal_perceiver_II_initialization.py  # Model architecture
        └── asl_pl_module.py            # PyTorch Lightning wrapper
```

---

## Three Approaches for Semantic Feature Extraction

### Approach 1: Geometric Parameter Projection

```python
Reconstructed Gaussians [B, 40k, 14]
    -> Per-Gaussian 3-layer MLP (14 -> 128 -> 128 -> 32)
    -> L2 normalization
    -> [B, 40k, 32] semantic features
```

Best for scale distribution improvement. Per-splat supervision directly
encourages semantically similar Gaussians to develop coherent representations.
PCA visualization shows clean clustering by structural class without
instance-level supervision.

### Approach 2: Hidden State Projection

```python
Decoder hidden state [B, 1024]
    -> 3-layer MLP (1024 -> 512 -> 256 -> 40k*32)
    -> Reshape to [B, 40k, 32]
    -> L2 normalization
```

Best overall L2 at epoch 200. Large projection head (329M parameters) learns
semantic structure quickly but at scene level, so scale distribution
improvement is limited compared to geometric mode.

### Approach 3: Cross-Attention Semantic Head

```python
Gaussian positions [B, 40k, 3] (queries)
    x Transformer tokens [B, 512, 384] (keys/values)
    -> Cross-attention
    -> Per-Gaussian features [B, 40k, 32]
```

Most expressive but computationally expensive. Requires batch size 64 for
fair comparison — contrastive learning with smaller batches has fewer
negative pairs per step, which directly weakens the semantic signal.

---

## Key Arguments
```bash
--semantic_mode {hidden,geometric,attention,none}  # Feature extraction method
--segment_loss_weight FLOAT   # Segment contrast weight (validated: 0.3)
--instance_loss_weight FLOAT  # Instance contrast weight
--semantic_temperature FLOAT  # Temperature for softmax (validated: 0.1)
--semantic_subsample INT      # Gaussians to sample for loss (validated: 10000)
--kl_weight FLOAT             # KL divergence weight (validated: 1e-6)
--scale_norm_mode {linear,log} # Scale storage mode (use linear)
--sampling_method {opacity,random,hybrid}  # Importance sampling strategy
--train_scenes INT            # Limit training scenes
--val_scenes INT              # Limit validation scenes
--eval_every INT              # Validation frequency in epochs
--use_wandb                   # Enable Weights & Biases logging
```