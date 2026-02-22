# Semantic-Aware Can3Tok: 3D Scene Generation with Semantic Scene Completion

## Implementation Differences from Original Can3Tok

### 1. Dataset Format and Normalization

**Can3Tok Original:**
- **Data Format:** Raw 3DGS PLY files
- **Scale Storage:** Log-space (stores `log(scale_meters)`)
- **Canonical Normalization:**
```python
  # Positions
  positions_norm = (positions - center) * scale_factor
  
  # Scales (log-space arithmetic)
  scales_norm = scales + log(scale_factor)
  # Because: log(scale_new) = log(scale_old × factor) 
  #                          = log(scale_old) + log(factor)
```

**Our Implementation:**
- **Data Format:** SceneSplat preprocessed .npy files
- **Scale Storage:** Linear-space (stores `scale_meters` after exp())
- **Canonical Normalization:**
```python
  # Positions (same as Can3Tok)
  positions_norm = (positions - center) * scale_factor
  
  # Scales (linear-space arithmetic)
  scales_norm = scales * scale_factor
  # Direct multiplication because scales are already in meters
```

**Location:** `normalize_to_canonical_sphere()` function in `gs_dataset_scenesplat.py`

### 2. Model Output
- **Our Implementation:** 14 parameters including RGB colors (xyz, rgb, opacity, scale, quaternion)


### 3. Decoder Architecture
- **Can3Tok:** Raw outputs without explicit activations
- **Our Implementation:** Added activation functions in final layer:
  - Colors: sigmoid → [0, 1]
  - Opacity: sigmoid → [0, 1]
  - Scales: softplus → (0, +∞)
  - Quaternions: L2 normalization → ||q|| = 1.0
- **Location:** `GS_decoder` class in `sal_perceiver_II_initialization.py` 

### 4. Encoder Structure
- **Fixed Issue:** Previously missing voxel center Fourier embeddings
- **Current:** Dual Fourier embeddings (voxel centers + actual xyz positions)
- **Location:** `CrossAttentionEncoder` in `sal_perceiver_II_initialization.py` 


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
  - Handles importance sampling (top 40k Gaussians by opacity)
  - Voxelization for positional encoding
  - Canonical sphere normalization

**Loss Functions:**
- `semantic_losses.py` - Contains:
  - `ScanNet72SemanticLoss` - InfoNCE contrastive loss for semantic learning
  - Handles segment-level

**PLY Reconstruction:**
- `gs_ply_reconstructor.py` - Converts model outputs to 3DGS PLY format
  - RGB to SH coefficient conversion: `f_dc = (RGB - 0.5) / C0`
  - Inverts activations: logit(opacity), log(scale)
  - Normalizes quaternions
  - 

**Visualization:**
- `visualize_input_scenes.py` - Converts input .npy scenes to PLY for comparison
  - Applies same canonical normalization as training
  

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
2. Sample top 40k Gaussians by opacity (deterministic)
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
   - voxel_centers → Fourier → coarse spatial encoding
   - xyz → Fourier → fine spatial encoding
3. Concatenate [xyz_fourier, voxel_fourier, gaussian_params]
4. Input projection → transformer width
5. Cross-attention: queries attend to Gaussian features
6. Self-attention: refine latent tokens
   Output: [B, 512, 384] latent tokens

VAE BOTTLENECK:
1. Split: shape_embed [B, 384] + latents [B, 511, 384]
2. Project to mean μ and log_var
3. Sample: z = μ + σ * ε
   Output: [B, 512, 32] compressed latents

DECODER:
1. Post-KL projection: [B, 512, 32] → [B, 512, 384]
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
Scaled by recon_scale (default 1) to balance gradients

KL DIVERGENCE:
L_KL = KL(q(z|x) || N(0,I)) weighted by kl_weight (default 1e-5)

SEMANTIC LOSS (if semantic_mode != 'none'):
1. For each scene, extract valid Gaussians with segment labels
2. Subsample 10k Gaussians for efficiency
3. Create category prototypes by averaging features per category
4. Compute InfoNCE loss with temperature=0.07
L_segment = CrossEntropy(similarity_matrix, target_categories)
Weighted by segment_loss_weight (0.1-10.0)

TOTAL LOSS:
L_total = L_recon + L_KL + segment_loss_weight * L_segment
```

### 4. Training Loop (gs_can3tok_2.py)
```
For each epoch:
  For each batch:
    1. Load batch of scenes with features and labels
    2. Forward pass through model
    3. Compute losses
    4. Backward pass and optimizer step
    5. Log metrics (L2 error, individual losses)
  
  Every 10 epochs:
    - Run validation (compute val loss and L2 error)
    - Save checkpoint
    - Reconstruct 4 scenes to PLY files
    - Track best model by validation L2 error
```

### 5. PLY Reconstruction (gs_ply_reconstructor.py)
```
Input: [B, 40k, 14] model predictions (post-activation)

For each scene:
1. Extract parameters:
   - positions [40k, 3]
   - colors [40k, 3] in [0,1]
   - opacity [40k, 1] in [0,1]
   - scales [40k, 3] in (0,∞)
   - quaternions [40k, 4] normalized

2. Convert to PLY format:
   - colors: f_dc = (RGB - 0.5) / C0  (SH coefficient conversion)
   - opacity: logit(opacity) = log(p / (1-p))  (inverse sigmoid)
   - scales: log(scale)  (inverse softplus/exp)
   - quaternions: already normalized

3. Write to .ply file with proper structure
4. View in supersplat.at 
```

---

## Current Training Configuration

### Default Hyperparameters
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
epochs = 200-2000
kl_weight = 1e-5

# Loss scaling
recon_scale = 1000.0
segment_loss_weight = 0.1-10.0
semantic_temperature = 0.07
semantic_subsample = 2000

# Dataset
sampling_method = 'opacity'  # Top 40k by opacity
canonical_normalization = True  # 10m radius sphere
```

### Semantic Modes
```bash
--semantic_mode none        # No semantic learning (pure VAE)
--semantic_mode hidden      # Hidden state projection (default)
--semantic_mode geometric   # Geometric parameter projection
--semantic_mode attention   # Cross-attention semantic head
```

---

## Running Training

### Basic Training (No Semantics)
```bash
python gs_can3tok_2.py \
    --batch_size 64 \
    --num_epochs 200 \
    --lr 1e-4 \
    --semantic_mode none
```

### Training with Semantic Learning
```bash
python gs_can3tok_2.py \
    --batch_size 64 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --semantic_mode hidden \
    --segment_loss_weight 0.1 \
    --use_wandb
```




### Check PLY Reconstruction
```bash
# Verify PLY writer works correctly
python gs_ply_reconstructor.py
# Should print: "All tests passed!"

# Compare input vs reconstructed
python visualize_input_scenes.py \
    --dataset-dir data/val_grid1.0cm_chunk8x8_stride6x6 \
    --output-dir input_scenes_ply \
    --num-scenes 5
```


## Evaluation Metrics

### Reconstruction Quality

1. **L2 Error** (primary metric)
```python
   L2_error = ||G_pred - G_true||_2 / sqrt(batch_size)
```


## Dataset: SceneSplat with ScanNet72 Labels

### Dataset Structure

Each scene directory contains:
```
scene_dir/
├── coord.npy          # [N, 3] Gaussian positions
├── color.npy          # [N, 3] RGB values
├── scale.npy          # [N, 3] Anisotropic scales (linear space)
├── quat.npy           # [N, 4] Rotation quaternions
├── opacity.npy        # [N] Opacity values [0,1]
├── segment.npy        # [N] ScanNet72 category labels (0-71)
└── instance.npy       # [N] Instance IDs (-1 = background)
```

### ScanNet72 Categories

- **Total categories:** 72 (indices 0-71)
- **Missing categories:** [13, 53, 61] (never appear in dataset, handled in loss)
- **Most frequent:** Wall (0), Floor (1), Ceiling (23), Cabinet (2), Window (8)

---

## Output Files

### Checkpoints
```
checkpoints/RGB_job_<jobid>_<semantic_mode>/
├── epoch_010.pth              # Model checkpoint
├── epoch_020.pth
├── best_model.pth             # Best model by validation L2
├── final.pth                  # Final epoch model
└── reconstructed_gaussians/
    ├── epoch_010/
    │   ├── scene_000_epoch_010.ply
    │   ├── scene_001_epoch_010.ply
    │   └── ...
    └── epoch_020/
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
├── gs_can3tok_2.py                    # Main training script
├── gs_dataset_scenesplat.py           # Dataset loader
├── semantic_losses.py                 # Contrastive loss functions
├── gs_ply_reconstructor.py            # PLY file writer (verified correct)
├── visualize_input_scenes.py          # Input scene PLY converter
├── job_can3tok.sh                     # SLURM job script
└── model/
    ├── configs/aligned_shape_latents/
    │   └── shapevae-256.yaml          # Model configuration
    └── michelangelo/models/tsal/
        ├── sal_perceiver_II_initialization.py           # Model architecture (main file)
        └── asl_pl_module.py           # PyTorch Lightning wrapper
```

---

## Three Approaches for Semantic Feature Extraction

### Approach 1: Hidden State Projection

**Architecture:**
```python
Decoder hidden state [B, 1024]
    ↓
3-layer MLP (1024 → 512 → 256 → 40k×32)
    ↓
Reshape to [B, 40k, 32]
    ↓
L2 normalization
```

**When to use:** Default semantic mode, extracts features from deepest learned representation

### Approach 2: Geometric Parameter Projection

**Architecture:**
```python
Reconstructed Gaussians [B, 40k, 14]
    ↓
Per-Gaussian 3-layer MLP (14 → 128 → 128 → 32)
    ↓
L2 normalization
```

**When to use:** When you want semantic features grounded in geometry

### Approach 3: Cross-Attention Semantic Head

**Architecture:**
```python
Gaussian positions [B, 40k, 3] (queries)
    ×
Transformer tokens [B, 512, 384] (keys/values)
    ↓
Cross-attention mechanism
    ↓
Per-Gaussian features [B, 40k, 32]
```

**When to use:** Most computationally expensive but most expressive

---

## Key Arguments
```bash
--semantic_mode {hidden,geometric,attention,none}  # Feature extraction method
--segment_loss_weight FLOAT                        # Segment contrast weight
--instance_loss_weight FLOAT                       # Instance contrast weight
--semantic_temperature FLOAT                       # Temperature for softmax
--semantic_subsample INT                           # Gaussians to sample for loss
--recon_scale FLOAT                                # Scale factor for recon loss
--sampling_method {opacity,random,hybrid}          # Importance sampling strategy
--train_scenes INT                                 # Limit training scenes
--val_scenes INT                                   # Limit validation scenes
--use_wandb                                        # Enable Weights & Biases logging
```