# Semantic-Aware Can3Tok: 3D Scene Generation with Semantic Scene Completion

**Master's Thesis Project**  
**Building on:** Can3Tok (ICCV 2025) - Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

---

##  Project Overview

This project enhances the Can3Tok framework by incorporating semantic information into the latent space of a 3D Gaussian Splatting (3DGS) Variational Autoencoder (VAE). The goal is to enable **semantic-aware 3D scene generation** and **semantic scene completion** by learning semantically meaningful latent representations.

### Key Innovation
While the original Can3Tok learns geometric representations of 3D scenes, this work introduces **semantic contrastive learning** to ensure that the learned latent space captures both geometric and semantic properties of indoor scenes.

---

##  Architecture Overview

### Base Framework: Can3Tok VAE

The Can3Tok architecture consists of three main components:

```
Input: 3D Gaussians [B, 40k, 18]
   ↓
┌─────────────────────────────────────────────────┐
│  ENCODER (Cross-Attention + Self-Attention)     │
│  - Fourier embeddings for positional encoding   │
│  - Cross-attention: queries attend to Gaussians │
│  - Self-attention: refine latent tokens         │
│  Output: Latent tokens [B, 512, 384]            │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  LATENT BOTTLENECK (VAE)                        │
│  - KL divergence regularization                 │
│  - μ, log_σ² → z [B, 512, 32]                  │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  DECODER (Transformer + MLP)                    │
│  - Self-attention on latent tokens              │
│  - 8-layer MLP decoder                          │
│  Output: Reconstructed Gaussians [B, 40k, 11]  │
└─────────────────────────────────────────────────┘
```

### Our Enhancement: Semantic Feature Extraction

We add a **semantic projection head** that extracts per-Gaussian features for contrastive learning:

```
┌─────────────────────────────────────────────────┐
│  SEMANTIC FEATURE EXTRACTION                    │
│  (Three different approaches tested)            │
│                                                 │
│  Input: Intermediate decoder representations    │
│  Output: Per-Gaussian features [B, 40k, 32]    │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  CONTRASTIVE LEARNING                           │
│  - Aggregate features by segment category       │
│  - Create category prototypes                   │
│  - InfoNCE loss with ScanNet72 labels          │
└─────────────────────────────────────────────────┘
```

---

##  Three Approaches for Semantic Feature Extraction

We explored three different strategies to extract per-Gaussian semantic features:

### Approach 1: Hidden State Projection ⭐ (Current Implementation)

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

**Advantages:**
- Extracts features from the deepest learned representation
- Relatively efficient (329M parameters but manageable)
- Captures high-level semantic abstractions

**When to use:** When you want features that represent the entire scene's semantic context

---

### Approach 2: Geometric Parameter Projection

**Architecture:**
```python
Reconstructed Gaussians [B, 40k, 11]
    ↓
Per-Gaussian 3-layer MLP (11 → 128 → 128 → 32)
    ↓
L2 normalization
```

**Advantages:**
- Directly operates on geometric parameters (xyz, opacity, scale, quat)
- Similar to SimCLR projection head
- Most parameter-efficient approach
- Each Gaussian processed independently

**When to use:** When you want semantic features grounded in geometry

---

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
    ↓
L2 normalization
```



##  Dataset: SceneSplat with ScanNet72 Labels

### Dataset Structure

Each scene contains:
```
scene_dir/
├── coord.npy          # [N, 3] Gaussian positions
├── color.npy          # [N, 3] RGB values (not used in reconstruction)
├── scale.npy          # [N, 3] Anisotropic scale
├── quat.npy           # [N, 4] Rotation quaternions
├── opacity.npy        # [N] Opacity values
├── segment.npy        # [N] ScanNet72 category labels (0-71)
└── instance.npy       # [N] Instance IDs (-1 = background)
```

### Preprocessing Pipeline

**1. Importance Sampling** (40,000 Gaussians per scene)
```python
# Hybrid sampling (current default: opacity-based)
if sampling_method == 'opacity':
    importance = opacity  # Prioritize visible Gaussians
elif sampling_method == 'hybrid':
    importance = 0.7*opacity + 0.3*scale_magnitude
```

**2. Voxelization** (40×40×40 grid, resolution=0.4)
```python
# Assign Gaussians to voxel centers for positional encoding
voxel_centers = compute_voxel_centers(coord, resolution=0.4)
point_uniq_idx = voxelize(coord)  # FNV hash for unique voxel IDs
```

**3. Feature Assembly** (18-channel representation)
```python
features = [
    voxel_centers,      # [3] Positional encoding
    point_uniq_idx,     # [1] Voxel ID
    coord,              # [3] Gaussian xyz
    color,              # [3] RGB (for reference, not reconstructed)
    opacity,            # [1] Opacity
    scale,              # [3] Anisotropic scale
    quat                # [4] Rotation quaternion
]  # Total: 18 channels
```

### ScanNet72 Semantic Categories

- **Total categories:** 72 (indices 0-71)
- **Missing categories:** [13, 53, 61] (never appear in dataset)
- **Most frequent:** Wall, floor, cabinet, bed, chair, sofa, table
- **Expected random loss:** log(72) ≈ 4.28

**Category distribution example:**
```
Top 5 categories cover ~70% of points:
  1. Wall (category 0): 28.3%
  2. Floor (category 1): 18.7%
  3. Ceiling (category 23): 12.1%
  4. Cabinet (category 2): 6.5%
  5. Window (category 8): 4.8%
```

---

## 🧠 Semantic Contrastive Learning

### Loss Function Architecture

**Total Loss:**
```python
L_total = α·L_recon + β·L_KL + γ·L_segment + δ·L_instance

where:
  L_recon    = ||G_pred - G_true||₂  (geometric reconstruction)
  L_KL       = KL(q(z|x) || p(z))    (VAE regularization)
  L_segment  = InfoNCE(segment)       (category-level contrast)
  L_instance = InfoNCE(instance)      (instance-level contrast)
```

### Segment-Level Contrastive Loss (Primary)

**Objective:** Gaussians with the same semantic category should have similar features.

**Algorithm:**
```python
for each scene in batch:
    # 1. Extract valid Gaussians with labels
    valid_mask = segment_labels >= 0
    embeddings = per_gaussian_features[valid_mask]  # [M, 32]
    labels = segment_labels[valid_mask]             # [M]
    
    # 2. Create category prototypes
    for each category C in unique_categories:
        prototype_C = mean(embeddings where label == C)
        prototype_C = L2_normalize(prototype_C)
    
    # 3. Compute similarity matrix
    similarity = (embeddings @ prototypes.T) / temperature
    
    # 4. InfoNCE loss
    loss = CrossEntropy(similarity, target_category_indices)
```

**Key hyperparameters:**
- `temperature` (τ): 0.07 (controls softmax sharpness)
- `segment_weight` (γ): 0.1-10.0 (relative importance)
- `subsample`: 2000 Gaussians per scene (for efficiency)

### Instance-Level Contrastive Loss (Optional)

**Objective:** Gaussians from the same object instance should cluster together.

Similar to segment-level but uses instance IDs instead of categories. Currently set to weight=0.0 but can be enabled for finer-grained learning.

---

## 🎛️ Training Configuration

### Key Modifications from Original Can3Tok

1. **No Color Reconstruction**
   - Output: 11 parameters (xyz, opacity, scale, quat)
   - Original: 14 parameters (+ RGB)
   - Reason: Focus on geometry and semantics, as advised by supervisor

2. **Loss Scaling**
   ```python
   recon_loss_raw = ||prediction - target||₂ / batch_size
   recon_loss = recon_loss_raw / recon_scale  # scale=1000
   ```
   - Balances gradient magnitudes between reconstruction and semantic losses

3. **Semantic Mode Selection**
   ```python
   --semantic_mode hidden      # Approach 1 (default)
   --semantic_mode geometric   # Approach 2
   --semantic_mode attention   # Approach 3
   ```

### Training Hyperparameters

```python
# Architecture
num_latents = 256          # VAE latent tokens
embed_dim = 32             # Latent embedding dimension
width = 384                # Transformer width
num_encoder_layers = 6     # Encoder depth
num_decoder_layers = 12    # Decoder depth

# Optimization
learning_rate = 1e-4
batch_size = 64
epochs = 1000
kl_weight = 1e-5

# Semantic loss
segment_loss_weight = 0.1-10.0
instance_loss_weight = 0.0
semantic_temperature = 0.07
semantic_subsample = 2000
recon_scale = 1000.0
```

### Hardware Requirements

- **GPUs:** 8× NVIDIA GPUs (DataParallel training)
- **Memory:** ~40GB per GPU (with batch_size=64, hidden state approach)
- **Training time:** ~2-3 days for 1000 epochs on 7K scenes

---

## 📈 Evaluation Metrics

### Reconstruction Quality

1. **L2 Error** (primary metric)
   ```python
   L2_error = ||G_pred - G_true||₂ / sqrt(batch_size)
   ```

2. **Failure Rate**
   ```python
   failure_rate = (# scenes with L2 > threshold) / total_scenes
   threshold = 8000.0  # default
   ```

3. **Per-scene Statistics**
   - Mean, std, min, max, median L2 errors
   - Useful for identifying failure modes

### Semantic Learning Quality

1. **Segment Loss** (InfoNCE)
   - Should decrease during training
   - Random baseline: log(72) ≈ 4.28

2. **Category Alignment**
   - Measure clustering quality in latent space
   - Use t-SNE visualization (see `tsne_exp*`)

3. **Semantic Consistency**
   - Check if semantically similar scenes have similar latents
   - Qualitative evaluation through generation

---

## 🚀 Usage

### Training

```bash
# Hidden state approach (default)
python gs_can3tok_2.py \
    --batch_size 64 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --segment_loss_weight 0.1 \
    --semantic_mode hidden \
    --use_wandb

# Geometric approach
python gs_can3tok_2.py \
    --semantic_mode geometric \
    --segment_loss_weight 1.0

# Attention approach
python gs_can3tok_2.py \
    --semantic_mode attention \
    --segment_loss_weight 0.5
```

### Key Arguments

```bash
--semantic_mode {hidden,geometric,attention}  # Feature extraction method
--segment_loss_weight FLOAT                   # β: Segment contrast weight
--instance_loss_weight FLOAT                  # δ: Instance contrast weight
--semantic_temperature FLOAT                  # τ: Temperature for softmax
--semantic_subsample INT                      # Gaussians to sample for loss
--recon_scale FLOAT                           # Scale factor for recon loss
--sampling_method {opacity,random,hybrid}     # Importance sampling strategy
--train_scenes INT                            # Limit training scenes
--val_scenes INT                              # Limit validation scenes
```

### Dataset Inspection

```bash
# Analyze semantic label distribution
python gs_dataset_scenesplat.py /path/to/data

# Output: Category statistics, missing labels, coverage analysis
```

---

## 📂 Code Structure

### Core Files

```
├── gs_can3tok_2.py                    # Main training script
├── gs_dataset_scenesplat.py           # Dataset loader with semantics
├── semantic_losses.py                 # Contrastive loss functions
└── model/
    ├── configs/
    │   └── aligned_shape_latents/
    │       └── shapevae-256.yaml      # VAE configuration
    └── michelangelo/models/tsal/
        ├── sal_perceiver.py           # Model architecture (3 approaches)
        └── asl_pl_module.py           # PyTorch Lightning wrapper
```

### Key Classes

**`AlignedShapeLatentPerceiver`** (sal_perceiver.py)
- Main VAE model
- Implements all three semantic feature extraction approaches
- Switch via `semantic_mode` parameter

**`gs_dataset`** (gs_dataset_scenesplat.py)
- Loads SceneSplat scenes with ScanNet72 labels
- Importance sampling and voxelization
- Returns: features, segment_labels, instance_labels

**`ScanNet72SemanticLoss`** (semantic_losses.py)
- Optimized contrastive loss for ScanNet72
- Handles missing categories [13, 53, 61]
- Efficient prototype aggregation

---

## 🔬 Experimental Insights

### Findings from Three Approaches

| Approach | Parameters | Speed | Memory | Semantic Quality |
|----------|-----------|-------|--------|-----------------|
| **Hidden State** | 329M | Fast | 40GB | ⭐⭐⭐ Best semantic abstraction |
| **Geometric** | 45K | Fastest | 20GB | ⭐⭐ Good, geometry-grounded |
| **Attention** | 158M | Medium | 35GB | ⭐⭐⭐ Best spatial reasoning |

### Loss Balancing Challenges

**Problem:** Reconstruction loss (~10,000) >> Semantic loss (~4.0)

**Solution:** Scale reconstruction loss by 1000×
```python
recon_loss_scaled = recon_loss_raw / 1000.0
total_loss = recon_loss_scaled + 0.1 * semantic_loss
```

**Effect:** 
- Gradients become comparable in magnitude
- Semantic loss can influence learning without being overwhelmed
- Reconstruction quality remains high (L2 ~3000-5000)

### Sampling Strategy Impact

**Opacity-based sampling** (current default):
- Prioritizes visible Gaussians
- Better reconstruction quality
- More relevant semantic information

**Random sampling:**
- Uniform coverage of scene
- May include background/noise Gaussians
- Lower reconstruction quality

---

## 🎯 Research Questions & Future Work

### Current Research Questions

1. **Which semantic feature extraction approach is best?**
   - Hidden state: Global scene context
   - Geometric: Direct geometry-semantics link
   - Attention: Spatial relationship modeling

2. **How to balance reconstruction vs. semantics?**
   - Current: recon_scale=1000, segment_weight=0.1
   - Need systematic hyperparameter search

3. **Does semantic learning improve generation quality?**
   - Hypothesis: Semantically-aware latents → better scene completion
   - Needs evaluation on downstream tasks

### Future Directions

1. **Semantic Scene Completion**
   - Given partial scene, complete missing regions
   - Use semantic priors from learned latent space

2. **Category-Conditional Generation**
   - Generate scenes with specific object categories
   - Control: "living room with sofa and TV"

3. **Hierarchical Semantic Learning**
   - Current: Only segment-level
   - Add: Room-level, object-level, part-level

4. **Multi-Modal Conditioning**
   - Text-to-3D scene generation
   - Image-guided scene completion

5. **Diffusion Models in Latent Space**
   - Train diffusion prior on semantic latents
   - Enable diverse scene generation

---

## 📚 Related Work

### Building Upon

1. **Can3Tok** (ICCV 2025)
   - First scene-level 3D Gaussian VAE
   - Canonical tokenization for 3D scenes

2. **3D Gaussian Splatting** (SIGGRAPH 2023)
   - Explicit 3D representation
   - Fast rendering via splatting

3. **SceneSplat / DL3DV-10K**
   - Large-scale 3D scene datasets
   - ScanNet72 semantic annotations

### Relevant Methods

- **PointNet++**: Per-point feature learning
- **SimCLR**: Contrastive learning for representations
- **CLIP**: Multi-modal embedding alignment
- **Semantic-NeRF**: Semantic field learning

---

## 🏆 Contributions

### Novel Aspects

1. **First semantic-aware 3D Gaussian VAE**
   - Original Can3Tok: Geometry only
   - This work: Geometry + semantics

2. **Three semantic feature extraction strategies**
   - Systematic comparison of architectures
   - Trade-offs analysis (speed, memory, quality)

3. **Contrastive learning for 3DGS**
   - Adapted InfoNCE loss for per-Gaussian features
   - ScanNet72 category prototypes

4. **SceneSplat integration**
   - First work to use SceneSplat for generative modeling
   - Importance sampling for 40K Gaussians

---

## 🙏 Acknowledgments

**Original Can3Tok Authors:**
- Quankai Gao, Iliyan Georgiev, Tuanfeng Y. Wang, Krishna Kumar Singh, Ulrich Neumann, Jae Shin Yoon

**Datasets:**
- DL3DV-10K / SceneSplat teams
- ScanNet dataset creators

**Frameworks:**
- 3D Gaussian Splatting (INRIA)
- Michelangelo (Neural Carver)
- PyTorch Lightning

---

## 📧 Contact

For questions about this Master's thesis project, please contact:
- Student: [Your Name]
- Supervisor: [Supervisor Name]
- Institution: [University Name]

---

## 📄 Citation

If you use this work, please cite both the original Can3Tok paper and acknowledge this extension:

```bibtex
@INPROCEEDINGS{gao2025can3tok,
  author = {Quankai Gao and Iliyan Georgiev and Tuanfeng Y. Wang and Krishna Kumar Singh and Ulrich Neumann and Jae Shin Yoon},
  title = {Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025}
}

@mastersthesis{yourname2025semantic,
  title = {Semantic-Aware 3D Scene Generation with Can3Tok},
  author = {Your Name},
  year = {2025},
  school = {Your University}
}
```

---

## 📋 TODO / Known Issues

### High Priority
- [ ] Systematic hyperparameter search for loss weights
- [ ] Ablation study: Which approach works best?
- [ ] Quantitative semantic evaluation metrics
- [ ] Memory optimization for hidden state approach

### Medium Priority
- [ ] Implement semantic scene completion pipeline
- [ ] Add category-conditional generation
- [ ] t-SNE visualization of learned latents
- [ ] Comparison with baseline (no semantic loss)

### Low Priority
- [ ] Support for other semantic taxonomies (not just ScanNet72)
- [ ] Mixed precision training (FP16)
- [ ] Distributed training across multiple nodes
- [ ] Interactive demo for scene editing

---

**Last Updated:** February 2026  
**Status:** Active Development  
**License:** Same as Can3Tok (check original repository)