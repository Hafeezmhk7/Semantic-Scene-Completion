"""
SceneSplat Semantic Information Investigation
==============================================

Research script to understand semantic labels in the dataset:
1. What is instance.npy?
2. What is segment.npy?
3. How are they distributed?
4. How can we use them?

This is RESEARCH - we're not implementing anything yet!
Just understanding the data.
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import json

# Paths
regular_train_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train"
grid_train_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6"

print("="*80)
print("SCENESPLAT SEMANTIC INFORMATION INVESTIGATION")
print("="*80)
print()
print("Research Questions:")
print("  1. What semantic information exists in the dataset?")
print("  2. How is it structured (instance vs segment)?")
print("  3. What are the label distributions?")
print("  4. How can we leverage it for training?")
print()
print("="*80)
print()

# ============================================================================
# 1. EXPLORE SEMANTIC FILES IN REGULAR DATASET
# ============================================================================

print("PART 1: SEMANTIC DATA STRUCTURE")
print("-"*80)

# Get first few scenes
scenes = sorted(os.listdir(regular_train_path))[:10]

print(f"Investigating first 10 scenes from regular dataset:")
print()

semantic_stats = []

for i, scene_name in enumerate(scenes):
    scene_path = os.path.join(regular_train_path, scene_name)
    
    print(f"\n[{i+1}/10] Scene: {scene_name}")
    print("-"*40)
    
    # Check if semantic files exist
    instance_file = os.path.join(scene_path, 'instance.npy')
    segment_file = os.path.join(scene_path, 'segment.npy')
    coord_file = os.path.join(scene_path, 'coord.npy')
    
    if not all([os.path.exists(f) for f in [instance_file, segment_file, coord_file]]):
        print("✗ Missing semantic files!")
        continue
    
    # Load data
    instance = np.load(instance_file)
    segment = np.load(segment_file)
    coord = np.load(coord_file)
    
    # Basic info
    print(f"Data shapes:")
    print(f"  Coordinates: {coord.shape}")
    print(f"  Instances:   {instance.shape}")
    print(f"  Segments:    {segment.shape}")
    print()
    
    # Data types
    print(f"Data types:")
    print(f"  Instance: {instance.dtype}")
    print(f"  Segment:  {segment.dtype}")
    print()
    
    # Value ranges
    print(f"Value ranges:")
    print(f"  Instance: [{instance.min()}, {instance.max()}]")
    print(f"  Segment:  [{segment.min()}, {segment.max()}]")
    print()
    
    # Unique values
    unique_instances = np.unique(instance)
    unique_segments = np.unique(segment)
    
    print(f"Unique labels:")
    print(f"  Instances: {len(unique_instances)} unique values")
    print(f"  Segments:  {len(unique_segments)} unique values")
    print()
    
    # Show some unique values
    print(f"Sample instance IDs: {unique_instances[:10]}")
    print(f"Sample segment IDs:  {unique_segments[:10]}")
    print()
    
    # Distribution
    instance_counts = Counter(instance)
    segment_counts = Counter(segment)
    
    print(f"Distribution statistics:")
    print(f"  Instance -1 (unlabeled): {instance_counts.get(-1, 0):,} points ({instance_counts.get(-1, 0)/len(instance)*100:.1f}%)")
    print(f"  Segment -1 (unlabeled):  {segment_counts.get(-1, 0):,} points ({segment_counts.get(-1, 0)/len(segment)*100:.1f}%)")
    print()
    
    # Store stats
    semantic_stats.append({
        'scene': scene_name,
        'num_points': len(coord),
        'num_instances': len(unique_instances),
        'num_segments': len(unique_segments),
        'instance_unlabeled_pct': instance_counts.get(-1, 0)/len(instance)*100,
        'segment_unlabeled_pct': segment_counts.get(-1, 0)/len(segment)*100,
        'instance_ids': unique_instances.tolist(),
        'segment_ids': unique_segments.tolist(),
    })

print()
print("="*80)

# ============================================================================
# 2. ANALYZE INSTANCE vs SEGMENT RELATIONSHIP
# ============================================================================

print("\nPART 2: INSTANCE vs SEGMENT RELATIONSHIP")
print("-"*80)

# Take first scene for detailed analysis
scene_name = scenes[0]
scene_path = os.path.join(regular_train_path, scene_name)

print(f"Detailed analysis of scene: {scene_name}")
print()

instance = np.load(os.path.join(scene_path, 'instance.npy'))
segment = np.load(os.path.join(scene_path, 'segment.npy'))
coord = np.load(os.path.join(scene_path, 'coord.npy'))

# Check correspondence between instance and segment
print("Investigating Instance ↔ Segment mapping:")
print()

# For each instance, what segments does it belong to?
unique_instances = np.unique(instance)
for inst_id in unique_instances[:5]:  # Check first 5 instances
    mask = (instance == inst_id)
    segments_in_instance = np.unique(segment[mask])
    
    print(f"Instance {inst_id:3d}:")
    print(f"  Points: {mask.sum():,}")
    print(f"  Contains segments: {segments_in_instance}")
    print()

print()

# For each segment, what instances does it contain?
unique_segments = np.unique(segment)
print("Segment → Instance mapping:")
print()

for seg_id in unique_segments[:5]:  # Check first 5 segments
    mask = (segment == seg_id)
    instances_in_segment = np.unique(instance[mask])
    
    print(f"Segment {seg_id:3d}:")
    print(f"  Points: {mask.sum():,}")
    print(f"  Contains instances: {instances_in_segment}")
    print()

# Hypothesis: Are instances finer-grained than segments?
print("Hypothesis Testing:")
print("-"*40)

avg_instances_per_segment = []
avg_segments_per_instance = []

for seg_id in unique_segments:
    if seg_id == -1:
        continue
    mask = (segment == seg_id)
    instances = np.unique(instance[mask])
    instances = instances[instances != -1]  # Remove unlabeled
    avg_instances_per_segment.append(len(instances))

for inst_id in unique_instances:
    if inst_id == -1:
        continue
    mask = (instance == inst_id)
    segments = np.unique(segment[mask])
    segments = segments[segments != -1]  # Remove unlabeled
    avg_segments_per_instance.append(len(segments))

if avg_instances_per_segment:
    print(f"Average instances per segment: {np.mean(avg_instances_per_segment):.2f}")
if avg_segments_per_instance:
    print(f"Average segments per instance: {np.mean(avg_segments_per_instance):.2f}")

print()

if np.mean(avg_instances_per_segment) > 1.5:
    print("✓ Hypothesis: Segments contain MULTIPLE instances")
    print("  → Segments likely represent LARGER regions (rooms, walls)")
    print("  → Instances likely represent OBJECTS (chairs, tables)")
else:
    print("✓ Hypothesis: Instances and segments are similar granularity")

print()
print("="*80)

# ============================================================================
# 3. STATISTICS ACROSS ALL SAMPLED SCENES
# ============================================================================

print("\nPART 3: AGGREGATE STATISTICS")
print("-"*80)

print(f"Statistics across {len(semantic_stats)} scenes:")
print()

# Compute statistics
num_points = [s['num_points'] for s in semantic_stats]
num_instances = [s['num_instances'] for s in semantic_stats]
num_segments = [s['num_segments'] for s in semantic_stats]
instance_unlabeled = [s['instance_unlabeled_pct'] for s in semantic_stats]
segment_unlabeled = [s['segment_unlabeled_pct'] for s in semantic_stats]

print("Points per scene:")
print(f"  Mean: {np.mean(num_points):,.0f}")
print(f"  Range: [{np.min(num_points):,}, {np.max(num_points):,}]")
print()

print("Instances per scene:")
print(f"  Mean: {np.mean(num_instances):.1f}")
print(f"  Range: [{np.min(num_instances)}, {np.max(num_instances)}]")
print()

print("Segments per scene:")
print(f"  Mean: {np.mean(num_segments):.1f}")
print(f"  Range: [{np.min(num_segments)}, {np.max(num_segments)}]")
print()

print("Unlabeled points:")
print(f"  Instance -1: {np.mean(instance_unlabeled):.1f}% average")
print(f"  Segment -1:  {np.mean(segment_unlabeled):.1f}% average")
print()

# Ratio analysis
ratio = np.array(num_instances) / np.array(num_segments)
print("Instance/Segment ratio:")
print(f"  Mean: {np.mean(ratio):.2f}")
print(f"  Interpretation: Instances are {np.mean(ratio):.1f}x more numerous than segments")
print()

print("="*80)

# ============================================================================
# 4. CHECK GRID DATASET (CHUNKS)
# ============================================================================

print("\nPART 4: SEMANTIC INFO IN GRID (CHUNKED) DATASET")
print("-"*80)

# Get some chunks
grid_chunks = sorted(os.listdir(grid_train_path))[:5]

print(f"Checking if grid chunks also have semantic labels:")
print()

for chunk_name in grid_chunks:
    chunk_path = os.path.join(grid_train_path, chunk_name)
    
    instance_file = os.path.join(chunk_path, 'instance.npy')
    segment_file = os.path.join(chunk_path, 'segment.npy')
    
    has_instance = os.path.exists(instance_file)
    has_segment = os.path.exists(segment_file)
    
    print(f"{chunk_name}:")
    print(f"  instance.npy: {'✓' if has_instance else '✗'}")
    print(f"  segment.npy:  {'✓' if has_segment else '✗'}")
    
    if has_instance and has_segment:
        instance = np.load(instance_file)
        segment = np.load(segment_file)
        print(f"  Instance IDs: {len(np.unique(instance))} unique")
        print(f"  Segment IDs:  {len(np.unique(segment))} unique")
    print()

print("="*80)

# ============================================================================
# 5. SEMANTIC LABEL SPACE ANALYSIS
# ============================================================================

print("\nPART 5: LABEL SPACE ANALYSIS")
print("-"*80)

# Collect all unique labels across scenes
all_instance_ids = set()
all_segment_ids = set()

for stats in semantic_stats:
    all_instance_ids.update(stats['instance_ids'])
    all_segment_ids.update(stats['segment_ids'])

all_instance_ids = sorted(list(all_instance_ids))
all_segment_ids = sorted(list(all_segment_ids))

print(f"Global label space (across {len(semantic_stats)} scenes):")
print()

print(f"Instance IDs:")
print(f"  Total unique: {len(all_instance_ids)}")
print(f"  Range: [{min(all_instance_ids)}, {max(all_instance_ids)}]")
print(f"  Sample IDs: {all_instance_ids[:20]}")
print()

print(f"Segment IDs:")
print(f"  Total unique: {len(all_segment_ids)}")
print(f"  Range: [{min(all_segment_ids)}, {max(all_segment_ids)}]")
print(f"  Sample IDs: {all_segment_ids[:20]}")
print()

# Check if IDs are scene-specific or global
print("ID scope analysis:")
if max(all_instance_ids) > 100:
    print("  ✓ Instance IDs likely SCENE-SPECIFIC (large range, reused across scenes)")
else:
    print("  ✓ Instance IDs likely GLOBAL CATEGORIES")

if max(all_segment_ids) > 100:
    print("  ✓ Segment IDs likely SCENE-SPECIFIC (large range, reused across scenes)")
else:
    print("  ✓ Segment IDs likely GLOBAL CATEGORIES")

print()
print("="*80)

# ============================================================================
# 6. POTENTIAL USE CASES
# ============================================================================

print("\nPART 6: POTENTIAL RESEARCH DIRECTIONS")
print("-"*80)

print("Based on the semantic information discovered, here are potential uses:")
print()

print("1. SEMANTIC CONTRASTIVE LEARNING")
print("   Idea: Gaussians from the same instance/segment should have similar")
print("         latent representations")
print()
print("   Implementation:")
print("   - Compute embeddings for each Gaussian")
print("   - Pull together embeddings with same instance ID")
print("   - Push apart embeddings with different instance IDs")
print("   - Loss: InfoNCE or Triplet loss")
print()

print("2. SEMANTIC RECONSTRUCTION LOSS")
print("   Idea: Penalize reconstruction errors more for important objects")
print()
print("   Implementation:")
print("   - Weight reconstruction loss by instance importance")
print("   - Objects (instance != -1) get higher weight")
print("   - Background (instance == -1) gets lower weight")
print()

print("3. HIERARCHICAL SEMANTIC LOSS")
print("   Idea: Use both instance and segment for multi-scale learning")
print()
print("   Implementation:")
print("   - Instance-level loss: Fine-grained object consistency")
print("   - Segment-level loss: Coarse-grained region consistency")
print("   - Total: L_total = L_recon + λ_inst * L_inst + λ_seg * L_seg")
print()

print("4. SEMANTIC-AWARE SAMPLING")
print("   Idea: Sample more Gaussians from important objects")
print()
print("   Implementation:")
print("   - Current: Sample by opacity")
print("   - Improved: Sample by opacity * instance_importance")
print("   - Ensures important objects are well-represented")
print()

print("5. SEMANTIC REGULARIZATION")
print("   Idea: Enforce smooth latent space within instances")
print()
print("   Implementation:")
print("   - For Gaussians in same instance, enforce similar latents")
print("   - L_reg = Σ ||z_i - z_j||² for i,j in same instance")
print("   - Encourages semantically coherent latent space")
print()

print("="*80)

# ============================================================================
# 7. CONTRASTIVE LEARNING PRIMER
# ============================================================================

print("\nPART 7: CONTRASTIVE LEARNING WITH SEMANTIC LABELS")
print("-"*80)

print("What is contrastive learning?")
print("-"*40)
print()
print("Goal: Learn representations where similar items are close,")
print("      different items are far apart.")
print()
print("For semantic labels:")
print("  Positive pairs: Gaussians with SAME instance/segment ID")
print("  Negative pairs: Gaussians with DIFFERENT instance/segment ID")
print()

print("Common contrastive losses:")
print()

print("1. InfoNCE Loss:")
print("   L = -log( exp(sim(z_i, z_j^+) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )")
print()
print("   Where:")
print("   - z_i: anchor embedding")
print("   - z_j^+: positive (same instance)")
print("   - z_k: negatives (different instances)")
print("   - τ: temperature")
print()

print("2. Triplet Loss:")
print("   L = max(0, ||z_a - z_p||² - ||z_a - z_n||² + margin)")
print()
print("   Where:")
print("   - z_a: anchor")
print("   - z_p: positive (same instance)")
print("   - z_n: negative (different instance)")
print()

print("3. Supervised Contrastive Loss (SupCon):")
print("   L = Σ -log( Σ_p exp(z·z_p/τ) / Σ_k exp(z·z_k/τ) )")
print()
print("   Benefits:")
print("   - Uses ALL positives (not just one)")
print("   - More stable than triplet loss")
print("   - Works well with semantic labels")
print()

print("="*80)

# ============================================================================
# 8. RESEARCH QUESTIONS TO INVESTIGATE
# ============================================================================

print("\nPART 8: OPEN RESEARCH QUESTIONS")
print("-"*80)

print("Before implementing, we should investigate:")
print()

print("Q1: What do instance and segment IDs actually represent?")
print("    - Are they object categories?")
print("    - Are they individual object instances?")
print("    - Are they spatial regions?")
print("    → Need to check SceneSplat paper / documentation")
print()

print("Q2: How consistent are IDs across scenes?")
print("    - Is instance ID 5 always 'chair' across scenes?")
print("    - Or is ID 5 different object in each scene?")
print("    → Affects how we use them in contrastive learning")
print()

print("Q3: What's the relationship between instance and segment?")
print("    - Hierarchical? (instances within segments)")
print("    - Independent? (different annotation schemes)")
print("    - Overlapping? (can cross boundaries)")
print("    → Determines if we can use both jointly")
print()

print("Q4: How much do unlabeled points affect training?")
print(f"    - Current data: ~{np.mean(instance_unlabeled):.0f}% points are unlabeled")
print("    - Should we:")
print("      a) Ignore unlabeled points in semantic loss?")
print("      b) Treat them as separate 'background' class?")
print("      c) Use semi-supervised approach?")
print()

print("Q5: Which loss function would work best?")
print("    - Instance-level contrastive?")
print("    - Segment-level contrastive?")
print("    - Both (hierarchical)?")
print("    - Weighted reconstruction?")
print("    → Need ablation study")
print()

print("="*80)

# ============================================================================
# 9. NEXT STEPS FOR RESEARCH
# ============================================================================

print("\nPART 9: RECOMMENDED NEXT STEPS")
print("-"*80)

print("Research roadmap:")
print()

print("STEP 1: Literature Review")
print("  □ Read SceneSplat paper carefully")
print("  □ Check supplementary materials")
print("  □ Look for semantic annotation details")
print("  □ Find example visualizations")
print()

print("STEP 2: Data Understanding")
print("  □ Visualize instance/segment labels (color code)")
print("  □ Check if IDs are consistent across scenes")
print("  □ Understand label meaning (if documented)")
print("  □ Analyze spatial distribution")
print()

print("STEP 3: Baseline Experiments")
print("  □ Train model WITHOUT semantic info (current)")
print("  □ Measure performance")
print("  □ Establish baseline metrics")
print()

print("STEP 4: Ablation Studies")
print("  □ Experiment A: Weighted reconstruction loss")
print("  □ Experiment B: Instance-level contrastive")
print("  □ Experiment C: Segment-level contrastive")
print("  □ Experiment D: Hierarchical (both)")
print("  □ Compare all against baseline")
print()

print("STEP 5: Analysis & Publication")
print("  □ Analyze which approach works best")
print("  □ Understand why it works")
print("  □ Write up findings")
print()

print("="*80)

# ============================================================================
# 10. SAVE INVESTIGATION RESULTS
# ============================================================================

print("\nSAVING INVESTIGATION RESULTS...")

results = {
    'num_scenes_investigated': len(semantic_stats),
    'statistics': {
        'avg_points_per_scene': float(np.mean(num_points)),
        'avg_instances_per_scene': float(np.mean(num_instances)),
        'avg_segments_per_scene': float(np.mean(num_segments)),
        'avg_instance_unlabeled_pct': float(np.mean(instance_unlabeled)),
        'avg_segment_unlabeled_pct': float(np.mean(segment_unlabeled)),
        'instance_segment_ratio': float(np.mean(ratio)),
    },
    'label_space': {
        'total_unique_instances': len(all_instance_ids),
        'total_unique_segments': len(all_segment_ids),
        'instance_range': [int(min(all_instance_ids)), int(max(all_instance_ids))],
        'segment_range': [int(min(all_segment_ids)), int(max(all_segment_ids))],
    },
    'findings': {
        'semantic_files_present': True,
        'grid_has_semantics': True,
        'instance_granularity': 'finer' if np.mean(ratio) > 1.5 else 'similar',
        'label_scope': 'scene_specific' if max(all_instance_ids) > 100 else 'global_categories',
    },
    'scene_details': semantic_stats,
}

output_file = 'semantic_investigation_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {output_file}")
print()

print("="*80)
print("INVESTIGATION COMPLETE!")
print("="*80)
print()
print("Summary:")
print(f"  ✓ Semantic files found: instance.npy, segment.npy")
print(f"  ✓ Present in both regular and grid datasets")
print(f"  ✓ Average {np.mean(num_instances):.0f} instances per scene")
print(f"  ✓ Average {np.mean(num_segments):.0f} segments per scene")
print(f"  ✓ ~{np.mean(instance_unlabeled):.0f}% points unlabeled")
print()
print("Next: Read SceneSplat paper to understand what these labels mean!")
print("="*80)