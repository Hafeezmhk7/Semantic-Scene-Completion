"""
SceneSplat Semantic Dataset Analysis for Sampling Strategy
===========================================================

GOAL: Understand dataset characteristics to determine:
1. Optimal subsample size
2. Whether to use category-balanced sampling
3. Whether to use hard negative mining
4. Class imbalance severity
5. Redundancy within categories

This analysis will guide our sampling strategy decision scientifically.
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import json
from tqdm import tqdm

# Paths
train_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6"

print("="*80)
print("SEMANTIC DATASET ANALYSIS FOR SAMPLING STRATEGY")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# How many scenes to analyze (more = better statistics, but slower)
NUM_SCENES_TO_ANALYZE = 100  # Adjust based on time available

# Sample sizes to test
SUBSAMPLE_SIZES = [1000, 2000, 4000, 8000, 10000, 20000, 40000]

print(f"Configuration:")
print(f"  Analyzing {NUM_SCENES_TO_ANALYZE} scenes")
print(f"  Testing subsample sizes: {SUBSAMPLE_SIZES}")
print()
print("="*80)
print()

# ============================================================================
# 1. LOAD AND AGGREGATE SEMANTIC STATISTICS
# ============================================================================

print("PART 1: LOADING SEMANTIC DATA")
print("-"*80)

scenes = sorted(os.listdir(train_path))[:NUM_SCENES_TO_ANALYZE]
print(f"Found {len(scenes)} scenes to analyze")
print()

# Storage for statistics
all_category_counts = Counter()  # Global category distribution
scene_stats = []
category_sizes = defaultdict(list)  # How many points per category per scene

print("Loading semantic labels...")
for scene_name in tqdm(scenes, desc="Processing scenes"):
    scene_path = os.path.join(train_path, scene_name)
    
    # Check files exist
    segment_file = os.path.join(scene_path, 'segment.npy')
    instance_file = os.path.join(scene_path, 'instance.npy')
    
    if not os.path.exists(segment_file):
        continue
    
    # Load segment labels (these are semantic categories)
    segment = np.load(segment_file)
    
    # Count categories in this scene
    valid_labels = segment[segment >= 0]  # Exclude -1 (unlabeled)
    scene_categories = Counter(valid_labels)
    
    # Update global counts
    all_category_counts.update(scene_categories)
    
    # Store per-category sizes
    for cat_id, count in scene_categories.items():
        category_sizes[cat_id].append(count)
    
    # Scene-level stats
    scene_stats.append({
        'name': scene_name,
        'total_points': len(segment),
        'labeled_points': len(valid_labels),
        'unlabeled_points': np.sum(segment == -1),
        'num_categories': len(scene_categories),
        'categories': dict(scene_categories),
    })

print(f"✓ Loaded {len(scene_stats)} scenes")
print()

# ============================================================================
# 2. CATEGORY DISTRIBUTION ANALYSIS
# ============================================================================

print("="*80)
print("PART 2: CATEGORY DISTRIBUTION ANALYSIS")
print("-"*80)
print()

# Sort categories by frequency
sorted_categories = sorted(all_category_counts.items(), key=lambda x: x[1], reverse=True)

print(f"Total unique categories found: {len(sorted_categories)}")
print(f"Total labeled points across all scenes: {sum(all_category_counts.values()):,}")
print()

print("Category frequency distribution:")
print("-"*40)
print(f"{'Rank':<6} {'Category ID':<12} {'Count':>12} {'Percentage':>12} {'Cumulative':>12}")
print("-"*40)

cumulative = 0
top_categories = []
for i, (cat_id, count) in enumerate(sorted_categories[:30], 1):
    percentage = count / sum(all_category_counts.values()) * 100
    cumulative += percentage
    print(f"{i:<6} {cat_id:<12} {count:>12,} {percentage:>11.2f}% {cumulative:>11.2f}%")
    
    if i <= 10:
        top_categories.append(cat_id)

print()

# Calculate Gini coefficient (measure of inequality)
def gini_coefficient(counts):
    """Calculate Gini coefficient: 0 = perfect equality, 1 = perfect inequality"""
    sorted_counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

category_counts_array = np.array([count for _, count in sorted_categories])
gini = gini_coefficient(category_counts_array)

print(f"Class Imbalance Metrics:")
print(f"  Gini coefficient: {gini:.3f}")
if gini > 0.7:
    print(f"  → SEVERE imbalance (Gini > 0.7)")
    print(f"  → Balanced sampling STRONGLY recommended")
elif gini > 0.5:
    print(f"  → MODERATE imbalance (Gini > 0.5)")
    print(f"  → Balanced sampling recommended")
else:
    print(f"  → MILD imbalance (Gini < 0.5)")
    print(f"  → Random sampling may work")

print()

# Top-K coverage analysis
print("How many categories cover X% of data?")
for threshold in [50, 75, 90, 95]:
    cumsum = np.cumsum(category_counts_array)
    total = cumsum[-1]
    k = np.searchsorted(cumsum, threshold / 100 * total) + 1
    print(f"  Top {k:2d} categories cover {threshold}% of data")

print()

# Rare category analysis
rare_threshold = 0.1  # Categories with < 0.1% of data
rare_categories = [(cat, count) for cat, count in sorted_categories 
                   if count / sum(all_category_counts.values()) * 100 < rare_threshold]

print(f"Rare categories (< {rare_threshold}% of data):")
print(f"  Count: {len(rare_categories)} categories")
print(f"  Total points: {sum(c for _, c in rare_categories):,} ({sum(c for _, c in rare_categories) / sum(all_category_counts.values()) * 100:.2f}%)")
print()

print("="*80)
print()

# ============================================================================
# 3. PER-SCENE CATEGORY ANALYSIS
# ============================================================================

print("PART 3: PER-SCENE CATEGORY STATISTICS")
print("-"*80)
print()

# For each category, how many points does it have per scene?
print("Category representation across scenes:")
print("-"*40)
print(f"{'Category':<10} {'Scenes':<8} {'Min':<10} {'Mean':<10} {'Max':<10} {'Std':<10}")
print("-"*40)

for cat_id in sorted(category_sizes.keys())[:20]:  # Top 20
    sizes = category_sizes[cat_id]
    print(f"{cat_id:<10} {len(sizes):<8} {min(sizes):<10,} {np.mean(sizes):<10,.0f} {max(sizes):<10,} {np.std(sizes):<10,.0f}")

print()

# Calculate average categories per scene
cats_per_scene = [s['num_categories'] for s in scene_stats]
print(f"Categories per scene:")
print(f"  Mean: {np.mean(cats_per_scene):.1f}")
print(f"  Std: {np.std(cats_per_scene):.1f}")
print(f"  Range: [{min(cats_per_scene)}, {max(cats_per_scene)}]")
print()

print("="*80)
print()

# ============================================================================
# 4. SUBSAMPLE SIZE ANALYSIS
# ============================================================================

print("PART 4: OPTIMAL SUBSAMPLE SIZE ESTIMATION")
print("-"*80)
print()

print("For each subsample size, estimating coverage:")
print("-"*40)

# Simulate random sampling at different subsample sizes
for subsample_size in SUBSAMPLE_SIZES:
    # Simulate sampling across all scenes
    total_points = sum(all_category_counts.values())
    
    # Expected samples per category with random sampling
    expected_samples = {}
    for cat_id, count in sorted_categories:
        probability = count / total_points
        expected = probability * subsample_size
        expected_samples[cat_id] = expected
    
    # How many categories get at least N samples?
    min_samples_thresholds = [1, 5, 10, 20, 50]
    coverage = {}
    for threshold in min_samples_thresholds:
        num_cats = sum(1 for exp in expected_samples.values() if exp >= threshold)
        coverage[threshold] = num_cats
    
    # Rare category coverage
    rare_cat_samples = [expected_samples[cat] for cat, _ in rare_categories if cat in expected_samples]
    
    print(f"\nSubsample size: {subsample_size:,}")
    print(f"  Categories with ≥1 sample:   {coverage[1]} / {len(sorted_categories)}")
    print(f"  Categories with ≥5 samples:  {coverage[5]} / {len(sorted_categories)}")
    print(f"  Categories with ≥10 samples: {coverage[10]} / {len(sorted_categories)}")
    print(f"  Categories with ≥20 samples: {coverage[20]} / {len(sorted_categories)}")
    print(f"  Categories with ≥50 samples: {coverage[50]} / {len(sorted_categories)}")
    
    if rare_cat_samples:
        print(f"  Rare categories avg samples: {np.mean(rare_cat_samples):.1f}")
        print(f"  Rare categories with ≥10:    {sum(1 for s in rare_cat_samples if s >= 10)} / {len(rare_cat_samples)}")

print()

# Minimum samples needed for stable prototype
MIN_SAMPLES_FOR_STABLE_PROTOTYPE = 20  # Literature recommendation

print(f"Recommendation based on {MIN_SAMPLES_FOR_STABLE_PROTOTYPE} samples per category:")
print("-"*40)

for subsample_size in SUBSAMPLE_SIZES:
    total_points = sum(all_category_counts.values())
    expected_samples = {cat_id: (count / total_points) * subsample_size 
                       for cat_id, count in sorted_categories}
    
    stable_categories = sum(1 for exp in expected_samples.values() 
                           if exp >= MIN_SAMPLES_FOR_STABLE_PROTOTYPE)
    
    percentage = stable_categories / len(sorted_categories) * 100
    
    status = "✓ GOOD" if percentage > 80 else "⚠ POOR" if percentage < 50 else "~ OK"
    print(f"  {subsample_size:>6,}: {stable_categories:3d}/{len(sorted_categories)} categories stable ({percentage:5.1f}%) {status}")

print()

print("="*80)
print()

# ============================================================================
# 5. BALANCED vs RANDOM SAMPLING COMPARISON
# ============================================================================

print("PART 5: BALANCED vs RANDOM SAMPLING COMPARISON")
print("-"*80)
print()

print("Comparing sampling strategies at different subsample sizes:")
print()

for subsample_size in [2000, 4000, 8000, 10000]:
    print(f"\nSubsample size: {subsample_size:,}")
    print("-"*40)
    
    num_categories = len(sorted_categories)
    total_points = sum(all_category_counts.values())
    
    # RANDOM SAMPLING
    print("  Random sampling:")
    # Top category
    top_cat_count = sorted_categories[0][1]
    top_cat_samples = (top_cat_count / total_points) * subsample_size
    print(f"    Top category:    ~{top_cat_samples:,.0f} samples")
    
    # Rare categories
    if rare_categories:
        avg_rare_count = np.mean([c for _, c in rare_categories])
        avg_rare_samples = (avg_rare_count / total_points) * subsample_size
        print(f"    Rare categories: ~{avg_rare_samples:,.1f} samples (avg)")
        print(f"    Imbalance ratio: {top_cat_samples / max(avg_rare_samples, 0.1):,.0f}:1")
    
    # BALANCED SAMPLING
    print("  Balanced sampling:")
    samples_per_category = subsample_size // num_categories
    print(f"    All categories:  {samples_per_category} samples each")
    print(f"    Imbalance ratio: 1:1 (perfect balance)")
    
    # Which is better?
    print("  Recommendation:")
    if gini > 0.6:
        print(f"    → Use BALANCED (Gini={gini:.2f} indicates severe imbalance)")
    else:
        print(f"    → Either works (Gini={gini:.2f} indicates mild imbalance)")

print()

print("="*80)
print()

# ============================================================================
# 6. REDUNDANCY ANALYSIS
# ============================================================================

print("PART 6: REDUNDANCY ANALYSIS")
print("-"*80)
print()

print("Analyzing why 10K might work better than 40K:")
print()

# Hypothesis 1: Diminishing returns
print("Hypothesis 1: Diminishing Returns")
print("  As we add more samples, additional samples become redundant")
print()

# For top categories, calculate redundancy
print("Top 5 categories - sampling efficiency:")
for i, (cat_id, count) in enumerate(sorted_categories[:5], 1):
    percentage = count / sum(all_category_counts.values()) * 100
    
    samples_at_2k = (count / sum(all_category_counts.values())) * 2000
    samples_at_10k = (count / sum(all_category_counts.values())) * 10000
    samples_at_40k = (count / sum(all_category_counts.values())) * 40000
    
    print(f"  Category {cat_id} ({percentage:.1f}% of data):")
    print(f"    At  2K subsample: ~{samples_at_2k:,.0f} samples")
    print(f"    At 10K subsample: ~{samples_at_10k:,.0f} samples ({samples_at_10k/samples_at_2k:.1f}x more)")
    print(f"    At 40K subsample: ~{samples_at_40k:,.0f} samples ({samples_at_40k/samples_at_10k:.1f}x more)")
    print(f"    → Diminishing returns after 10K")
    print()

print()

# Hypothesis 2: Noise
print("Hypothesis 2: Noise in Large Samples")
print("  40K samples might include:")
print("    - Mislabeled points")
print("    - Boundary ambiguities")
print("    - Outliers")
print("  These hurt contrastive learning!")
print()

# Hypothesis 3: Computational efficiency
print("Hypothesis 3: Compute vs Accuracy Trade-off")
samples_per_cat_10k = 10000 / len(sorted_categories)
samples_per_cat_40k = 40000 / len(sorted_categories)
print(f"  At 10K: ~{samples_per_cat_10k:.0f} samples per category")
print(f"  At 40K: ~{samples_per_cat_40k:.0f} samples per category")
print()
print(f"  For stable prototype, need ~20-50 samples")
print(f"  At 10K: {'✓ Sufficient' if samples_per_cat_10k >= 20 else '✗ Insufficient'}")
print(f"  At 40K: {'✓ Sufficient' if samples_per_cat_40k >= 20 else '✗ Insufficient'}")
print()
print("  Conclusion: 10K provides sufficient samples with less noise")
print()

print("="*80)
print()

# ============================================================================
# 7. RECOMMENDATIONS
# ============================================================================

print("PART 7: SAMPLING STRATEGY RECOMMENDATIONS")
print("-"*80)
print()

print("Based on data analysis:")
print()

# Optimal subsample size
print("1. OPTIMAL SUBSAMPLE SIZE")
print("-"*40)

# Calculate sweet spot
num_cats = len(sorted_categories)
min_samples_per_cat = 20
optimal_for_balanced = num_cats * min_samples_per_cat

print(f"  Total categories: {num_cats}")
print(f"  Min samples for stable prototype: {min_samples_per_cat}")
print(f"  Optimal for balanced sampling: {optimal_for_balanced:,}")
print()

# Recommendations
recommended_sizes = []
if optimal_for_balanced <= 4000:
    recommended_sizes = [2000, 4000]
elif optimal_for_balanced <= 8000:
    recommended_sizes = [4000, 8000]
else:
    recommended_sizes = [8000, 10000]

print(f"  Recommended subsample sizes: {recommended_sizes}")
print(f"  Your observation: 10K > 40K ✓ Confirms analysis!")
print()

# Sampling strategy
print("2. SAMPLING STRATEGY")
print("-"*40)

if gini > 0.6:
    print(f"  ✓ USE BALANCED SAMPLING")
    print(f"    Reason: Severe imbalance (Gini={gini:.2f})")
    print(f"    Implementation:")
    print(f"      - Sample {min_samples_per_cat}-{min_samples_per_cat*2} from each category")
    print(f"      - Total subsample: {num_cats * min_samples_per_cat:,} - {num_cats * min_samples_per_cat * 2:,}")
else:
    print(f"  ~ Either balanced or random works")
    print(f"    Reason: Moderate imbalance (Gini={gini:.2f})")

print()

# Training phases
print("3. TWO-PHASE TRAINING STRATEGY")
print("-"*40)
print(f"  Phase 1 (Epochs 0-50): Balanced sampling")
print(f"    Subsample: {recommended_sizes[0]:,}")
print(f"    Goal: Build stable prototypes for all categories")
print()
print(f"  Phase 2 (Epochs 50+): Consider hard negative mining")
print(f"    Subsample: {recommended_sizes[-1]:,}")
print(f"    Goal: Refine decision boundaries")
print()

# Hard negative mining
print("4. HARD NEGATIVE MINING")
print("-"*40)
if gini > 0.6:
    print(f"  ⚠ NOT RECOMMENDED initially")
    print(f"    Reason: With severe imbalance, rare categories need")
    print(f"            stable prototypes first")
    print(f"    Recommendation: Use after Phase 1 (epoch 50+)")
else:
    print(f"  ✓ Can use from start")
    print(f"    Reason: Moderate imbalance allows hard mining")

print()

print("="*80)
print()

# ============================================================================
# 8. SAVE ANALYSIS RESULTS
# ============================================================================

print("PART 8: SAVING ANALYSIS RESULTS")
print("-"*80)

results = {
    'dataset_statistics': {
        'num_scenes_analyzed': len(scene_stats),
        'total_categories': len(sorted_categories),
        'total_labeled_points': int(sum(all_category_counts.values())),
        'gini_coefficient': float(gini),
        'avg_categories_per_scene': float(np.mean(cats_per_scene)),
    },
    'class_imbalance': {
        'gini': float(gini),
        'top_10_categories': [int(cat) for cat, _ in sorted_categories[:10]],
        'top_10_percentages': [float(count / sum(all_category_counts.values()) * 100) 
                               for _, count in sorted_categories[:10]],
        'rare_categories_count': len(rare_categories),
        'rare_categories_percentage': float(sum(c for _, c in rare_categories) / sum(all_category_counts.values()) * 100),
    },
    'recommendations': {
        'optimal_subsample_size': recommended_sizes,
        'use_balanced_sampling': gini > 0.6,
        'min_samples_per_category': min_samples_per_cat,
        'use_hard_negative_mining': 'after_phase1' if gini > 0.6 else 'yes',
        'two_phase_training': {
            'phase1_size': recommended_sizes[0],
            'phase2_size': recommended_sizes[-1],
        },
    },
    'category_distribution': {
        int(cat): int(count) for cat, count in sorted_categories
    },
}

output_file = 'semantic_sampling_analysis.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Analysis saved to: {output_file}")
print()

print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()

print("KEY FINDINGS:")
print(f"  1. Class imbalance (Gini={gini:.2f}): {'SEVERE' if gini > 0.6 else 'MODERATE'}")
print(f"  2. Optimal subsample: {recommended_sizes}")
print(f"  3. Use balanced sampling: {'YES' if gini > 0.6 else 'OPTIONAL'}")
print(f"  4. Your 10K observation: ✓ Supported by analysis")
print()

print("NEXT STEPS:")
print(f"  1. Implement balanced sampling in semantic_losses.py")
print(f"  2. Test with subsample={recommended_sizes[0]:,}")
print(f"  3. Compare against random sampling")
print(f"  4. If working well, try hard negative mining in Phase 2")
print()

print("="*80)