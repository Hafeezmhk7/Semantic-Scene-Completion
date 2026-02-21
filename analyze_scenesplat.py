"""
UPDATED: Full Dataset Analysis for 72 ScanNet Categories
=========================================================

The previous analysis only found 39 categories in 100 scenes.
But ScanNet has 72 categories total.

This script:
1. Analyzes MORE scenes to find all categories
2. Identifies which categories are extremely rare
3. Recalculates optimal subsample size for 72 categories
"""

import numpy as np
import os
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import json

# Paths
train_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6"

print("="*80)
print("FULL DATASET ANALYSIS - FINDING ALL 72 SCANNET CATEGORIES")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Analyze ALL scenes (or as many as needed to find all 72 categories)
MAX_SCENES_TO_CHECK = 1000  # Will stop early if all 72 found

print(f"Configuration:")
print(f"  Will check up to {MAX_SCENES_TO_CHECK} scenes")
print(f"  Looking for all 72 ScanNet categories")
print()

# ============================================================================
# LOAD DATA AND FIND ALL CATEGORIES
# ============================================================================

print("PART 1: FINDING ALL CATEGORIES")
print("-"*80)

all_scenes = sorted(os.listdir(train_path))
print(f"Total scenes available: {len(all_scenes)}")
print()

# Track statistics
all_category_counts = Counter()
categories_found = set()
scenes_checked = 0
scenes_with_data = 0

print("Scanning scenes for semantic labels...")
print(f"(Will stop after finding all 72 categories or checking {MAX_SCENES_TO_CHECK} scenes)")
print()

for scene_name in tqdm(all_scenes[:MAX_SCENES_TO_CHECK], desc="Scanning"):
    scene_path = os.path.join(train_path, scene_name)
    segment_file = os.path.join(scene_path, 'segment.npy')
    
    if not os.path.exists(segment_file):
        continue
    
    scenes_checked += 1
    
    # Load segment labels
    segment = np.load(segment_file)
    valid_labels = segment[segment >= 0]
    
    if len(valid_labels) == 0:
        continue
        
    scenes_with_data += 1
    
    # Count categories
    scene_categories = Counter(valid_labels)
    all_category_counts.update(scene_categories)
    categories_found.update(scene_categories.keys())
    
    # Progress update every 100 scenes
    if scenes_checked % 100 == 0:
        print(f"\n  Checked {scenes_checked} scenes, found {len(categories_found)} unique categories so far...")
    
    # Stop if we found all 72
    if len(categories_found) >= 72:
        print(f"\n✓ Found all 72 categories after checking {scenes_checked} scenes!")
        break

print()
print(f"✓ Scanned {scenes_checked} scenes with valid data")
print(f"✓ Found {len(categories_found)} unique categories")
print()

# ============================================================================
# ANALYZE CATEGORY DISTRIBUTION
# ============================================================================

print("="*80)
print("PART 2: CATEGORY DISTRIBUTION (FULL DATASET)")
print("-"*80)
print()

sorted_categories = sorted(all_category_counts.items(), key=lambda x: x[1], reverse=True)
total_points = sum(all_category_counts.values())

print(f"Total categories found: {len(sorted_categories)}")
print(f"Total labeled points: {total_points:,}")
print()

# Check if we have all 72
if len(sorted_categories) < 72:
    missing_count = 72 - len(sorted_categories)
    print(f"⚠️  WARNING: Only found {len(sorted_categories)}/72 categories")
    print(f"   {missing_count} categories missing from dataset!")
    print(f"   These categories are either:")
    print(f"     1. Extremely rare (not in training set)")
    print(f"     2. Not present in SceneSplat interior scenes")
    print()
else:
    print(f"✓ Found all {len(sorted_categories)} categories!")
    print()

# Display distribution
print("Category Distribution:")
print("-"*80)
print(f"{'Rank':<6} {'Category ID':<12} {'Count':>15} {'Percentage':>12} {'Cumulative':>12}")
print("-"*80)

cumulative = 0
for i, (cat_id, count) in enumerate(sorted_categories[:50], 1):  # Show top 50
    percentage = count / total_points * 100
    cumulative += percentage
    print(f"{i:<6} {cat_id:<12} {count:>15,} {percentage:>11.2f}% {cumulative:>11.2f}%")

if len(sorted_categories) > 50:
    print(f"...")
    print(f"({len(sorted_categories) - 50} more categories)")

print()

# Gini coefficient
def gini_coefficient(counts):
    sorted_counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

category_counts_array = np.array([count for _, count in sorted_categories])
gini = gini_coefficient(category_counts_array)

print(f"Class Imbalance Metrics:")
print(f"  Gini coefficient: {gini:.3f}")
if gini > 0.7:
    print(f"  → SEVERE imbalance")
elif gini > 0.5:
    print(f"  → MODERATE imbalance")
else:
    print(f"  → MILD imbalance")

print()

# ============================================================================
# CATEGORIES PRESENT vs MISSING
# ============================================================================

print("="*80)
print("PART 3: WHICH CATEGORIES ARE PRESENT?")
print("-"*80)
print()

categories_present = set([int(cat) for cat, _ in sorted_categories])
all_scannet_ids = set(range(72))  # ScanNet72 uses IDs 0-71
categories_missing = all_scannet_ids - categories_present

if len(categories_missing) > 0:
    print(f"Categories MISSING from dataset:")
    print(f"  Count: {len(categories_missing)}")
    print(f"  IDs: {sorted(categories_missing)}")
    print()
    print(f"  These categories are not present in the {scenes_checked} scenes analyzed.")
    print(f"  They are either:")
    print(f"    - Extremely rare in interior scenes")
    print(f"    - Not applicable to SceneSplat scenes")
    print()
else:
    print(f"✓ All 72 ScanNet categories present in dataset!")
    print()

print(f"Categories PRESENT:")
print(f"  Count: {len(categories_present)}")
print(f"  IDs: {sorted(categories_present)[:20]}... ({len(categories_present)} total)")
print()

# ============================================================================
# RECALCULATE OPTIMAL SUBSAMPLE SIZE
# ============================================================================

print("="*80)
print("PART 4: OPTIMAL SUBSAMPLE SIZE (FOR ALL CATEGORIES)")
print("-"*80)
print()

# Use actual number of categories found
num_categories = len(sorted_categories)
min_samples_per_cat = 20

print(f"Calculating for {num_categories} categories found in dataset:")
print()

# If user expects 72 but we only found fewer
if num_categories < 72:
    print(f"NOTE: You mentioned 72 categories, but analysis found {num_categories}.")
    print(f"      Calculations will use {num_categories} (actual categories present).")
    print(f"      The {72 - num_categories} missing categories likely don't exist")
    print(f"      in your training data.")
    print()

subsample_sizes = [2000, 4000, 6000, 8000, 10000, 15000, 20000]

print("Subsample size options:")
print("-"*60)
print(f"{'Size':<10} {'Samples/Cat':<15} {'≥20 samples?':<15} {'Recommendation'}")
print("-"*60)

for size in subsample_sizes:
    samples_per_cat = size / num_categories
    sufficient = samples_per_cat >= min_samples_per_cat
    
    if samples_per_cat < 20:
        rec = "⚠️  TOO SMALL"
    elif 20 <= samples_per_cat < 50:
        rec = "~ Minimum"
    elif 50 <= samples_per_cat < 100:
        rec = "✓ Good"
    elif 100 <= samples_per_cat < 200:
        rec = "✓✓ Optimal"
    else:
        rec = "✓✓✓ Excellent"
    
    print(f"{size:<10,} {samples_per_cat:<15.1f} {'YES' if sufficient else 'NO':<15} {rec}")

print()

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("="*80)
print("PART 5: UPDATED RECOMMENDATIONS")
print("-"*80)
print()

# Calculate optimal subsample
optimal_min = num_categories * 50  # 50 samples per category (good)
optimal_max = num_categories * 150  # 150 samples per category (excellent)

print(f"For {num_categories} categories:")
print()
print(f"Minimum viable subsample: {num_categories * 20:,} (20 samples/cat)")
print(f"Good subsample range:     {optimal_min:,} - {optimal_max:,}")
print(f"Recommended subsample:    {optimal_min:,}")
print()

# Compare to random sampling
print(f"Coverage comparison:")
print("-"*40)

for size in [4000, 8000, 10000]:
    # Random sampling
    num_sufficient_random = sum(1 for _, count in sorted_categories 
                               if (count / total_points) * size >= 20)
    
    # Balanced sampling
    samples_per_cat_balanced = size / num_categories
    num_sufficient_balanced = num_categories if samples_per_cat_balanced >= 20 else 0
    
    print(f"\nSubsample = {size:,}:")
    print(f"  Random:   {num_sufficient_random}/{num_categories} categories get ≥20 samples ({num_sufficient_random/num_categories*100:.0f}%)")
    print(f"  Balanced: {num_sufficient_balanced}/{num_categories} categories get ≥20 samples ({num_sufficient_balanced/num_categories*100 if num_sufficient_balanced > 0 else 0:.0f}%)")

print()
print("="*80)

# ============================================================================
# KEY FINDINGS SUMMARY
# ============================================================================

print()
print("KEY FINDINGS:")
print("="*80)
print()
print(f"1. CATEGORIES FOUND: {len(sorted_categories)} / 72")
if len(sorted_categories) < 72:
    print(f"   → {72 - len(sorted_categories)} categories missing from training data")
    print(f"   → Design for {len(sorted_categories)} categories (what's actually present)")
else:
    print(f"   → All 72 categories present!")

print()
print(f"2. CLASS IMBALANCE: Gini = {gini:.3f}")
print(f"   → {'SEVERE' if gini > 0.7 else 'MODERATE' if gini > 0.5 else 'MILD'}")

print()
print(f"3. OPTIMAL SUBSAMPLE:")
if num_categories <= 40:
    recommended = 4000
elif num_categories <= 60:
    recommended = 6000
else:
    recommended = 8000
    
print(f"   → {recommended:,} samples (balanced)")
print(f"   → {recommended / num_categories:.0f} samples per category")

print()
print(f"4. SAMPLING STRATEGY:")
print(f"   → MUST use balanced sampling (Gini = {gini:.3f})")
print(f"   → Random sampling will fail on rare categories")

print()
print("="*80)

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'scenes_analyzed': scenes_checked,
    'total_categories_found': len(sorted_categories),
    'expected_categories': 72,
    'categories_missing': len(categories_missing),
    'missing_category_ids': sorted(list(categories_missing)) if categories_missing else [],
    'gini_coefficient': float(gini),
    'total_labeled_points': int(total_points),
    'recommended_subsample': int(recommended),
    'samples_per_category': float(recommended / num_categories),
    'category_distribution': {
        int(cat): int(count) for cat, count in sorted_categories
    },
}

# Convert to JSON-serializable
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return sorted(list(obj))
    else:
        return obj

results_serializable = convert_to_serializable(results)

output_file = 'full_dataset_analysis.json'
with open(output_file, 'w') as f:
    json.dump(results_serializable, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)