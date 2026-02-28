"""
discover_training_scenes.py
============================
Helper script to scan the InteriorGS training split and identify scenes
per category for the latent space t-SNE analysis.

USAGE:
    python discover_training_scenes.py

This will:
1. Scan the training split directory
2. List all available scene IDs (unique scene prefixes from chunk names)
3. Show chunk counts per scene
4. Save a starter scene_config.json that you can manually annotate with categories

WORKFLOW:
    Step 1: Run this script to generate scene_config.json
    Step 2: Open scene_config.json and fill in the 'category' field for each scene
            using the known categories below
    Step 3: Run latent_tsne_analysis.py using the annotated config

KNOWN CATEGORIES IN InteriorGS (from supervisor):
    - concert_hall
    - restaurant
    - apartment
    - cinema
    - convenience_store
    - library
    - go_kart
    - (other / unknown)

KNOWN TRAINING SCENES (from supervisor):
    train/0210_840153  -> concert_hall
    train/0832_840249  -> apartment
    train/0839_841757  -> apartment
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_ROOT = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"
TRAIN_SPLIT  = "train_grid1.0cm_chunk8x8_stride6x6"
OUTPUT_JSON  = "scene_config.json"

# Scenes the supervisor already identified with categories (training split)
KNOWN_CATEGORIES = {
    "0210_840153": "concert_hall",
    "0832_840249": "apartment",
    "0839_841757": "apartment",
}

# ============================================================================
# HELPER: Extract scene ID from chunk filename
# ============================================================================

def extract_scene_id(filename: str) -> str:
    """
    Extract the base scene ID from a chunk filename.

    Chunk naming conventions we handle:
        0210_840153.npy                  → 0210_840153
        0210_840153_chunk_2_3.npy        → 0210_840153
        0210_840153_c2_r3.npy            → 0210_840153
        scene_0210_840153_000.npy        → 0210_840153

    The scene ID pattern is: DDDD_DDDDDD (4 digits _ 6 digits)
    """
    stem = Path(filename).stem
    # Match pattern NNNN_NNNNNN
    match = re.search(r'(\d{4}_\d{6})', stem)
    if match:
        return match.group(1)
    return stem   # fallback: use full stem


# ============================================================================
# SCAN DATASET
# ============================================================================

def scan_dataset(root: str, split: str) -> dict:
    """
    Scan the split directory and return:
        { scene_id: { 'chunks': [list of filenames], 'path': str } }
    """
    split_dir = os.path.join(root, split)

    if not os.path.exists(split_dir):
        print(f"ERROR: Directory not found: {split_dir}")
        return {}

    scenes = defaultdict(lambda: {'chunks': [], 'path': split_dir})

    # Walk the directory — chunks may be flat files or in subdirectories
    for root_dir, subdirs, files in os.walk(split_dir):
        for fname in sorted(files):
            if fname.endswith(('.npy', '.pt', '.npz')):
                scene_id = extract_scene_id(fname)
                rel_path = os.path.relpath(os.path.join(root_dir, fname), split_dir)
                scenes[scene_id]['chunks'].append(rel_path)

    # Also check if chunks are stored as subdirectories
    top_level_dirs = [d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d))]
    for d in sorted(top_level_dirs):
        scene_id = extract_scene_id(d)
        if scene_id not in scenes:
            # Scan subdirectory
            sub_path = os.path.join(split_dir, d)
            sub_files = [f for f in os.listdir(sub_path)
                         if f.endswith(('.npy', '.pt', '.npz'))]
            if sub_files:
                scenes[scene_id]['chunks'].extend(sub_files)
                scenes[scene_id]['subdir'] = d

    return dict(scenes)


# ============================================================================
# GENERATE CONFIG JSON
# ============================================================================

def generate_config(scenes: dict, known_categories: dict) -> list:
    """
    Generate a list of scene entries for the config JSON.
    Pre-fills known categories, leaves others as 'unknown'.
    """
    config = []
    for scene_id in sorted(scenes.keys()):
        info = scenes[scene_id]
        entry = {
            "scene_id":   scene_id,
            "category":   known_categories.get(scene_id, "unknown"),
            "num_chunks": len(info['chunks']),
            "split":      "train",
            "note":       "",
        }
        config.append(entry)
    return config


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_summary(scenes: dict, known_categories: dict):
    print(f"\n{'='*70}")
    print(f"TRAINING SPLIT SCENE DISCOVERY")
    print(f"{'='*70}")
    print(f"Directory: {DATASET_ROOT}/{TRAIN_SPLIT}")
    print(f"Total scenes found: {len(scenes)}")
    print()

    # Group by category
    categorized   = {k: v for k, v in known_categories.items() if k in scenes}
    uncategorized = [s for s in scenes if s not in known_categories]

    print(f"Already categorized: {len(categorized)}")
    for scene_id, cat in sorted(categorized.items()):
        n = len(scenes[scene_id]['chunks'])
        print(f"  {scene_id}  →  {cat}  ({n} chunks)")

    print()
    print(f"Uncategorized scenes: {len(uncategorized)}")
    print(f"  (First 50 shown — check scene_config.json for full list)")
    for scene_id in sorted(uncategorized)[:50]:
        n = len(scenes[scene_id]['chunks'])
        print(f"  {scene_id}  ({n} chunks)")

    print()
    print("Chunk count distribution:")
    counts = defaultdict(int)
    for info in scenes.values():
        counts[len(info['chunks'])] += 1
    for n_chunks in sorted(counts.keys()):
        print(f"  {n_chunks} chunks: {counts[n_chunks]} scenes")

    print(f"{'='*70}")


# ============================================================================
# PRINT ANNOTATION GUIDE
# ============================================================================

def print_annotation_guide():
    print(f"""
{'='*70}
HOW TO ANNOTATE scene_config.json
{'='*70}
1. Open scene_config.json in any text editor
2. For each scene entry, change "category": "unknown" to the correct category
3. Use these category names (consistent with the analysis script):
     "concert_hall"
     "restaurant"
     "apartment"
     "cinema"
     "convenience_store"
     "library"
     "go_kart"
     "office"
     "gym"
     "unknown"   ← leave as unknown to exclude from analysis

4. Aim for at least 5 scenes per category for meaningful t-SNE clustering
   (more is better — 10+ gives clear visual separation)

5. Leave "note" field empty or add hints (e.g. "large concert hall, 2 floors")

6. After annotation, run:
     python latent_tsne_analysis.py \\
         --scene_config scene_config.json \\
         --checkpoint_baseline /path/to/baseline/best_model.pth \\
         --checkpoint_semantic /path/to/semantic/best_model.pth \\
         --output_dir tsne_results/

TIPS FOR FINDING SCENES:
- Browse the InteriorGS dataset viewer or check the ScanNet72 labels
- The 6-digit suffix (e.g. 840153) is the ScanNet scene number
- ScanNet scene metadata is at: http://kaldir.vc.in.tum.de/scannet/
- Ask your supervisor for a scene-to-category mapping CSV if available
{'='*70}
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Scanning dataset directory...")

    scenes = scan_dataset(DATASET_ROOT, TRAIN_SPLIT)

    if not scenes:
        print(f"\nNo scenes found in {DATASET_ROOT}/{TRAIN_SPLIT}")
        print("Please check the path and run again.")
        print("\nExpected structure:")
        print("  DATASET_ROOT/TRAIN_SPLIT/")
        print("    0210_840153_chunk_0_0.npy")
        print("    0210_840153_chunk_0_1.npy")
        print("    ... (or subdirectories per scene)")
        return

    print_summary(scenes, KNOWN_CATEGORIES)

    config = generate_config(scenes, KNOWN_CATEGORIES)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved: {OUTPUT_JSON}  ({len(config)} scenes)")
    print_annotation_guide()


if __name__ == "__main__":
    main()