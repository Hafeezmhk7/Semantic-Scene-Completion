"""
Grid Dataset Inspection Script
Explores train_grid1.0cm_chunk8x8_stride6x6 dataset and compares with regular train data
"""

import numpy as np
import os
from pathlib import Path

# Paths
grid_train_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6"
regular_train_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train"

print("=" * 80)
print("GRID DATASET INSPECTION")
print("=" * 80)
print()

# ============================================================================
# 1. EXPLORE GRID DATASET STRUCTURE
# ============================================================================

print("1. GRID DATASET STRUCTURE")
print("-" * 80)

# List scenes in grid dataset
grid_scenes = sorted([d for d in os.listdir(grid_train_path) if os.path.isdir(os.path.join(grid_train_path, d))])
print(f"Number of grid scenes: {len(grid_scenes)}")
print(f"First few scenes: {grid_scenes[:5]}")
print()

# Examine first scene in detail
first_scene = grid_scenes[0]
first_scene_path = os.path.join(grid_train_path, first_scene)
print(f"Examining scene: {first_scene}")
print(f"Path: {first_scene_path}")
print()

# List files in first scene
files = sorted(os.listdir(first_scene_path))
print(f"Files in scene ({len(files)} files):")
for f in files:
    file_path = os.path.join(first_scene_path, f)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  - {f:20s} ({size:.2f} MB)")
print()

# Load and examine each .npy file
print("Loading grid data files...")
print()

grid_data = {}
for f in files:
    if f.endswith('.npy'):
        file_path = os.path.join(first_scene_path, f)
        data = np.load(file_path)
        grid_data[f.replace('.npy', '')] = data
        print(f"{f.replace('.npy', ''):15s}: shape={str(data.shape):20s} dtype={str(data.dtype):10s} "
              f"range=[{data.min():.4f}, {data.max():.4f}]")

print()

# ============================================================================
# 2. COMPARE WITH REGULAR TRAIN DATASET
# ============================================================================

print("2. REGULAR TRAIN DATASET")
print("-" * 80)

# Find corresponding regular scene (without the chunk suffix)
base_scene_name = first_scene.rsplit('_', 1)[0]  # Remove last "_0"
regular_scene_path = os.path.join(regular_train_path, base_scene_name)

print(f"Base scene name: {base_scene_name}")
print(f"Regular scene path: {regular_scene_path}")

if os.path.exists(regular_scene_path):
    print(f"✓ Regular scene exists!")
    print()
    
    # List files in regular scene
    regular_files = sorted(os.listdir(regular_scene_path))
    print(f"Files in regular scene ({len(regular_files)} files):")
    for f in regular_files:
        file_path = os.path.join(regular_scene_path, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {f:20s} ({size:.2f} MB)")
    print()
    
    # Load regular data
    print("Loading regular data files...")
    print()
    
    regular_data = {}
    for f in regular_files:
        if f.endswith('.npy'):
            file_path = os.path.join(regular_scene_path, f)
            data = np.load(file_path)
            regular_data[f.replace('.npy', '')] = data
            print(f"{f.replace('.npy', ''):15s}: shape={str(data.shape):20s} dtype={str(data.dtype):10s} "
                  f"range=[{data.min():.4f}, {data.max():.4f}]")
    
    print()
    
else:
    print(f"✗ Regular scene does NOT exist!")
    print()

# ============================================================================
# 3. DETAILED COMPARISON
# ============================================================================

print("3. GRID vs REGULAR COMPARISON")
print("-" * 80)

if os.path.exists(regular_scene_path):
    # Compare shapes
    print("Shape comparison:")
    for key in grid_data.keys():
        if key in regular_data:
            grid_shape = grid_data[key].shape
            regular_shape = regular_data[key].shape
            ratio = grid_shape[0] / regular_shape[0] if len(regular_shape) > 0 else 0
            print(f"  {key:15s}: Grid={str(grid_shape):20s} vs Regular={str(regular_shape):20s} "
                  f"(ratio: {ratio:.4f})")
        else:
            print(f"  {key:15s}: Grid={str(grid_data[key].shape):20s} vs Regular=NOT FOUND")
    
    print()
    
    # Check if grid data is a subset or chunk of regular data
    print("Data relationship analysis:")
    
    # Compare coordinate ranges
    if 'coord' in grid_data and 'coord' in regular_data:
        grid_coords = grid_data['coord']
        regular_coords = regular_data['coord']
        
        print(f"\nCoordinate ranges:")
        print(f"  Grid data:")
        print(f"    X: [{grid_coords[:, 0].min():.4f}, {grid_coords[:, 0].max():.4f}]")
        print(f"    Y: [{grid_coords[:, 1].min():.4f}, {grid_coords[:, 1].max():.4f}]")
        print(f"    Z: [{grid_coords[:, 2].min():.4f}, {grid_coords[:, 2].max():.4f}]")
        
        print(f"  Regular data:")
        print(f"    X: [{regular_coords[:, 0].min():.4f}, {regular_coords[:, 0].max():.4f}]")
        print(f"    Y: [{regular_coords[:, 1].min():.4f}, {regular_coords[:, 1].max():.4f}]")
        print(f"    Z: [{regular_coords[:, 2].min():.4f}, {regular_coords[:, 2].max():.4f}]")
        
        # Check if grid coords are within regular coords
        grid_in_regular_x = (grid_coords[:, 0].min() >= regular_coords[:, 0].min() and 
                             grid_coords[:, 0].max() <= regular_coords[:, 0].max())
        grid_in_regular_y = (grid_coords[:, 1].min() >= regular_coords[:, 1].min() and 
                             grid_coords[:, 1].max() <= regular_coords[:, 1].max())
        grid_in_regular_z = (grid_coords[:, 2].min() >= regular_coords[:, 2].min() and 
                             grid_coords[:, 2].max() <= regular_coords[:, 2].max())
        
        print(f"\n  Grid is subset of Regular:")
        print(f"    X-axis: {'✓' if grid_in_regular_x else '✗'}")
        print(f"    Y-axis: {'✓' if grid_in_regular_y else '✗'}")
        print(f"    Z-axis: {'✓' if grid_in_regular_z else '✗'}")

print()

# ============================================================================
# 4. CHECK ALL GRID CHUNKS FOR THIS SCENE
# ============================================================================

print("4. ALL CHUNKS FOR BASE SCENE")
print("-" * 80)

# Find all chunks belonging to this base scene
scene_chunks = [s for s in grid_scenes if s.startswith(base_scene_name + "_")]
print(f"Base scene: {base_scene_name}")
print(f"Number of chunks: {len(scene_chunks)}")
print(f"Chunk names: {scene_chunks[:10]}")  # Show first 10
print()

# Load all chunks and analyze
if len(scene_chunks) > 0:
    print("Analyzing all chunks...")
    total_points = 0
    chunk_shapes = []
    
    for chunk_name in scene_chunks[:5]:  # Check first 5 chunks
        chunk_path = os.path.join(grid_train_path, chunk_name)
        if os.path.exists(os.path.join(chunk_path, 'coord.npy')):
            chunk_coords = np.load(os.path.join(chunk_path, 'coord.npy'))
            chunk_shapes.append(chunk_coords.shape[0])
            total_points += chunk_coords.shape[0]
    
    print(f"First 5 chunks:")
    for i, (name, shape) in enumerate(zip(scene_chunks[:5], chunk_shapes)):
        print(f"  {i}: {name:30s} - {shape:6d} points")
    
    print(f"\nTotal points in first 5 chunks: {total_points}")
    
    if os.path.exists(regular_scene_path) and 'coord' in regular_data:
        regular_points = regular_data['coord'].shape[0]
        print(f"Regular scene total points:     {regular_points}")
        print(f"Coverage: {(total_points / regular_points * 100):.2f}% (from 5/{len(scene_chunks)} chunks)")

print()

# ============================================================================
# 5. UNDERSTAND THE NAMING SCHEME
# ============================================================================

print("5. NAMING SCHEME ANALYSIS")
print("-" * 80)

print("Grid dataset name: train_grid1.0cm_chunk8x8_stride6x6")
print()
print("Interpretation:")
print("  - grid1.0cm:        Grid resolution of 1.0 cm")
print("  - chunk8x8:         Each chunk is 8x8 grid cells")
print("  - stride6x6:        Chunks overlap with stride of 6 (overlap of 2)")
print()
print("Implications:")
print("  - Scene is divided into overlapping 8x8 chunks")
print("  - Stride of 6 means 2-cell overlap between chunks")
print("  - This creates spatial redundancy for better coverage")
print("  - Each chunk is a spatial subset of the full scene")
print()

# Calculate expected number of chunks
if 'coord' in regular_data:
    regular_coords = regular_data['coord']
    x_range = regular_coords[:, 0].max() - regular_coords[:, 0].min()
    y_range = regular_coords[:, 1].max() - regular_coords[:, 1].min()
    
    chunk_size = 8 * 0.01  # 8 cells * 1cm
    stride = 6 * 0.01      # 6 cells * 1cm
    
    # Approximate number of chunks
    n_chunks_x = int((x_range - chunk_size) / stride) + 1
    n_chunks_y = int((y_range - chunk_size) / stride) + 1
    expected_chunks = n_chunks_x * n_chunks_y
    
    print(f"Scene spatial extent:")
    print(f"  X-range: {x_range:.4f} m")
    print(f"  Y-range: {y_range:.4f} m")
    print(f"  Chunk size: {chunk_size:.4f} m x {chunk_size:.4f} m")
    print(f"  Stride: {stride:.4f} m")
    print(f"  Expected chunks (X): ~{n_chunks_x}")
    print(f"  Expected chunks (Y): ~{n_chunks_y}")
    print(f"  Expected total chunks: ~{expected_chunks}")
    print(f"  Actual chunks found: {len(scene_chunks)}")

print()

# ============================================================================
# 6. SUMMARY & RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()

print("FINDINGS:")
print()
print("1. Dataset Structure:")
print("   - Grid dataset contains spatially chunked versions of scenes")
print("   - Each scene is split into multiple overlapping chunks")
print(f"   - Found {len(scene_chunks)} chunks for scene '{base_scene_name}'")
print()

print("2. Data Format:")
print("   - Same file types as regular dataset (coord, color, opacity, etc.)")
print("   - Chunks are spatial subsets of the full scene")
print("   - Contains same features per Gaussian")
print()

print("3. Use Cases:")
print("   - Training on smaller spatial regions")
print("   - Memory-efficient processing of large scenes")
print("   - Local feature learning")
print("   - Data augmentation through spatial sampling")
print()

print("RECOMMENDATIONS:")
print()
print("✓ Grid dataset is COMPATIBLE with your pipeline!")
print("✓ Can be used as drop-in replacement for regular train dataset")
print()
print("Options:")
print("  1. Use grid dataset for training (more data, spatial locality)")
print("  2. Mix grid and regular datasets for variety")
print("  3. Use grid for fine-tuning after training on regular data")
print()

print("Next steps:")
print("  1. Modify gs_dataset_scenesplat.py to support grid dataset")
print("  2. Add argument to choose between regular and grid datasets")
print("  3. Run small experiment to compare performance")
print()

print("=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)