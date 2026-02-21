#!/usr/bin/env python3
"""
Quick test to verify SceneSplat-7K dataset loads correctly
Run this BEFORE submitting the full training job
"""

import os
import sys

print("="*60)
print("Dataset Loading Test")
print("="*60)
print()

# Add Can3Tok to path
can3tok_path = "/gpfs/work3/0/prjs1291/Hafeez_thesis/Can3Tok"
if os.path.exists(can3tok_path):
    sys.path.insert(0, can3tok_path)
    os.chdir(can3tok_path)
    print(f"✓ Using Can3Tok from: {can3tok_path}")
else:
    print(f"⚠ Can3Tok path not found: {can3tok_path}")
print()

# Test different paths
data_paths = [
    "/gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/interior_gs/train",
    "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train"
]

print("Checking data paths...")
valid_path = None
for path in data_paths:
    if os.path.exists(path):
        scenes = os.listdir(path)
        print(f"✓ Found: {path}")
        print(f"  Scenes: {len(scenes)}")
        print(f"  First 3: {scenes[:3]}")
        valid_path = path
        break
    else:
        print(f"✗ Not found: {path}")

if not valid_path:
    print("\nERROR: No valid data path found!")
    print("Please check your dataset location.")
    sys.exit(1)

print()
print(f"Using: {valid_path}")
print()

# Try loading the dataset
print("Loading dataset...")
try:
    from gs_dataset_scenesplat import gs_dataset
    
    dataset = gs_dataset(
        root=valid_path,
        resol=200,
        random_permute=True,
        train=True,
        sampling_method='hybrid'
    )
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total scenes: {len(dataset)}")
    print()
    
    # Try loading one sample
    print("Testing sample loading...")
    sample, idx = dataset[0]
    print(f"✓ Sample loaded")
    print(f"  Shape: {sample.shape}")
    print(f"  Scene index: {idx}")
    print()
    
    # Test data loader
    print("Testing data loader...")
    import torch.utils.data as Data
    
    loader = Data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # 0 for testing
        drop_last=True
    )
    
    print(f"✓ Data loader created")
    print(f"  Batches per epoch: {len(loader)}")
    print()
    
    if len(loader) == 0:
        print("ERROR: Data loader has 0 batches!")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Batch size: 32")
        print(f"  Problem: Dataset too small or batch size too large")
    else:
        # Try loading one batch
        print("Testing batch loading...")
        batch, indices = next(iter(loader))
        print(f"✓ Batch loaded")
        print(f"  Batch shape: {batch.shape}")
        print(f"  Indices: {indices[:5].tolist()}...")
        print()
    
    print("="*60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("="*60)
    print()
    print("Dataset is ready for training!")
    print(f"Run: sbatch train_minimal.job")
    
except Exception as e:
    print()
    print("="*60)
    print("✗✗✗ TEST FAILED ✗✗✗")
    print("="*60)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("Please fix the error before training.")
    sys.exit(1)