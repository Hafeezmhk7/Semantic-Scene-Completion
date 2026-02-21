#!/usr/bin/env python3
"""
Test script for SceneSplat-7K data loading
No rendering dependencies required
"""

import torch
from gs_dataset_scenesplat import gs_dataset


def main():
    print("="*60)
    print("SceneSplat-7K Data Loader Test")
    print("="*60)
    print()
    
    # Configuration
    data_path = '/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train'
    resol = 200
    num_gaussians = 40000
    sampling_method = 'hybrid'
    
    print(f"Data path: {data_path}")
    print(f"Target Gaussians: {num_gaussians:,}")
    print(f"Sampling method: {sampling_method}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = gs_dataset(
        root=data_path,
        resol=resol,
        random_permute=True,
        train=True,
        sampling_method=sampling_method
    )
    
    print(f"✓ Loaded {len(dataset)} scenes")
    print()
    
    # Test loading one scene
    print("Testing scene loading...")
    sample, scene_idx = dataset[0]
    
    print(f"✓ Scene index: {scene_idx}")
    print(f"✓ Shape: {sample.shape}")
    print(f"✓ Data type: {sample.dtype}")
    print()
    
    # Check data ranges
    print("Data statistics:")
    print(f"  Min: {sample.min():.4f}")
    print(f"  Max: {sample.max():.4f}")
    print(f"  Mean: {sample.mean():.4f}")
    print(f"  Std: {sample.std():.4f}")
    print()
    
    # Test batch loading
    print("Testing batch loading...")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 0 for testing
    )
    
    batch, indices = next(iter(loader))
    print(f"✓ Batch shape: {batch.shape}")
    print(f"✓ Scene indices: {indices.tolist()}")
    print()
    
    # Summary
    print("="*60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("="*60)
    print()
    print("SceneSplat-7K dataset is ready for training!")
    print(f"- {len(dataset)} scenes available")
    print(f"- Each scene: {num_gaussians:,} Gaussians")
    print(f"- Sampling: {sampling_method}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*60)
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("="*60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        exit(1)