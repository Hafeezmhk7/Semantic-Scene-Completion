"""
Test Script for PCA Feature Visualization (NO Open3D!)
======================================================

Tests the PCA visualization WITHOUT needing Open3D.
Uses plyfile or manual PLY writing.

Usage:
    python test_pca_visualization_no_open3d.py
"""

import numpy as np
import torch
from pathlib import Path

# Try to import the PCA visualization module
try:
    from pca_feature_visualization import visualize_semantic_features, visualize_comparison
    print("✓ Successfully imported pca_feature_visualization_no_open3d")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("\nMake sure pca_feature_visualization.py is in the same directory!")
    exit(1)


def create_synthetic_scene(n_points=10000, n_classes=5):
    """Create synthetic point cloud with clustered features."""
    print(f"\nCreating synthetic scene with {n_points} points, {n_classes} semantic classes...")
    
    # Create 3D positions (random point cloud)
    coords = np.random.randn(n_points, 3).astype(np.float32)
    
    # Create semantic features with structure (32-dim)
    semantic_features = np.random.randn(n_points, 32).astype(np.float32) * 0.5
    
    # Assign points to semantic classes
    class_ids = np.random.randint(0, n_classes, n_points)
    
    # Add class-specific offsets to semantic features
    for i in range(n_classes):
        mask = class_ids == i
        # Each class gets a unique offset in feature space
        offset = np.random.randn(32) * 5.0
        semantic_features[mask] += offset
    
    # Create position features (just XYZ)
    position_features = coords.copy()
    
    # Create original colors (rainbow based on position)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    colors[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    colors[:, 2] = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min())
    
    print(f"✓ Created scene:")
    print(f"  Coords shape: {coords.shape}")
    print(f"  Semantic features shape: {semantic_features.shape}")
    print(f"  Position features shape: {position_features.shape}")
    print(f"  Colors shape: {colors.shape}")
    print(f"  Classes: {n_classes}")
    
    return coords, semantic_features, position_features, colors


def test_single_visualization():
    """Test 1: Single semantic feature visualization."""
    print("\n" + "="*70)
    print("TEST 1: Single Semantic Feature Visualization (NO Open3D!)")
    print("="*70)
    
    coords, semantic_features, _, _ = create_synthetic_scene(n_points=5000)
    
    output_path = "test_single_semantic.ply"
    result = visualize_semantic_features(
        coords=coords,
        features=semantic_features,
        output_path=output_path,
        brightness=1.25,
        verbose=True
    )
    
    if result:
        print(f"\n✅ TEST 1 PASSED: {output_path}")
        return True
    else:
        print(f"\n❌ TEST 1 FAILED")
        return False


def test_comparison_visualization():
    """Test 2: Comparison visualization (semantic vs position vs colors)."""
    print("\n" + "="*70)
    print("TEST 2: Comparison Visualization (NO Open3D!)")
    print("="*70)
    
    coords, semantic_features, position_features, colors = create_synthetic_scene(n_points=5000)
    
    output_dir = Path("test_comparison")
    results = visualize_comparison(
        coords=coords,
        semantic_features=semantic_features,
        positions=position_features,
        colors=colors,
        output_dir=output_dir,
        scene_name="test_scene",
        brightness=1.25
    )
    
    # Check all 3 files were created
    expected_files = ['semantic', 'position', 'original']
    all_created = all(results.get(key) is not None for key in expected_files)
    
    if all_created:
        print(f"\n✅ TEST 2 PASSED: Created {len(results)} files")
        for key, path in results.items():
            print(f"  - {key}: {path}")
        return True
    else:
        print(f"\n❌ TEST 2 FAILED: Missing files")
        return False


def test_torch_device():
    """Test 3: Verify torch device handling."""
    print("\n" + "="*70)
    print("TEST 3: Torch Device Handling")
    print("="*70)
    
    coords, semantic_features, _, _ = create_synthetic_scene(n_points=1000)
    
    # Test CPU
    print("\nTesting CPU device...")
    result_cpu = visualize_semantic_features(
        coords=coords,
        features=semantic_features,
        output_path="test_device_cpu.ply",
        device="cpu",
        verbose=False
    )
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("Testing CUDA device...")
        result_cuda = visualize_semantic_features(
            coords=coords,
            features=semantic_features,
            output_path="test_device_cuda.ply",
            device="cuda",
            verbose=False
        )
        
        if result_cpu and result_cuda:
            print(f"\n✅ TEST 3 PASSED: Both CPU and CUDA work")
            return True
        else:
            print(f"\n❌ TEST 3 FAILED")
            return False
    else:
        if result_cpu:
            print(f"\n✅ TEST 3 PASSED: CPU works (CUDA not available)")
            return True
        else:
            print(f"\n❌ TEST 3 FAILED: CPU failed")
            return False


def test_large_scene():
    """Test 4: Large scene (40k points like validation)."""
    print("\n" + "="*70)
    print("TEST 4: Large Scene (40k points)")
    print("="*70)
    
    coords, semantic_features, position_features, colors = create_synthetic_scene(n_points=40000)
    
    import time
    start = time.time()
    
    result = visualize_semantic_features(
        coords=coords,
        features=semantic_features,
        output_path="test_large_scene.ply",
        brightness=1.25,
        verbose=True
    )
    
    elapsed = time.time() - start
    
    if result:
        print(f"\n✅ TEST 4 PASSED: 40k points processed in {elapsed:.2f}s")
        return True
    else:
        print(f"\n❌ TEST 4 FAILED")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("PCA FEATURE VISUALIZATION TEST SUITE (NO Open3D!)")
    print("="*70)
    print("\n✓ This version does NOT require Open3D!")
    print("✓ Uses plyfile (if available) or manual PLY writing")
    
    tests = [
        ("Single Visualization", test_single_visualization),
        ("Comparison Visualization", test_comparison_visualization),
        ("Torch Device Handling", test_torch_device),
        ("Large Scene (40k points)", test_large_scene),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! PCA visualization is working correctly!")
        print("\n✓ NO Open3D required!")
        print("\nNext steps:")
        print("  1. Copy pca_feature_visualization.py to your project")
        print("  2. Rename it to pca_feature_visualization.py")
        print("  3. Run training to generate visualizations")
    else:
        print("\n⚠️  SOME TESTS FAILED! Check errors above.")
    
    # Cleanup test files
    print("\nCleaning up test files...")
    test_files = [
        "test_single_semantic.ply",
        "test_device_cpu.ply",
        "test_device_cuda.ply",
        "test_large_scene.ply",
        "test_comparison/test_scene_semantic_pca.ply",
        "test_comparison/test_scene_position_pca.ply",
        "test_comparison/test_scene_original_colors.ply",
    ]
    
    import os
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed: {f}")
    
    if os.path.exists("test_comparison"):
        os.rmdir("test_comparison")
        print(f"  Removed: test_comparison/")
    
    print("\n✓ Cleanup complete")


if __name__ == "__main__":
    main()