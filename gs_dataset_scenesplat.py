"""
SceneSplat Dataset for Can3Tok Training - WITH SEMANTIC LABELS (ScanNet72)
Loads 3D Gaussians + segment + instance labels for contrastive learning
ScanNet72 dataset: 72 semantic categories (0-71, with 13, 53, 61 missing)
"""

import os
import numpy as np
from torch.utils.data import Dataset
import json
from tqdm import tqdm


def voxelize(coord, voxel_size=0.4, hash_type='fnv'):
    """
    Voxelize point cloud coordinates.
    
    Returns:
        uniq_idx: Unique voxel indices (FNV hash values)
        inv_idx: Inverse indices mapping each point to its voxel
        count: Number of points per unique voxel
    """
    discrete_coord = np.floor(coord / voxel_size).astype(np.int32)
    
    if hash_type == 'fnv':
        # FNV-1a hash
        offset_basis = 2166136261
        fnv_prime = 16777619
        
        # Hash each voxel coordinate
        hash_vals = np.full(len(discrete_coord), offset_basis, dtype=np.int64)
        for i in range(3):
            hash_vals = (hash_vals ^ discrete_coord[:, i]) * fnv_prime
        
        uniq_idx, inv_idx, count = np.unique(hash_vals, return_inverse=True, return_counts=True)
        
    else:
        # Ravel multi-index as fallback
        try:
            # Shift to positive
            min_coord = discrete_coord.min(axis=0)
            shifted = discrete_coord - min_coord
            max_coord = shifted.max(axis=0)
            
            # Ravel
            ravel_idx = np.ravel_multi_index(shifted.T, max_coord + 1)
            uniq_idx, inv_idx, count = np.unique(ravel_idx, return_inverse=True, return_counts=True)
            
        except:
            # Ultimate fallback: just use sequential IDs
            uniq_idx = np.arange(len(coord))
            inv_idx = uniq_idx
            count = np.ones(len(coord), dtype=np.int64)
    
    return uniq_idx, inv_idx, count


class gs_dataset(Dataset):
    """
    SceneSplat-7K Dataset Loader with Semantic Labels (ScanNet72)
    
    ScanNet72 Categories: 72 semantic categories (0-71)
    Missing labels: 13, 53, 61 (these categories don't appear in dataset)
    
    Loads:
    - 3D Gaussian parameters (coord, color, scale, quat, opacity)
    - Segment labels (72 global categories, ScanNet72)
    - Instance labels (scene-specific objects)
    
    Outputs:
    - features: (40000, 18) Gaussian parameters
    - segment_labels: (40000,) segment IDs (0-71, ScanNet72 labels)
    - instance_labels: (40000,) instance IDs (-1 = background, ≥0 = objects)
    """
    
    # ScanNet72 category names (matching your label distribution)
    SCANNET72_CATEGORIES = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", 
        "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", 
        "toilet", "sink", "bathtub", "other furniture", "kitchen cabinet", "display", 
        "rug", "ceiling", "beam", "column", "clutter", "other structure", "other prop", 
        "whiteboard", "person", "bag", "box", "pillow", "lamp", "books", "clothes", 
        "object", "towel", "mirror", "plant", "monitor", "keyboard", "mouse", "printer", 
        "telephone", "scanner", "electronics", "blinds", "clock", "books", "paper", 
        "tools", "instrument", "sports equipment", "food", "cup", "bottle", "bowl", 
        "utensil", "can", "basket", "cart", "tissue", "fire extinguisher", "trash can", 
        "other", "stairs", "stairs", "stairs", "stairs", "stairs", "stairs"
    ]
    
    def __init__(self, root, resol=200, random_permute=False, train=True, 
                 sampling_method='opacity', max_scenes=None):
        """
        Args:
            root: Path to directory containing scene directories
            resol: Resolution parameter (not used, kept for compatibility)
            random_permute: Whether to randomly permute points
            train: Training mode flag
            sampling_method: 'hybrid', 'opacity', or 'random'
        """
        self.root = root
        self.resol = resol
        self.random_permute = random_permute
        self.train = train
        self.sampling_method = sampling_method
        
        # Get all scene directories
        self.scene_dirs = sorted([
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        # Limit scenes if max_scenes parameter is provided
        if max_scenes is not None and max_scenes < len(self.scene_dirs):
            self.scene_dirs = self.scene_dirs[:max_scenes]
            print(f"  Limited to {max_scenes} scenes (out of {len(os.listdir(root))} available)")

        if len(self.scene_dirs) == 0:
            raise ValueError(f"No scene directories found in {root}")
        
        # ScanNet72 has 72 categories (0-71)
        self.num_segment_categories = 72
        
        print(f"✓ Loaded {len(self.scene_dirs)} scenes from {root}")
        print(f"✓ Sampling method: {sampling_method}")
        print(f"✓ Dataset type: ScanNet72 with {self.num_segment_categories} categories")
        print(f"✓ Missing categories in dataset: [13, 53, 61] (never appear)")
        print(f"✓ Expected InfoNCE loss with random features: log(72) ≈ {np.log(72):.4f}")
    
    def __len__(self):
        return len(self.scene_dirs)
    
    def __getitem__(self, idx):
        """
        Load and process a single scene with semantic labels.
        
        Returns:
            dict with keys:
                'features': (40000, 18) Gaussian parameters
                'segment_labels': (40000,) segment IDs (0-71, ScanNet72)
                'instance_labels': (40000,) instance IDs (-1 = background, ≥0 = objects)
                'scene_idx': int, scene index
                'has_semantics': bool, whether semantic labels exist
        """
        scene_dir = self.scene_dirs[idx]
        
        # ====================================================================
        # STEP 1: Load Gaussian parameters
        # ====================================================================
        coord = np.load(os.path.join(scene_dir, 'coord.npy'))      # (N, 3)
        color = np.load(os.path.join(scene_dir, 'color.npy'))      # (N, 3)
        scale = np.load(os.path.join(scene_dir, 'scale.npy'))      # (N, 3)
        quat = np.load(os.path.join(scene_dir, 'quat.npy'))        # (N, 4)
        opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))  # (N,)
        
        # ====================================================================
        # STEP 2: Load semantic labels (ScanNet72)
        # ====================================================================
        try:
            segment = np.load(os.path.join(scene_dir, 'segment.npy'))    # (N,)
            instance = np.load(os.path.join(scene_dir, 'instance.npy'))  # (N,)
            has_semantics = True
            
            # Verify labels are in ScanNet72 range (0-71)
            valid_segments = segment[segment >= 0]
            if len(valid_segments) > 0:
                max_label = valid_segments.max()
                min_label = valid_segments.min()
                if max_label >= 72 or min_label < 0:
                    print(f"⚠️  Warning: Scene {idx} has labels outside ScanNet72 range: {min_label} to {max_label}")
                    
        except FileNotFoundError:
            # If semantic files don't exist, create dummy labels
            segment = np.full(len(coord), -1, dtype=np.int16)
            instance = np.full(len(coord), -1, dtype=np.int32)
            has_semantics = False
        
        num_points = len(coord)
        
        # ====================================================================
        # STEP 3: Importance sampling
        # ====================================================================
        target_points = 40000
        
        if self.sampling_method == 'hybrid':
            # Hybrid: 0.7*opacity + 0.3*scale_magnitude
            scale_magnitude = np.linalg.norm(scale, axis=1)
            scale_norm = (scale_magnitude - scale_magnitude.min()) / (scale_magnitude.max() - scale_magnitude.min() + 1e-8)
            opacity_norm = (opacity - opacity.min()) / (opacity.max() - opacity.min() + 1e-8)
            importance = 0.7 * opacity_norm + 0.3 * scale_norm
            
        elif self.sampling_method == 'opacity':
            # Pure opacity-based sampling
            importance = opacity
            
        else:
            # Random sampling
            importance = np.ones(num_points)
        
        # Sample top 2x important points, then randomly select target_points
        if num_points > target_points * 2:
            top_indices = np.argsort(importance)[-target_points * 2:]
            selected = np.random.choice(top_indices, target_points, replace=False)
        elif num_points > target_points:
            selected = np.random.choice(num_points, target_points, replace=False)
        else:
            # Pad if too few points
            selected = np.concatenate([
                np.arange(num_points),
                np.random.choice(num_points, target_points - num_points, replace=True)
            ])
        
        # Apply selection to ALL arrays (including semantic labels!)
        coord = coord[selected]
        color = color[selected]
        scale = scale[selected]
        quat = quat[selected]
        opacity = opacity[selected]
        segment = segment[selected]    # ScanNet72 labels
        instance = instance[selected]  # instance labels
        
        # ====================================================================
        # STEP 4: Voxelization
        # ====================================================================
        volume_dims = 40
        resolution = 16.0 / volume_dims  # 0.4
        
        # Properly unpack 3 return values from voxelize
        uniq_idx, inv_idx, count = voxelize(coord, resolution, 'fnv')
        
        # Compute voxel centers (Positional Encoding)
        origin_offset = np.array([
            (volume_dims - 1) / 2,
            (volume_dims - 1) / 2,
            (volume_dims - 1) / 2
        ]) * resolution
        
        shifted_points = coord + origin_offset
        voxel_indices = np.floor(shifted_points / resolution)
        voxel_indices = np.clip(voxel_indices, 0, volume_dims - 1)
        voxel_centers = (voxel_indices - (volume_dims - 1) / 2) * resolution
        
        # Map each point to its unique voxel ID
        point_uniq_idx = uniq_idx[inv_idx]
        
        # ====================================================================
        # STEP 5: Assemble 18-feature format
        # ====================================================================
        # Expand opacity to column
        opacity = opacity[:, np.newaxis]
        
        # Concatenate: xyz(3) + rgb(3) + opacity(1) + scale(3) + quat(4) = 14
        gs_params = np.concatenate((coord, color, opacity, scale, quat), axis=1)
        
        # Add voxel info: voxel_centers(3) + uniq_idx(1) + gs_params(14) = 18
        point_uniq_idx = point_uniq_idx[:, np.newaxis]
        gs_full_params = np.concatenate((voxel_centers, point_uniq_idx, gs_params), axis=1)
        
        # Verify shape
        assert gs_full_params.shape == (40000, 18), \
            f"Expected shape (40000, 18), got {gs_full_params.shape}"
        
        # ====================================================================
        # STEP 6: Return as dictionary
        # ====================================================================
        return {
            'features': gs_full_params,      # [40000, 18]
            'segment_labels': segment,        # [40000] ScanNet72 labels
            'instance_labels': instance,      # [40000] instance labels
            'scene_idx': idx,
            'has_semantics': has_semantics,
            'num_categories': self.num_segment_categories  # Add this for easy access
        }
    
    def get_category_distribution(self, num_scenes=50):
        """
        Get the distribution of categories across scenes.
        
        Returns:
            category_counts: dict mapping category ID to count
            category_percentages: dict mapping category ID to percentage
        """
        category_counts = {i: 0 for i in range(self.num_segment_categories)}
        total_points = 0
        
        for i in tqdm(range(min(num_scenes, len(self.scene_dirs))), desc="Analyzing categories"):
            scene_dir = self.scene_dirs[i]
            segment_path = os.path.join(scene_dir, 'segment.npy')
            
            if os.path.exists(segment_path):
                segments = np.load(segment_path)
                valid_segments = segments[segments >= 0]
                
                for cat_id in valid_segments:
                    category_counts[int(cat_id)] += 1
                total_points += len(valid_segments)
        
        # Calculate percentages
        category_percentages = {}
        for cat_id, count in category_counts.items():
            if total_points > 0:
                category_percentages[cat_id] = (count / total_points) * 100
            else:
                category_percentages[cat_id] = 0.0
        
        return category_counts, category_percentages
    
    def print_category_summary(self, num_scenes=50):
        """Print a summary of category distribution."""
        print("\n" + "="*80)
        print("SCANNET72 CATEGORY DISTRIBUTION SUMMARY")
        print("="*80)
        
        category_counts, category_percentages = self.get_category_distribution(num_scenes)
        
        # Sort by frequency
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 20 most frequent categories (out of {self.num_segment_categories} total):")
        print("-" * 60)
        for i, (cat_id, count) in enumerate(sorted_categories[:20]):
            percentage = category_percentages[cat_id]
            print(f"  {i+1:2d}. Category {cat_id:2d}: {count:10,d} points ({percentage:6.2f}%)")
        
        # Missing categories
        missing_categories = [cat_id for cat_id, count in category_counts.items() if count == 0]
        print(f"\nMissing categories (never appear): {missing_categories}")
        
        # Cumulative coverage
        cumulative = 0
        print("\nCumulative coverage of top categories:")
        print("-" * 60)
        for i, (cat_id, count) in enumerate(sorted_categories[:15]):
            cumulative += category_percentages[cat_id]
            print(f"  Top {i+1:2d} categories: {cumulative:6.2f}% coverage")
        
        print(f"\n✓ 95% coverage achieved with top {self._get_coverage_threshold(sorted_categories, category_percentages, 95)} categories")
        print(f"✓ 99% coverage achieved with top {self._get_coverage_threshold(sorted_categories, category_percentages, 99)} categories")
        print("="*80)
    
    def _get_coverage_threshold(self, sorted_categories, category_percentages, threshold):
        """Get how many categories needed for threshold coverage."""
        cumulative = 0
        for i, (cat_id, _) in enumerate(sorted_categories):
            cumulative += category_percentages[cat_id]
            if cumulative >= threshold:
                return i + 1
        return len(sorted_categories)


# Test the dataset if run directly
if __name__ == "__main__":
    import sys
    
    print("Testing SceneSplat dataset with ScanNet72 semantic labels...")
    print()
    
    data_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train"
    
    # Allow command line override
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    dataset = gs_dataset(
        root=data_path,
        resol=200,
        random_permute=False,
        train=True,
        sampling_method='opacity',
        max_scenes=5
    )
    
    print(f"\nDataset size: {len(dataset)} scenes")
    
    # Test first sample
    print("\nLoading first sample...")
    batch_data = dataset[0]
    
    print(f"✓ Sample loaded successfully")
    print(f"\nShapes:")
    print(f"  features:         {batch_data['features'].shape}")
    print(f"  segment_labels:   {batch_data['segment_labels'].shape}")
    print(f"  instance_labels:  {batch_data['instance_labels'].shape}")
    print(f"  has_semantics:    {batch_data['has_semantics']}")
    print(f"  num_categories:   {batch_data['num_categories']}")
    
    # Check semantic label statistics
    seg_labels = batch_data['segment_labels']
    inst_labels = batch_data['instance_labels']
    
    seg_valid = (seg_labels != -1).sum()
    inst_valid = (inst_labels != -1).sum()
    
    unique_segs = np.unique(seg_labels[seg_labels != -1])
    unique_insts = np.unique(inst_labels[inst_labels != -1])
    
    print(f"\nSemantic label statistics for first scene:")
    print(f"  Segments: {seg_valid}/40000 valid ({100*seg_valid/40000:.1f}%)")
    print(f"  Unique segments: {len(unique_segs)} -> {sorted(unique_segs)}")
    print(f"  Instances: {inst_valid}/40000 valid ({100*inst_valid/40000:.1f}%)")
    print(f"  Unique instances: {len(unique_insts)}")
    
    # Print category summary
    dataset.print_category_summary(num_scenes=10)
    
    print("\n✓ Dataset test passed!")