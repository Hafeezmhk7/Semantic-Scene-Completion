#!/usr/bin/env python3
"""
SceneSplat-7K Dataset Inspector
Analyzes .npy files for each scene and provides statistics
"""

import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

def inspect_scene(scene_path):
    """Inspect a single scene and return statistics"""
    scene_name = scene_path.name
    
    # Expected files
    files = {
        'coord': 'coord.npy',
        'color': 'color.npy',
        'scale': 'scale.npy',
        'quat': 'quat.npy',
        'opacity': 'opacity.npy',
        'normal': 'normal.npy',
        'instance': 'instance.npy',
        'segment': 'segment.npy'
    }
    
    stats = {'scene_name': scene_name}
    
    # Load each file and gather statistics
    for key, filename in files.items():
        filepath = scene_path / filename
        
        if filepath.exists():
            try:
                data = np.load(filepath)
                stats[key] = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std())
                }
                
                # Special check for number of Gaussians
                if key == 'coord':
                    stats['num_gaussians'] = len(data)
                    
            except Exception as e:
                stats[key] = {'error': str(e)}
        else:
            stats[key] = {'status': 'missing'}
    
    return stats


def print_scene_stats(stats, detailed=False):
    """Pretty print scene statistics"""
    print(f"\n{'='*80}")
    print(f"Scene: {stats['scene_name']}")
    print(f"{'='*80}")
    
    if 'num_gaussians' in stats:
        print(f"Number of Gaussians: {stats['num_gaussians']:,}")
    
    print(f"\n{'Attribute':<12} {'Shape':<20} {'Type':<12} {'Min':>12} {'Max':>12} {'Mean':>12}")
    print('-' * 80)
    
    # Core attributes for 3D Gaussians
    core_attrs = ['coord', 'color', 'scale', 'quat', 'opacity']
    extra_attrs = ['normal', 'instance', 'segment']
    
    for attr in core_attrs:
        if attr in stats and isinstance(stats[attr], dict):
            data = stats[attr]
            if 'shape' in data:
                print(f"{attr:<12} {str(data['shape']):<20} {data['dtype']:<12} "
                      f"{data['min']:>12.4f} {data['max']:>12.4f} {data['mean']:>12.4f}")
            elif 'error' in data:
                print(f"{attr:<12} ERROR: {data['error']}")
            elif 'status' in data:
                print(f"{attr:<12} {data['status'].upper()}")
    
    if detailed:
        print(f"\n{'Extra Attributes:'}")
        print('-' * 80)
        for attr in extra_attrs:
            if attr in stats and isinstance(stats[attr], dict):
                data = stats[attr]
                if 'shape' in data:
                    print(f"{attr:<12} {str(data['shape']):<20} {data['dtype']:<12} "
                          f"{data['min']:>12.4f} {data['max']:>12.4f} {data['mean']:>12.4f}")
                elif 'status' in data:
                    print(f"{attr:<12} {data['status'].upper()}")


def analyze_split(split_path, split_name, max_scenes=None, detailed=False):
    """Analyze all scenes in a split"""
    print(f"\n{'#'*80}")
    print(f"# Analyzing {split_name.upper()} Split")
    print(f"{'#'*80}")
    
    scene_folders = sorted([d for d in split_path.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')])
    
    print(f"\nFound {len(scene_folders)} scenes in {split_name} split")
    
    if max_scenes:
        print(f"Showing first {max_scenes} scenes (use --all to see all)")
        scene_folders = scene_folders[:max_scenes]
    
    # Collect statistics
    all_stats = []
    gaussian_counts = []
    
    for scene_path in scene_folders:
        stats = inspect_scene(scene_path)
        all_stats.append(stats)
        
        if 'num_gaussians' in stats:
            gaussian_counts.append(stats['num_gaussians'])
        
        # Print individual scene stats
        print_scene_stats(stats, detailed=detailed)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY for {split_name.upper()} split")
    print(f"{'='*80}")
    print(f"Total scenes analyzed: {len(all_stats)}")
    
    if gaussian_counts:
        print(f"\nGaussian Count Statistics:")
        print(f"  Min:     {min(gaussian_counts):>12,}")
        print(f"  Max:     {max(gaussian_counts):>12,}")
        print(f"  Mean:    {np.mean(gaussian_counts):>12,.0f}")
        print(f"  Median:  {np.median(gaussian_counts):>12,.0f}")
        print(f"  Std:     {np.std(gaussian_counts):>12,.0f}")
        
        # Histogram
        print(f"\nGaussian Count Distribution:")
        bins = [0, 500_000, 1_000_000, 1_500_000, 2_000_000, np.inf]
        labels = ['0-500K', '500K-1M', '1M-1.5M', '1.5M-2M', '2M+']
        hist, _ = np.histogram(gaussian_counts, bins=bins)
        
        for label, count in zip(labels, hist):
            if count > 0:
                bar = '█' * int(count / len(gaussian_counts) * 50)
                print(f"  {label:<12} {count:>4} scenes {bar}")
    
    # Check data consistency
    print(f"\nData Format Checks:")
    
    # Check if colors are normalized
    color_ranges = [s['color'] for s in all_stats if 'color' in s and 'max' in s['color']]
    if color_ranges:
        max_color = max([c['max'] for c in color_ranges])
        if max_color > 1.0:
            print(f"  ⚠️  Colors NOT normalized (max: {max_color:.2f}) - need to divide by 255")
        else:
            print(f"  ✅ Colors normalized to [0, 1]")
    
    # Check opacity range
    opacity_ranges = [s['opacity'] for s in all_stats if 'opacity' in s and 'max' in s['opacity']]
    if opacity_ranges:
        max_opacity = max([o['max'] for o in opacity_ranges])
        min_opacity = min([o['min'] for o in opacity_ranges])
        print(f"  Opacity range: [{min_opacity:.4f}, {max_opacity:.4f}]")
    
    return all_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect SceneSplat-7K dataset')
    parser.add_argument('--root', type=str, 
                       default='/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs',
                       help='Root directory of SceneSplat-7K dataset')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'val', 'all'],
                       default='train', help='Which split to analyze')
    parser.add_argument('--max-scenes', type=int, default=5,
                       help='Maximum number of scenes to show details for (None for all)')
    parser.add_argument('--all', action='store_true',
                       help='Show all scenes (overrides --max-scenes)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed attributes (normals, segments, etc.)')
    
    args = parser.parse_args()
    
    root_path = Path(args.root)
    
    if not root_path.exists():
        print(f"❌ Error: Dataset path not found: {root_path}")
        sys.exit(1)
    
    print(f"\n🔍 SceneSplat-7K Dataset Inspector")
    print(f"📁 Dataset root: {root_path}")
    
    max_scenes = None if args.all else args.max_scenes
    
    # Analyze requested splits
    if args.split == 'all':
        for split in ['train', 'test', 'val']:
            split_path = root_path / split
            if split_path.exists():
                analyze_split(split_path, split, max_scenes, args.detailed)
    else:
        split_path = root_path / args.split
        if split_path.exists():
            analyze_split(split_path, args.split, max_scenes, args.detailed)
        else:
            print(f"❌ Error: Split not found: {split_path}")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"✅ Inspection complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()