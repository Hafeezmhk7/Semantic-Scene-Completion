"""
Enhanced script to parse ALL CAN3TOK training logs and upload to W&B
Handles multiple naming patterns and formats
"""

import re
import wandb
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime


def infer_config_from_filename(filename: str) -> Dict:
    """Infer configuration from filename patterns."""
    config = {}
    
    # Extract from filename
    name_lower = filename.lower()
    
    # Semantic mode
    if 'hidden' in name_lower:
        config['semantic_mode'] = 'hidden'
        config['semantic_enabled'] = True
    elif 'geometric' in name_lower:
        config['semantic_mode'] = 'geometric'
        config['semantic_enabled'] = True
    elif 'attention' in name_lower:
        config['semantic_mode'] = 'attention'
        config['semantic_enabled'] = True
    elif 'baseline' in name_lower:
        config['semantic_mode'] = 'none'
        config['semantic_enabled'] = False
    else:
        config['semantic_mode'] = 'unknown'
        config['semantic_enabled'] = None
    
    # Number of Gaussians
    if '10k' in name_lower or '10_' in name_lower:
        config['num_gaussians'] = 10000
    elif '40k' in name_lower or '40_' in name_lower:
        config['num_gaussians'] = 40000
    else:
        config['num_gaussians'] = None
    
    # Balanced sampling
    if 'bala' in name_lower or 'balanced' in name_lower:
        config['balanced_sampling'] = True
    else:
        config['balanced_sampling'] = False
    
    # Cross-batch
    if 'cb' in name_lower or 'cross' in name_lower or 'batch' in name_lower:
        config['cross_batch'] = True
    else:
        config['cross_batch'] = False
    
    # Segment weight
    match = re.search(r'sw(\d+)', name_lower)
    if match:
        config['segment_weight'] = float(match.group(1))
    
    # Job ID
    match = re.search(r'(\d{8})', filename)
    if match:
        config['job_id'] = match.group(1)
    
    return config


def parse_config_from_log(log_content: str, filename: str) -> Dict:
    """Extract configuration from log file content and filename."""
    config = {}
    
    # Start with filename-based inference
    config.update(infer_config_from_filename(filename))
    
    # Job ID (from content overrides filename)
    match = re.search(r'Job ID:\s*(\d+)', log_content)
    if match:
        config['job_id'] = match.group(1)
    
    # Semantic mode (from content overrides filename)
    match = re.search(r'Semantic Mode \(effective\):\s*(\w+)', log_content)
    if match:
        config['semantic_mode'] = match.group(1).lower()
        config['semantic_enabled'] = match.group(1).lower() != 'none'
    else:
        match = re.search(r'Mode:\s*(HIDDEN|GEOMETRIC|ATTENTION|NONE)', log_content, re.IGNORECASE)
        if match:
            config['semantic_mode'] = match.group(1).lower()
            config['semantic_enabled'] = match.group(1).lower() != 'none'
    
    # Batch size
    match = re.search(r'Batch Size:\s*(\d+)', log_content)
    if match:
        config['batch_size'] = int(match.group(1))
    
    # Learning rate
    match = re.search(r'Learning Rate:\s*([\d.e-]+)', log_content)
    if match:
        config['learning_rate'] = float(match.group(1))
    
    # KL weight
    match = re.search(r'KL Weight:\s*([\d.e-]+)', log_content)
    if match:
        config['kl_weight'] = float(match.group(1))
    
    # Segment weight
    match = re.search(r'Segment \(β\):\s*([\d.]+)', log_content)
    if match:
        config['segment_weight'] = float(match.group(1))
    
    # Instance weight
    match = re.search(r'Instance \(γ\):\s*([\d.]+)', log_content)
    if match:
        config['instance_weight'] = float(match.group(1))
    
    # Temperature
    match = re.search(r'Temperature.*?:\s*([\d.]+)', log_content)
    if match:
        config['temperature'] = float(match.group(1))
    
    # Subsample
    match = re.search(r'Subsample:\s*(\d+)', log_content)
    if match:
        config['subsample'] = int(match.group(1))
    
    # Training scenes
    match = re.search(r'Training Scenes:\s*(\d+)', log_content)
    if match:
        config['train_scenes'] = int(match.group(1))
    
    # Val scenes
    match = re.search(r'Val Scenes:\s*(\d+)', log_content)
    if match:
        config['val_scenes'] = int(match.group(1))
    
    # Sampling method
    match = re.search(r'Sampling:\s*(\w+)', log_content)
    if match:
        config['sampling_method'] = match.group(1).lower()
    
    # Check for balanced sampling in content
    if 'balanced' in log_content.lower():
        config['balanced_sampling'] = True
    
    return config


def parse_training_metrics(log_content: str) -> List[Dict]:
    """Extract training metrics from log."""
    metrics = []
    
    # Pattern 1: With semantic loss
    pattern1 = r'Epoch (\d+)/\d+ \| Loss: ([\d.]+) \| Recon: ([\d.]+) \| KL: ([\d.]+) \| Semantic: ([\d.]+)'
    
    for match in re.finditer(pattern1, log_content):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        recon = float(match.group(3))
        kl = float(match.group(4))
        semantic = float(match.group(5))
        
        metric = {
            'epoch': epoch,
            'train/loss': loss,
            'train/recon_loss': recon,
            'train/kl_loss': kl,
            'train/semantic_loss': semantic,
        }
        metrics.append(metric)
    
    # Pattern 2: Without semantic loss (for runs already processed)
    if not metrics:
        pattern2 = r'Epoch (\d+)/\d+ \| Loss: ([\d.]+) \| Recon: ([\d.]+) \| KL: ([\d.]+)(?:\s|$)'
        
        for match in re.finditer(pattern2, log_content):
            epoch = int(match.group(1))
            loss = float(match.group(2))
            recon = float(match.group(3))
            kl = float(match.group(4))
            
            metric = {
                'epoch': epoch,
                'train/loss': loss,
                'train/recon_loss': recon,
                'train/kl_loss': kl,
            }
            metrics.append(metric)
    
    return metrics


def parse_validation_metrics(log_content: str) -> List[Dict]:
    """Extract validation metrics from log."""
    metrics = []
    
    # Pattern for validation blocks
    val_pattern = r'VALIDATION \(Epoch (\d+)\).*?L2 Error: ([\d.]+) ± ([\d.]+).*?Failure Rate: ([\d.]+)%'
    
    for match in re.finditer(val_pattern, log_content, re.DOTALL):
        epoch = int(match.group(1))
        l2_error = float(match.group(2))
        l2_std = float(match.group(3))
        failure_rate = float(match.group(4))
        
        metric = {
            'epoch': epoch,
            'val/l2_error': l2_error,
            'val/l2_std': l2_std,
            'val/failure_rate': failure_rate,
        }
        metrics.append(metric)
    
    return metrics


def parse_final_results(log_content: str) -> Dict:
    """Extract final results from log."""
    results = {}
    
    # Final L2
    match = re.search(r'Final L2:\s*([\d.]+)', log_content)
    if match:
        results['final_l2'] = float(match.group(1))
    
    # Best L2
    match = re.search(r'Best L2:\s*([\d.]+)', log_content)
    if match:
        results['best_l2'] = float(match.group(1))
    
    # Training duration
    match = re.search(r'Duration:\s*(\d+) minutes', log_content)
    if match:
        results['duration_minutes'] = int(match.group(1))
    
    return results


def create_run_name(config: Dict, filename: str) -> str:
    """Create descriptive run name from config."""
    parts = []
    
    # Job ID or filename
    if 'job_id' in config:
        parts.append(f"job_{config['job_id']}")
    else:
        # Use filename without extension
        parts.append(Path(filename).stem)
    
    # Semantic mode
    if config.get('semantic_enabled'):
        mode = config.get('semantic_mode', 'unknown')
        parts.append(mode)
        if 'segment_weight' in config:
            parts.append(f"beta{config['segment_weight']}")
    else:
        parts.append('baseline')
    
    # Num Gaussians
    if 'num_gaussians' in config and config['num_gaussians']:
        parts.append(f"{config['num_gaussians']//1000}k")
    
    # Special flags
    if config.get('balanced_sampling'):
        parts.append('balanced')
    if config.get('cross_batch'):
        parts.append('cross-batch')
    
    return '_'.join(parts)


def create_tags(config: Dict, filename: str) -> List[str]:
    """Create tags for W&B run."""
    tags = []
    
    # Semantic status
    if config.get('semantic_enabled'):
        tags.append('semantic')
        if 'semantic_mode' in config:
            tags.append(f"mode_{config['semantic_mode']}")
    else:
        tags.append('baseline')
    
    # Num Gaussians
    if 'num_gaussians' in config and config['num_gaussians']:
        tags.append(f"{config['num_gaussians']//1000}k_gaussians")
    
    # Balanced sampling
    if config.get('balanced_sampling'):
        tags.append('balanced_sampling')
    
    # Cross-batch
    if config.get('cross_batch'):
        tags.append('cross_batch')
    
    # Upload date
    tags.append(f"uploaded_{datetime.now().strftime('%Y%m%d')}")
    
    return tags


def upload_to_wandb(log_file: Path, project: str = "Can3Tok-Semantic", 
                   entity: Optional[str] = None, additional_tags: Optional[List[str]] = None,
                   dry_run: bool = False):
    """Parse log file and upload to W&B."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {log_file.name}")
    print(f"{'='*70}")
    
    # Read log file
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # Parse configuration
    config = parse_config_from_log(log_content, log_file.name)
    print(f"✓ Parsed config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Parse metrics
    train_metrics = parse_training_metrics(log_content)
    val_metrics = parse_validation_metrics(log_content)
    final_results = parse_final_results(log_content)
    
    print(f"✓ Parsed {len(train_metrics)} training steps")
    print(f"✓ Parsed {len(val_metrics)} validation steps")
    
    if not train_metrics and not val_metrics:
        print(f"⚠️  No metrics found - skipping")
        return None
    
    # Create run name and tags
    run_name = create_run_name(config, log_file.name)
    run_tags = create_tags(config, log_file.name)
    if additional_tags:
        run_tags.extend(additional_tags)
    
    print(f"✓ Run name: {run_name}")
    print(f"✓ Tags: {', '.join(run_tags)}")
    
    if dry_run:
        print(f"✓ DRY RUN - would upload {len(train_metrics)} epochs")
        return run_name
    
    # Initialize W&B run
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            tags=run_tags,
            resume='allow',
            id=run_name.replace('/', '_')  # Use run name as ID for consistency
        )
        
        print(f"✓ Initialized W&B run: {run_name}")
    except Exception as e:
        print(f"❌ Error initializing W&B: {e}")
        return None
    
    # Merge metrics by epoch
    all_metrics = {}
    
    # Add training metrics
    for metric in train_metrics:
        epoch = metric['epoch']
        if epoch not in all_metrics:
            all_metrics[epoch] = {'epoch': epoch}
        all_metrics[epoch].update(metric)
    
    # Add validation metrics
    for metric in val_metrics:
        epoch = metric['epoch']
        if epoch not in all_metrics:
            all_metrics[epoch] = {'epoch': epoch}
        all_metrics[epoch].update(metric)
    
    # Log metrics in order
    print("✓ Logging metrics to W&B...")
    for epoch in sorted(all_metrics.keys()):
        wandb.log(all_metrics[epoch], step=epoch)
    
    # Log final results as summary
    if final_results:
        wandb.run.summary.update(final_results)
    
    print(f"✓ Logged {len(all_metrics)} epochs to W&B")
    print(f"✓ Run URL: {run.url}")
    
    # Finish run
    run.finish()
    
    return run_name


def main():
    parser = argparse.ArgumentParser(description='Upload CAN3TOK training logs to W&B')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory containing log files')
    parser.add_argument('--log_files', type=str, nargs='+',
                       help='Specific log files to upload')
    parser.add_argument('--pattern', type=str, default='*.out',
                       help='File pattern to match (e.g., "*.out", "can3tok_*.out")')
    parser.add_argument('--project', type=str, default='Can3Tok-Semantic',
                       help='W&B project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='W&B entity (username or team)')
    parser.add_argument('--tags', type=str, nargs='*',
                       help='Additional tags for runs')
    parser.add_argument('--dry-run', action='store_true',
                       help='Parse files but do not upload')
    
    args = parser.parse_args()
    
    # Find log files
    if args.log_files:
        log_files = [Path(f) for f in args.log_files]
    else:
        log_dir = Path(args.log_dir)
        log_files = sorted(log_dir.glob(args.pattern))
    
    if not log_files:
        print(f"❌ No log files found matching pattern: {args.pattern}")
        return
    
    print(f"\n{'='*70}")
    print(f"Found {len(log_files)} log files to upload")
    print(f"Project: {args.project}")
    if args.entity:
        print(f"Entity: {args.entity}")
    if args.dry_run:
        print("DRY RUN MODE - No actual uploads")
    print(f"{'='*70}\n")
    
    # Upload each log file
    successful = []
    failed = []
    
    for log_file in log_files:
        try:
            run_name = upload_to_wandb(
                log_file,
                project=args.project,
                entity=args.entity,
                additional_tags=args.tags,
                dry_run=args.dry_run
            )
            if run_name:
                successful.append((log_file.name, run_name))
            else:
                failed.append(log_file.name)
        except Exception as e:
            print(f"❌ Error processing {log_file.name}: {e}")
            failed.append(log_file.name)
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"UPLOAD {'SIMULATION' if args.dry_run else 'COMPLETE'}")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful uploads:")
        for filename, run_name in successful:
            print(f"  ✓ {filename} → {run_name}")
    
    if failed:
        print(f"\nFailed files:")
        for filename in failed:
            print(f"  ❌ {filename}")
    
    if not args.dry_run and successful:
        dashboard_url = f"https://wandb.ai/{args.entity or 'your-username'}/{args.project}"
        print(f"\n✓ View your runs at: {dashboard_url}")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()