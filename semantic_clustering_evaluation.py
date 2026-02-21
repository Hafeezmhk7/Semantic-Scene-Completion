"""
Semantic Clustering Quality Evaluation
========================================

Measures how well semantic features cluster by category during training.

Key Metrics:
- Silhouette Score: Cohesion vs separation (-1 to 1, higher better)
- Calinski-Harabasz: Between/within cluster variance (higher better)
- Davies-Bouldin: Cluster similarity (lower better)

Usage in training loop:
    from semantic_clustering_evaluation import SemanticClusteringEvaluator
    
    evaluator = SemanticClusteringEvaluator()
    
    # During validation:
    metrics = evaluator.evaluate(
        semantic_features=features,    # [N, 32]
        labels=segment_labels,         # [N]
        positions=positions,           # [N, 3] for baseline
        colors=colors                  # [N, 3] for baseline
    )
    
    print(f"Silhouette (semantic): {metrics['silhouette_semantic']:.3f}")
    print(f"Silhouette (position): {metrics['silhouette_position']:.3f}")
"""

import numpy as np
import torch
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
# import seaborn as sns


class SemanticClusteringEvaluator:
    """
    Evaluate semantic feature quality by clustering metrics.
    
    Compares semantic features against baseline features (position, color)
    to validate that semantic learning is happening.
    """
    
    def __init__(
        self,
        subsample_size: int = 10000,
        min_class_size: int = 10,
        compute_per_class: bool = True
    ):
        """
        Args:
            subsample_size: Max points to use (for speed)
            min_class_size: Minimum samples per class to include
            compute_per_class: Whether to compute per-class metrics
        """
        self.subsample_size = subsample_size
        self.min_class_size = min_class_size
        self.compute_per_class = compute_per_class
        
        # Track metrics over time
        self.history = {
            'epoch': [],
            'silhouette_semantic': [],
            'silhouette_position': [],
            'silhouette_color': [],
            'ch_semantic': [],
            'ch_position': [],
            'db_semantic': [],
            'db_position': [],
        }
    
    def evaluate(
        self,
        semantic_features: np.ndarray,  # [N, 32]
        labels: np.ndarray,              # [N]
        positions: Optional[np.ndarray] = None,  # [N, 3]
        colors: Optional[np.ndarray] = None,     # [N, 3]
        epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.
        
        Returns:
            Dictionary with metrics:
            - silhouette_semantic: Silhouette for semantic features
            - silhouette_position: Silhouette for position (baseline)
            - silhouette_color: Silhouette for color (baseline)
            - ch_semantic: Calinski-Harabasz for semantic
            - db_semantic: Davies-Bouldin for semantic
            - per_class_scores: Per-class silhouette (if enabled)
        """
        # Convert to numpy if needed
        if torch.is_tensor(semantic_features):
            semantic_features = semantic_features.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if positions is not None and torch.is_tensor(positions):
            positions = positions.cpu().numpy()
        if colors is not None and torch.is_tensor(colors):
            colors = colors.cpu().numpy()
        
        # Remove invalid labels
        valid_mask = (labels >= 0) & (labels < 72)
        semantic_features = semantic_features[valid_mask]
        labels = labels[valid_mask]
        if positions is not None:
            positions = positions[valid_mask]
        if colors is not None:
            colors = colors[valid_mask]
        
        # Subsample for speed
        if len(semantic_features) > self.subsample_size:
            indices = np.random.choice(
                len(semantic_features),
                self.subsample_size,
                replace=False
            )
            semantic_features = semantic_features[indices]
            labels = labels[indices]
            if positions is not None:
                positions = positions[indices]
            if colors is not None:
                colors = colors[indices]
        
        # Filter rare classes
        class_counts = np.bincount(labels, minlength=72)
        valid_classes = np.where(class_counts >= self.min_class_size)[0]
        
        if len(valid_classes) < 2:
            print(f"⚠️  Not enough classes for clustering (found {len(valid_classes)})")
            return {}
        
        # Keep only points from valid classes
        valid_mask = np.isin(labels, valid_classes)
        semantic_features = semantic_features[valid_mask]
        labels = labels[valid_mask]
        if positions is not None:
            positions = positions[valid_mask]
        if colors is not None:
            colors = colors[valid_mask]
        
        # Compute metrics
        metrics = {}
        
        # Semantic features
        try:
            metrics['silhouette_semantic'] = silhouette_score(
                semantic_features, labels, metric='euclidean'
            )
            metrics['ch_semantic'] = calinski_harabasz_score(
                semantic_features, labels
            )
            metrics['db_semantic'] = davies_bouldin_score(
                semantic_features, labels
            )
        except Exception as e:
            print(f"⚠️  Error computing semantic metrics: {e}")
            metrics['silhouette_semantic'] = 0.0
            metrics['ch_semantic'] = 0.0
            metrics['db_semantic'] = 999.0
        
        # Baseline: Position
        if positions is not None:
            try:
                metrics['silhouette_position'] = silhouette_score(
                    positions, labels, metric='euclidean'
                )
                metrics['ch_position'] = calinski_harabasz_score(
                    positions, labels
                )
                metrics['db_position'] = davies_bouldin_score(
                    positions, labels
                )
            except Exception as e:
                print(f"⚠️  Error computing position metrics: {e}")
                metrics['silhouette_position'] = 0.0
        
        # Baseline: Color
        if colors is not None:
            try:
                metrics['silhouette_color'] = silhouette_score(
                    colors, labels, metric='euclidean'
                )
                metrics['ch_color'] = calinski_harabasz_score(
                    colors, labels
                )
                metrics['db_color'] = davies_bouldin_score(
                    colors, labels
                )
            except Exception as e:
                print(f"⚠️  Error computing color metrics: {e}")
                metrics['silhouette_color'] = 0.0
        
        # Per-class analysis
        if self.compute_per_class:
            try:
                sample_scores = silhouette_samples(semantic_features, labels)
                per_class = {}
                for class_id in valid_classes:
                    mask = (labels == class_id)
                    per_class[int(class_id)] = float(sample_scores[mask].mean())
                metrics['per_class_silhouette'] = per_class
            except Exception as e:
                print(f"⚠️  Error computing per-class metrics: {e}")
        
        # Track history
        if epoch is not None:
            self.history['epoch'].append(epoch)
            self.history['silhouette_semantic'].append(
                metrics.get('silhouette_semantic', 0.0)
            )
            self.history['silhouette_position'].append(
                metrics.get('silhouette_position', 0.0)
            )
            self.history['silhouette_color'].append(
                metrics.get('silhouette_color', 0.0)
            )
            self.history['ch_semantic'].append(
                metrics.get('ch_semantic', 0.0)
            )
            self.history['ch_position'].append(
                metrics.get('ch_position', 0.0)
            )
            self.history['db_semantic'].append(
                metrics.get('db_semantic', 999.0)
            )
            self.history['db_position'].append(
                metrics.get('db_position', 999.0)
            )
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a nice format."""
        print("\n" + "="*70)
        print("SEMANTIC CLUSTERING QUALITY")
        print("="*70)
        
        # Silhouette scores
        print("\nSilhouette Score (range: -1 to 1, higher better):")
        print(f"  Semantic features: {metrics.get('silhouette_semantic', 0.0):>6.3f}", end="")
        if metrics.get('silhouette_semantic', 0) > 0.3:
            print(" ✅ Excellent!")
        elif metrics.get('silhouette_semantic', 0) > 0.2:
            print(" ✅ Good")
        elif metrics.get('silhouette_semantic', 0) > 0.1:
            print(" → Moderate")
        else:
            print(" ⚠️  Poor")
        
        if 'silhouette_position' in metrics:
            print(f"  Position baseline: {metrics['silhouette_position']:>6.3f}", end="")
            diff = metrics.get('silhouette_semantic', 0) - metrics['silhouette_position']
            if diff > 0.1:
                print(f" (semantic +{diff:.3f} better!) ✅")
            elif diff > 0:
                print(f" (semantic +{diff:.3f} better)")
            else:
                print(f" (semantic {diff:.3f} worse) ⚠️")
        
        if 'silhouette_color' in metrics:
            print(f"  Color baseline:    {metrics['silhouette_color']:>6.3f}")
        
        # Calinski-Harabasz
        print("\nCalinski-Harabasz Index (higher better):")
        print(f"  Semantic features: {metrics.get('ch_semantic', 0.0):>8.1f}", end="")
        if metrics.get('ch_semantic', 0) > 500:
            print(" ✅ Excellent!")
        elif metrics.get('ch_semantic', 0) > 200:
            print(" ✅ Good")
        else:
            print(" → Moderate")
        
        if 'ch_position' in metrics:
            print(f"  Position baseline: {metrics['ch_position']:>8.1f}")
        
        # Davies-Bouldin
        print("\nDavies-Bouldin Index (lower better):")
        print(f"  Semantic features: {metrics.get('db_semantic', 999.0):>6.3f}", end="")
        if metrics.get('db_semantic', 999) < 1.5:
            print(" ✅ Excellent!")
        elif metrics.get('db_semantic', 999) < 2.0:
            print(" ✅ Good")
        else:
            print(" → Moderate")
        
        if 'db_position' in metrics:
            print(f"  Position baseline: {metrics['db_position']:>6.3f}")
        
        print("="*70)
        
        # Interpretation
        sil_sem = metrics.get('silhouette_semantic', 0.0)
        sil_pos = metrics.get('silhouette_position', 0.0)
        
        print("\n📊 INTERPRETATION:")
        if sil_sem > 0.3 and sil_sem > sil_pos + 0.1:
            print("  ✅ EXCELLENT: Semantic features show strong class clustering!")
            print("     Model is learning semantic structure beyond position.")
        elif sil_sem > 0.2 and sil_sem > sil_pos:
            print("  ✅ GOOD: Semantic features cluster better than position.")
            print("     Model is learning some semantic awareness.")
        elif sil_sem > sil_pos:
            print("  → MODERATE: Slight improvement over position baseline.")
            print("     Semantic learning is happening but weak.")
        else:
            print("  ⚠️  WARNING: Semantic features not better than position!")
            print("     Contrastive loss may not be learning semantics.")
        
        # Per-class analysis
        if 'per_class_silhouette' in metrics:
            per_class = metrics['per_class_silhouette']
            best_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)[:5]
            worst_classes = sorted(per_class.items(), key=lambda x: x[1])[:5]
            
            print("\n📈 Best clustering classes:")
            for class_id, score in best_classes:
                print(f"     Class {class_id:2d}: {score:>6.3f}")
            
            print("\n📉 Worst clustering classes:")
            for class_id, score in worst_classes:
                print(f"     Class {class_id:2d}: {score:>6.3f}")
        
        print()
    
    def plot_history(self, save_path: str = "semantic_clustering_history.png"):
        """Plot clustering metrics over training."""
        if len(self.history['epoch']) == 0:
            print("No history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = self.history['epoch']
        
        # Silhouette score
        ax = axes[0, 0]
        ax.plot(epochs, self.history['silhouette_semantic'],
               'b-', linewidth=2, label='Semantic', marker='o')
        if any(self.history['silhouette_position']):
            ax.plot(epochs, self.history['silhouette_position'],
                   'r--', linewidth=2, label='Position', marker='s')
        if any(self.history['silhouette_color']):
            ax.plot(epochs, self.history['silhouette_color'],
                   'g--', linewidth=2, label='Color', marker='^')
        ax.axhline(0.3, color='gray', linestyle=':', label='Good threshold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontweight='bold')
        ax.set_title('Clustering Quality (higher = better)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calinski-Harabasz
        ax = axes[0, 1]
        ax.plot(epochs, self.history['ch_semantic'],
               'b-', linewidth=2, label='Semantic', marker='o')
        if any(self.history['ch_position']):
            ax.plot(epochs, self.history['ch_position'],
                   'r--', linewidth=2, label='Position', marker='s')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Index', fontweight='bold')
        ax.set_title('Between/Within Cluster Variance (higher = better)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Davies-Bouldin
        ax = axes[1, 0]
        ax.plot(epochs, self.history['db_semantic'],
               'b-', linewidth=2, label='Semantic', marker='o')
        if any(self.history['db_position']):
            ax.plot(epochs, self.history['db_position'],
                   'r--', linewidth=2, label='Position', marker='s')
        ax.axhline(2.0, color='gray', linestyle=':', label='Good threshold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Index', fontweight='bold')
        ax.set_title('Cluster Similarity (lower = better)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Improvement over position
        ax = axes[1, 1]
        if any(self.history['silhouette_position']):
            improvements = [
                sem - pos for sem, pos in zip(
                    self.history['silhouette_semantic'],
                    self.history['silhouette_position']
                )
            ]
            ax.plot(epochs, improvements, 'g-', linewidth=2, marker='o')
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.axhline(0.1, color='green', linestyle=':', label='Significant improvement')
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Silhouette Improvement', fontweight='bold')
            ax.set_title('Semantic vs Position Baseline', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text annotation
            if improvements:
                final_improvement = improvements[-1]
                ax.text(0.05, 0.95, 
                       f'Final: {final_improvement:+.3f}',
                       transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved clustering history plot: {save_path}")
        plt.close()


def extract_semantic_features_from_model(
    model,
    batch,
    device='cuda'
):
    """
    Extract semantic features from model for a batch.
    
    Args:
        model: Your AlignedShapeLatentPerceiver model
        batch: Batch dict with 'features', 'segment_labels', etc.
        device: Device to run on
    
    Returns:
        semantic_features: [N, 32] numpy array
        labels: [N] numpy array
        positions: [N, 3] numpy array
        colors: [N, 3] numpy array
    """
    model.eval()
    with torch.no_grad():
        # Get input
        input_features = batch['features'].to(device)  # [B, 40000, 18]
        segment_labels = batch['segment_labels']  # [B, 40000]
        
        # Forward pass
        reconstruction, _, semantic_out = model(input_features, return_semantic=True)
        
        # semantic_out is dict with 'features' [B, 40000, 32]
        semantic_features = semantic_out['features']  # [B, 40000, 32]
        
        # Flatten batch dimension
        B = semantic_features.shape[0]
        semantic_features = semantic_features.reshape(-1, 32)  # [B*40000, 32]
        segment_labels = segment_labels.reshape(-1)  # [B*40000]
        
        # Extract position and color from input
        # Assuming format: [voxel_centers(3), uniq_idx(1), xyz(3), rgb(3), opacity(1), scale(3), quat(4)]
        positions = input_features[:, :, 4:7].reshape(-1, 3)  # [B*40000, 3]
        colors = input_features[:, :, 7:10].reshape(-1, 3)  # [B*40000, 3]
        
        # Convert to numpy
        semantic_features = semantic_features.cpu().numpy()
        segment_labels = segment_labels.cpu().numpy()
        positions = positions.cpu().numpy()
        colors = colors.cpu().numpy()
    
    return semantic_features, segment_labels, positions, colors


# Example usage in training script
if __name__ == "__main__":
    print("Testing semantic clustering evaluator...")
    
    # Simulate some features
    np.random.seed(42)
    n_samples = 5000
    n_classes = 20
    
    # Create fake semantic features with some structure
    semantic_features = np.random.randn(n_samples, 32)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Add class-specific structure
    for class_id in range(n_classes):
        mask = labels == class_id
        # Each class has a different center
        center = np.random.randn(32) * 5
        semantic_features[mask] += center
    
    # Create position features (spatial clustering)
    positions = np.random.randn(n_samples, 3)
    for class_id in range(n_classes):
        mask = labels == class_id
        center = np.random.randn(3) * 3
        positions[mask] += center
    
    # Evaluate
    evaluator = SemanticClusteringEvaluator()
    metrics = evaluator.evaluate(
        semantic_features=semantic_features,
        labels=labels,
        positions=positions,
        epoch=0
    )
    
    evaluator.print_metrics(metrics)