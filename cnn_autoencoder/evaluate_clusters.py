"""
IMPROVED: Comprehensive evaluation and visualization of clustered SWR events
with biological validation.
"""

import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse

# Import validation modules
from biological_validation import (
    validate_clusters_biologically,
    compare_clusters_statistically,
    plot_cluster_characteristics,
    export_validation_report
)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def evaluate_clusters_improved(feature_type='combined'):
    """
    Main evaluation function with comprehensive analysis.
    
    Parameters:
    -----------
    feature_type : str
        Type of features used: 'combined' or 'autoencoder_only'
    """
    
    print("="*80)
    print("COMPREHENSIVE CLUSTER EVALUATION")
    print("="*80)
    
    # --- Load Data ---
    events_path = Path(f"events_with_clusters_{feature_type}.pkl")
    clustering_info_path = Path(f"clustering_info_{feature_type}.pkl")
    images_path = Path("all_spectrograms")
    
    print("\n--- Loading Data ---")
    
    if not events_path.exists():
        print(f"Error: '{events_path}' not found")
        print("Run cluster_events_improved.py first")
        return
    
    with open(events_path, "rb") as f:
        events = pickle.load(f)
    print(f"✓ Loaded {len(events)} clustered events")
    
    if clustering_info_path.exists():
        with open(clustering_info_path, "rb") as f:
            clustering_info = pickle.load(f)
        print(f"✓ Loaded clustering info")
        print(f"  Method: {clustering_info['best_method']}")
        print(f"  k: {clustering_info['best_k']}")
        print(f"  Silhouette: {clustering_info['silhouette']:.3f}")
    else:
        clustering_info = None
    
    # --- Create DataFrame ---
    print("\n--- Preparing Data ---")
    df = pd.DataFrame(events)
    df['original_index'] = df.index
    
    if 'cluster_id' not in df.columns:
        print("Error: No cluster_id column found")
        return
    
    cluster_ids = sorted(df['cluster_id'].unique())
    print(f"✓ Found {len(cluster_ids)} clusters")
    
    # --- Biological Validation ---
    print("\n" + "="*80)
    print("PART 1: BIOLOGICAL VALIDATION")
    print("="*80)
    
    validation_results = validate_clusters_biologically(df)
    
    # --- Statistical Comparison ---
    print("\n" + "="*80)
    print("PART 2: STATISTICAL COMPARISON")
    print("="*80)
    
    comparison_results = compare_clusters_statistically(df)
    
    # --- Visualizations ---
    print("\n" + "="*80)
    print("PART 3: VISUALIZATIONS")
    print("="*80)
    
    # 3.1: Cluster characteristics
    print("\n3.1: Plotting cluster characteristics...")
    plot_cluster_characteristics(df, save_prefix=f'cluster_{feature_type}')
    
    # 3.2: Feature space visualization
    print("\n3.2: Visualizing feature space...")
    
    # Load features for visualization
    if feature_type == 'combined':
        # Try to load combined features
        try:
            with open("biological_features.pkl", "rb") as f:
                bio_data = pickle.load(f)
            features = bio_data['feature_matrix'][:len(df)]
        except:
            features = None
    else:
        try:
            arch = clustering_info.get('architecture', 'resnet')
            with open(f"autoencoder_features_{arch}.pkl", "rb") as f:
                features = pickle.load(f)[:len(df)]
        except:
            features = None
    
    if features is not None:
        labels = df['cluster_id'].values
        
        # PCA visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        scatter = axes[0].scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=30
        )
        plt.colorbar(scatter, ax=axes[0], label='Cluster ID')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].set_title('PCA: Feature Space Projection')
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE if dataset not too large
        if len(features) < 5000:
            print("  Computing t-SNE (may take a moment)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
            features_tsne = tsne.fit_transform(features)
            
            scatter = axes[1].scatter(
                features_tsne[:, 0],
                features_tsne[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=30
            )
            plt.colorbar(scatter, ax=axes[1], label='Cluster ID')
            axes[1].set_xlabel('t-SNE Dimension 1')
            axes[1].set_ylabel('t-SNE Dimension 2')
            axes[1].set_title('t-SNE: Feature Space Projection')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, f'Dataset too large for t-SNE\n({len(features)} samples)',
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'feature_space_{feature_type}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved feature space plot to 'feature_space_{feature_type}.png'")
        plt.show()
    
    # 3.3: Average spectrograms per cluster
    print("\n3.3: Computing average spectrograms per cluster...")
    
    for cluster_id in cluster_ids:
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        # Load images for this cluster
        cluster_images = []
        for idx in cluster_df['original_index'].values[:100]:  # Limit to 100 for memory
            img_path = images_path / f"event_{idx:05d}.png"
            if img_path.exists():
                img = np.array(Image.open(img_path).convert('L'))
                cluster_images.append(img)
        
        if len(cluster_images) > 0:
            avg_img = np.mean(cluster_images, axis=0)
            std_img = np.std(cluster_images, axis=0)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            im0 = axes[0].imshow(avg_img, cmap='viridis', aspect='auto')
            axes[0].set_title(f'Cluster {cluster_id} - Average Spectrogram')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Frequency')
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(std_img, cmap='hot', aspect='auto')
            axes[1].set_title(f'Cluster {cluster_id} - Std Deviation')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Frequency')
            plt.colorbar(im1, ax=axes[1])
            
            plt.suptitle(f'Cluster {cluster_id} Summary (n={len(cluster_df)})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'cluster_{cluster_id}_avg_spectrogram.png', dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved average spectrogram for cluster {cluster_id}")
            plt.show()
    
    # 3.4: Sample images from each cluster
    print("\n3.4: Showing sample images from each cluster...")
    
    n_samples = 6
    for cluster_id in cluster_ids:
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        sample_events = cluster_df.sample(min(n_samples, len(cluster_df)))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        fig.suptitle(f"Sample Events from Cluster {cluster_id} (n={len(cluster_df)})", fontsize=14)
        
        for ax, (_, event_row) in zip(axes, sample_events.iterrows()):
            img_path = images_path / f"event_{event_row['original_index']:05d}.png"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img, cmap='viridis', aspect='auto')
                
                # Add event info
                duration_ms = event_row.get('duration', 0) * 1000
                power = event_row.get('ripple_power', 0)
                ax.set_title(f"Event {event_row['original_index']}\n"
                           f"Dur: {duration_ms:.0f}ms, Pwr: {power:.2f}",
                           fontsize=9)
            else:
                ax.set_title(f"Image not found")
            ax.axis('off')
        
        # Hide unused subplots
        for ax in axes[len(sample_events):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'cluster_{cluster_id}_samples.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved samples for cluster {cluster_id}")
        plt.show()
    
    # --- Export Report ---
    print("\n--- Exporting Report ---")
    report_filename = f'cluster_validation_report_{feature_type}.txt'
    export_validation_report(validation_results, df, filename=report_filename)
    
    # --- Final Summary ---
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nCluster quality summary:")
    for cluster_name, results in validation_results.items():
        if 'quality_score' in results:
            score = results['quality_score']
            print(f"  {cluster_name}: {score*100:.0f}% ", end="")
            if score >= 0.8:
                print("✓✓ Excellent")
            elif score >= 0.6:
                print("✓ Good")
            else:
                print("⚠️  Needs review")
    
    print(f"\nGenerated files:")
    print(f"  - feature_space_{feature_type}.png")
    print(f"  - cluster_*_characteristics_boxplots.png")
    print(f"  - cluster_*_summary.png")
    for cid in cluster_ids:
        print(f"  - cluster_{cid}_avg_spectrogram.png")
        print(f"  - cluster_{cid}_samples.png")
    print(f"  - {report_filename}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustered SWR events')
    parser.add_argument('--feature_type', type=str, default='combined',
                       choices=['combined', 'autoencoder_only'],
                       help='Type of features used for clustering')
    
    args = parser.parse_args()
    
    evaluate_clusters_improved(feature_type=args.feature_type)
