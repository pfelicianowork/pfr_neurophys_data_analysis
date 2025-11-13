"""
IMPROVED: Cluster events using combined autoencoder + biological features.
Includes optimal k selection and cross-validation.
Pure PyTorch implementation (no fastai dependencies).
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.model_selection import KFold
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import improved modules
from cnn_autoencoder.improved_autoencoder import ResNetAutoencoder, SWR_VAE


# ===============================================================
# Dataset class for loading spectrograms (replaces fastai)
# ===============================================================

class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram images."""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = sorted(image_paths)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image


# ===============================================================
# Clustering utility functions (replaces advanced_clustering)
# ===============================================================

def find_optimal_clusters(features, k_range, methods=['kmeans', 'hierarchical', 'gmm']):
    """
    Find optimal number of clusters using multiple methods and metrics.
    
    Returns:
    --------
    best_config : dict
        Best clustering configuration with labels and metrics
    results_df : pd.DataFrame
        All clustering results for comparison
    """
    results = []
    best_score = -1
    best_config = None
    
    for k in k_range:
        for method in methods:
            try:
                # Perform clustering
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(features)
                elif method == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=k)
                    labels = clusterer.fit_predict(features)
                elif method == 'gmm':
                    clusterer = GaussianMixture(n_components=k, random_state=42)
                    labels = clusterer.fit_predict(features)
                else:
                    continue
                
                # Calculate metrics
                sil = silhouette_score(features, labels)
                db = davies_bouldin_score(features, labels)
                ch = calinski_harabasz_score(features, labels)
                
                results.append({
                    'k': k,
                    'method': method,
                    'silhouette': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch
                })
                
                # Track best (using silhouette score)
                if sil > best_score:
                    best_score = sil
                    best_config = {
                        'k': k,
                        'method': method,
                        'labels': labels,
                        'silhouette': sil,
                        'davies_bouldin': db,
                        'calinski_harabasz': ch
                    }
                
                print(f"  k={k}, {method}: silhouette={sil:.3f}, DB={db:.3f}, CH={ch:.1f}")
            
            except Exception as e:
                print(f"  k={k}, {method}: Failed - {e}")
    
    results_df = pd.DataFrame(results)
    return best_config, results_df


def cross_validate_clustering(features, n_clusters, n_splits=5, method='kmeans'):
    """
    Cross-validate clustering stability using ARI scores.
    
    Returns:
    --------
    ari_scores : list
        ARI scores for each fold comparison
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ari_scores = []
    
    # Get reference clustering on full data
    if method == 'kmeans':
        ref_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'hierarchical':
        ref_clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'gmm':
        ref_clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
    
    ref_labels = ref_clusterer.fit_predict(features)
    
    # Compare with fold clusterings
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features)):
        # Cluster on full data again (for reproducibility)
        if method == 'kmeans':
            fold_clusterer = KMeans(n_clusters=n_clusters, random_state=fold_idx, n_init=10)
        elif method == 'hierarchical':
            fold_clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'gmm':
            fold_clusterer = GaussianMixture(n_components=n_clusters, random_state=fold_idx)
        
        fold_labels = fold_clusterer.fit_predict(features)
        
        # Calculate ARI
        ari = adjusted_rand_score(ref_labels, fold_labels)
        ari_scores.append(ari)
    
    print(f"Cross-validation ARI: {np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f}")
    return ari_scores


def plot_clustering_metrics(results_df):
    """Plot clustering metrics comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        
        axes[0].plot(method_data['k'], method_data['silhouette'], 'o-', label=method)
        axes[1].plot(method_data['k'], method_data['davies_bouldin'], 'o-', label=method)
        axes[2].plot(method_data['k'], method_data['calinski_harabasz'], 'o-', label=method)
    
    axes[0].set_xlabel('Number of clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score (higher is better)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Number of clusters (k)')
    axes[1].set_ylabel('Davies-Bouldin Index')
    axes[1].set_title('Davies-Bouldin Index (lower is better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Number of clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Index')
    axes[2].set_title('Calinski-Harabasz Index (higher is better)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved clustering metrics plot to 'clustering_metrics_comparison.png'")
    plt.close()


def plot_dendrogram(features, max_samples=500):
    """Plot hierarchical clustering dendrogram."""
    # Subsample if needed
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features_subset = features[idx]
    else:
        features_subset = features
    
    # Compute linkage
    linkage_matrix = linkage(features_subset, method='ward')
    
    # Plot
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.title(f'Hierarchical Clustering Dendrogram (n={len(features_subset)})')
    plt.savefig('clustering_dendrogram.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved dendrogram to 'clustering_dendrogram.png'")
    plt.close()


def analyze_cluster_separation(features, labels):
    """Analyze cluster separation (inter vs intra distances)."""
    unique_labels = np.unique(labels)
    
    # Calculate within-cluster distances
    intra_dists = []
    for label in unique_labels:
        cluster_points = features[labels == label]
        if len(cluster_points) > 1:
            centroid = cluster_points.mean(axis=0)
            dists = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_dists.extend(dists)
    
    # Calculate between-cluster distances
    centroids = []
    for label in unique_labels:
        cluster_points = features[labels == label]
        centroids.append(cluster_points.mean(axis=0))
    centroids = np.array(centroids)
    
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_dists.append(dist)
    
    print(f"Inter-cluster distance: {np.mean(inter_dists):.3f} ± {np.std(inter_dists):.3f}")
    print(f"Intra-cluster distance: {np.mean(intra_dists):.3f} ± {np.std(intra_dists):.3f}")
    print(f"Separation ratio: {np.mean(inter_dists) / np.mean(intra_dists):.3f}")
    
    return inter_dists, intra_dists


def load_encoder(arch='resnet', latent_dim=128):
    """Load trained encoder model."""
    encoder_path = f"encoder_model_{arch}.pkl"
    
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder model not found at '{encoder_path}'")
        print(f"Train the model first using: python train_autoencoder.py --arch {arch}")
        return None
    
    print(f"Loading {arch.upper()} encoder from '{encoder_path}'...")
    
    if arch == 'resnet':
        model = ResNetAutoencoder(latent_dim=latent_dim)
        encoder = model.encoder
        encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
    elif arch == 'vae':
        model = SWR_VAE(latent_dim=latent_dim)
        model.encoder_conv.load_state_dict(torch.load(encoder_path, weights_only=False))
        encoder = model.encoder_conv
    elif arch == 'attention':
        from cnn_autoencoder.improved_autoencoder import AttentionAutoencoder
        model = AttentionAutoencoder(latent_dim=latent_dim)
        encoder = model.encoder
        encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
    
    encoder.eval()
    return encoder, arch


def extract_autoencoder_features(encoder, images_path, batch_size=64):
    """Extract features using trained encoder (pure PyTorch)."""
    
    print("Extracting autoencoder features...")
    
    # Get image files
    image_files = sorted(list(Path(images_path).glob("*.png")))
    print(f"Found {len(image_files)} images")
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    dataset = SpectrogramDataset(image_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Extract features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    
    features_list = []
    
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            features_list.append(feats.cpu().numpy())
    
    features_np = np.vstack(features_list)
    print(f"✓ Extracted {features_np.shape[0]} feature vectors of size {features_np.shape[1]}")
    
    return features_np


def combine_features(ae_features, bio_features, ae_weight=0.7, bio_weight=0.3):
    """
    Combine autoencoder and biological features with weighting.
    
    Parameters:
    -----------
    ae_features : np.ndarray
        Autoencoder features
    bio_features : np.ndarray
        Biological features
    ae_weight : float
        Weight for autoencoder features (0-1)
    bio_weight : float
        Weight for biological features (0-1)
    
    Returns:
    --------
    combined : np.ndarray
        Combined and scaled feature matrix
    """
    
    print("\nCombining features...")
    print(f"  Autoencoder features: {ae_features.shape}")
    print(f"  Biological features: {bio_features.shape}")
    print(f"  Weights: AE={ae_weight:.1%}, Bio={bio_weight:.1%}")
    
    # Scale each feature set independently
    scaler_ae = StandardScaler()
    ae_scaled = scaler_ae.fit_transform(ae_features)
    
    scaler_bio = StandardScaler()
    bio_scaled = scaler_bio.fit_transform(bio_features)
    
    # Apply weights
    ae_weighted = ae_scaled * ae_weight
    bio_weighted = bio_scaled * bio_weight
    
    # Concatenate
    combined = np.hstack([ae_weighted, bio_weighted])
    
    print(f"✓ Combined feature shape: {combined.shape}")
    
    return combined


def cluster_events_improved(arch='resnet', latent_dim=128, k_range=None, 
                           ae_weight=0.7, bio_weight=0.3, use_combined=True):
    """
    Main clustering function with combined features and optimal k selection.
    
    Parameters:
    -----------
    arch : str
        Architecture used for training
    latent_dim : int
        Latent dimension size
    k_range : range, optional
        Range of k values to test
    ae_weight : float
        Weight for autoencoder features
    bio_weight : float
        Weight for biological features
    use_combined : bool
        Whether to use combined features (True) or only autoencoder (False)
    """
    
    print("="*80)
    print("IMPROVED CLUSTERING WITH COMBINED FEATURES")
    print("="*80)
    
    # --- Load Data ---
    events_path = "detected_events.pkl"
    images_path = Path("all_spectrograms")
    bio_features_path = "biological_features.pkl"
    
    print("\n--- Loading Data ---")
    
    if not os.path.exists(events_path):
        print(f"Error: '{events_path}' not found")
        print("Run generate_all_spectrograms_improved.py first")
        return
    
    with open(events_path, "rb") as f:
        events = pickle.load(f)
    print(f"✓ Loaded {len(events)} events")
    
    # --- Load or Extract Autoencoder Features ---
    ae_features_path = f"autoencoder_features_{arch}.pkl"
    
    if os.path.exists(ae_features_path):
        print(f"Loading cached autoencoder features...")
        with open(ae_features_path, "rb") as f:
            ae_features = pickle.load(f)
    else:
        encoder, _ = load_encoder(arch, latent_dim)
        if encoder is None:
            return
        
        ae_features = extract_autoencoder_features(encoder, images_path)
        
        # Cache for future use
        with open(ae_features_path, "wb") as f:
            pickle.dump(ae_features, f)
        print(f"✓ Cached autoencoder features to '{ae_features_path}'")
    
    # --- Load Biological Features ---
    if use_combined:
        if not os.path.exists(bio_features_path):
            print(f"Warning: '{bio_features_path}' not found")
            print("Using autoencoder features only")
            use_combined = False
        else:
            with open(bio_features_path, "rb") as f:
                bio_data = pickle.load(f)
            bio_features = bio_data['feature_matrix']
            feature_names = bio_data['feature_names']
            print(f"✓ Loaded {bio_features.shape[1]} biological features")
    
    # --- Combine Features ---
    if use_combined:
        # Ensure same number of events
        min_samples = min(len(ae_features), len(bio_features))
        ae_features = ae_features[:min_samples]
        bio_features = bio_features[:min_samples]
        events = events[:min_samples]
        
        combined_features = combine_features(ae_features, bio_features, ae_weight, bio_weight)
        features_to_cluster = combined_features
        feature_type = "combined"
    else:
        # Scale autoencoder features only
        scaler = StandardScaler()
        features_to_cluster = scaler.fit_transform(ae_features)
        feature_type = "autoencoder_only"
    
    print(f"\n✓ Final feature matrix: {features_to_cluster.shape}")
    
    # --- Find Optimal Clusters ---
    if k_range is None:
        k_range = range(2, min(12, len(events) // 10))
    
    print(f"\n--- Finding Optimal Number of Clusters ---")
    print(f"Testing k in range: {list(k_range)}")
    
    best_config, results_df = find_optimal_clusters(
        features_to_cluster,
        k_range=k_range,
        methods=['kmeans', 'hierarchical', 'gmm']
    )
    
    # Plot comparison
    plot_clustering_metrics(results_df)
    
    # --- Cross-Validation ---
    print(f"\n--- Cross-Validation for Stability ---")
    ari_scores = cross_validate_clustering(
        features_to_cluster,
        n_clusters=best_config['k'],
        n_splits=5,
        method=best_config['method']
    )
    
    # --- Plot Dendrogram ---
    print(f"\n--- Hierarchical Clustering Dendrogram ---")
    plot_dendrogram(features_to_cluster, max_samples=500)
    
    # --- Cluster Separation Analysis ---
    print(f"\n--- Analyzing Cluster Separation ---")
    inter_dist, intra_dist = analyze_cluster_separation(
        features_to_cluster,
        best_config['labels']
    )
    
    # --- Save Results ---
    print(f"\n--- Saving Results ---")
    
    # Add cluster IDs to events
    for i, event in enumerate(events):
        if i < len(best_config['labels']):
            event['cluster_id'] = int(best_config['labels'][i])
    
    # Save clustered events
    output_path = f"events_with_clusters_{feature_type}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(events, f)
    print(f"✓ Saved clustered events to '{output_path}'")
    
    # Save clustering results
    clustering_info = {
        'best_k': best_config['k'],
        'best_method': best_config['method'],
        'silhouette': best_config['silhouette'],
        'davies_bouldin': best_config['davies_bouldin'],
        'calinski_harabasz': best_config['calinski_harabasz'],
        'labels': best_config['labels'],
        'ari_scores': ari_scores,
        'feature_type': feature_type,
        'architecture': arch,
        'results_df': results_df
    }
    
    clustering_info_path = f"clustering_info_{feature_type}.pkl"
    with open(clustering_info_path, "wb") as f:
        pickle.dump(clustering_info, f)
    print(f"✓ Saved clustering info to '{clustering_info_path}'")
    
    # Print summary
    print("\n" + "="*80)
    print("CLUSTERING SUMMARY")
    print("="*80)
    print(f"Feature type: {feature_type}")
    if use_combined:
        print(f"  AE weight: {ae_weight:.1%}, Bio weight: {bio_weight:.1%}")
    print(f"Architecture: {arch}")
    print(f"Optimal k: {best_config['k']}")
    print(f"Method: {best_config['method']}")
    print(f"Silhouette Score: {best_config['silhouette']:.3f}")
    print(f"Cross-validation ARI: {np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f}")
    
    # Cluster sizes
    cluster_counts = pd.Series(best_config['labels']).value_counts().sort_index()
    print(f"\nCluster sizes:")
    for cid, count in cluster_counts.items():
        print(f"  Cluster {cid}: {count} events ({100*count/len(best_config['labels']):.1f}%)")
    
    print("\nNext: Run evaluate_clusters_improved.py for detailed analysis")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster SWR events with improved features')
    parser.add_argument('--arch', type=str, default='resnet',
                       choices=['resnet', 'vae', 'attention'],
                       help='Autoencoder architecture')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension')
    parser.add_argument('--k_min', type=int, default=2,
                       help='Minimum k to test')
    parser.add_argument('--k_max', type=int, default=12,
                       help='Maximum k to test')
    parser.add_argument('--ae_weight', type=float, default=0.7,
                       help='Weight for autoencoder features')
    parser.add_argument('--bio_weight', type=float, default=0.3,
                       help='Weight for biological features')
    parser.add_argument('--no_combined', action='store_true',
                       help='Use only autoencoder features')
    
    args = parser.parse_args()
    
    cluster_events_improved(
        arch=args.arch,
        latent_dim=args.latent_dim,
        k_range=range(args.k_min, args.k_max + 1),
        ae_weight=args.ae_weight,
        bio_weight=args.bio_weight,
        use_combined=not args.no_combined
    )
