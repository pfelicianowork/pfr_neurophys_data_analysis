"""
IMPROVED: Cluster events using combined autoencoder + biological features.
Includes optimal k selection and cross-validation.
"""

import torch
from torch import nn
from fastai.vision.all import *
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse

# Import improved modules
from improved_autoencoder import ResNetAutoencoder, SWR_VAE
from advanced_clustering import (
    find_optimal_clusters,
    cross_validate_clustering,
    plot_clustering_metrics,
    plot_dendrogram,
    analyze_cluster_separation
)


def load_encoder(arch='resnet', latent_dim=128):
    """Load trained encoder model."""
    encoder_path = f"encoder_model_{arch}.pkl"
    
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder model not found at '{encoder_path}'")
        print(f"Train the model first using: python train_autoencoder_improved.py --arch {arch}")
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
        from improved_autoencoder import AttentionAutoencoder
        model = AttentionAutoencoder(latent_dim=latent_dim)
        encoder = model.encoder
        encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
    
    encoder.eval()
    return encoder, arch


def extract_autoencoder_features(encoder, images_path, batch_size=64):
    """Extract features using trained encoder."""
    
    print("Extracting autoencoder features...")
    
    # Create DataBlock for inference
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW)),
        get_items=lambda p: p,
        item_tfms=Resize(128)
    )
    
    img_files = get_image_files(images_path).sorted()
    dls = dblock.dataloaders(img_files, bs=batch_size, num_workers=0)
    test_dl = dls.test_dl(img_files)
    
    # Extract features
    features_list = []
    
    with torch.no_grad():
        for batch in test_dl:
            imgs = batch[0]
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
