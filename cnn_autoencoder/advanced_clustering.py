"""
Advanced clustering methods with optimal k selection and validation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def compute_gap_statistic(features, labels, k, n_refs=10):
    """
    Compute gap statistic for cluster validation.
    
    The gap statistic compares the within-cluster dispersion to that expected
    under a null reference distribution.
    """
    # Compute within-cluster dispersion
    within_disp = 0
    for i in range(k):
        cluster_points = features[labels == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            within_disp += np.sum((cluster_points - centroid) ** 2)
    
    log_wk = np.log(within_disp + 1e-10)
    
    # Generate reference datasets
    log_wk_refs = []
    for _ in range(n_refs):
        # Generate uniform random data in same range as features
        random_features = np.random.uniform(
            features.min(axis=0), 
            features.max(axis=0), 
            features.shape
        )
        
        # Cluster random data
        random_labels = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(random_features)
        
        # Compute within-cluster dispersion for random data
        random_within_disp = 0
        for i in range(k):
            cluster_points = random_features[random_labels == i]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                random_within_disp += np.sum((cluster_points - centroid) ** 2)
        
        log_wk_refs.append(np.log(random_within_disp + 1e-10))
    
    gap = np.mean(log_wk_refs) - log_wk
    return gap


def find_optimal_clusters(features, k_range=range(2, 15), methods=['kmeans', 'hierarchical', 'gmm']):
    """
    Find optimal number of clusters using multiple metrics and methods.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    k_range : range
        Range of k values to test
    methods : list
        Clustering methods to try
    
    Returns:
    --------
    best_config : dict
        Best clustering configuration
    results_df : pd.DataFrame
        All results for analysis
    """
    
    results = []
    
    print("Testing clustering configurations...")
    for k in k_range:
        print(f"\nTesting k={k}...")
        
        # K-Means
        if 'kmeans' in methods:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=256)
            labels_km = kmeans.fit_predict(features)
            
            sil = silhouette_score(features, labels_km)
            db = davies_bouldin_score(features, labels_km)
            ch = calinski_harabasz_score(features, labels_km)
            gap = compute_gap_statistic(features, labels_km, k, n_refs=5)
            
            results.append({
                'k': k,
                'method': 'kmeans',
                'silhouette': sil,
                'davies_bouldin': db,
                'calinski_harabasz': ch,
                'gap_statistic': gap,
                'labels': labels_km,
                'model': kmeans
            })
            print(f"  K-Means: Sil={sil:.3f}, DB={db:.3f}, CH={ch:.1f}")
        
        # Hierarchical
        if 'hierarchical' in methods:
            hier = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels_hier = hier.fit_predict(features)
            
            sil = silhouette_score(features, labels_hier)
            db = davies_bouldin_score(features, labels_hier)
            ch = calinski_harabasz_score(features, labels_hier)
            gap = compute_gap_statistic(features, labels_hier, k, n_refs=5)
            
            results.append({
                'k': k,
                'method': 'hierarchical',
                'silhouette': sil,
                'davies_bouldin': db,
                'calinski_harabasz': ch,
                'gap_statistic': gap,
                'labels': labels_hier,
                'model': hier
            })
            print(f"  Hierarchical: Sil={sil:.3f}, DB={db:.3f}, CH={ch:.1f}")
        
        # GMM
        if 'gmm' in methods:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
            labels_gmm = gmm.fit_predict(features)
            
            sil = silhouette_score(features, labels_gmm)
            db = davies_bouldin_score(features, labels_gmm)
            ch = calinski_harabasz_score(features, labels_gmm)
            gap = compute_gap_statistic(features, labels_gmm, k, n_refs=5)
            
            results.append({
                'k': k,
                'method': 'gmm',
                'silhouette': sil,
                'davies_bouldin': db,
                'calinski_harabasz': ch,
                'gap_statistic': gap,
                'labels': labels_gmm,
                'model': gmm
            })
            print(f"  GMM: Sil={sil:.3f}, DB={db:.3f}, CH={ch:.1f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Normalize metrics for combined score
    df['silhouette_norm'] = (df['silhouette'] - df['silhouette'].min()) / (df['silhouette'].max() - df['silhouette'].min())
    df['db_norm'] = 1 - ((df['davies_bouldin'] - df['davies_bouldin'].min()) / (df['davies_bouldin'].max() - df['davies_bouldin'].min()))
    df['ch_norm'] = (df['calinski_harabasz'] - df['calinski_harabasz'].min()) / (df['calinski_harabasz'].max() - df['calinski_harabasz'].min())
    df['gap_norm'] = (df['gap_statistic'] - df['gap_statistic'].min()) / (df['gap_statistic'].max() - df['gap_statistic'].min())
    
    # Combined score (weighted average)
    df['combined_score'] = (
        df['silhouette_norm'] * 0.35 +
        df['db_norm'] * 0.25 +
        df['ch_norm'] * 0.25 +
        df['gap_norm'] * 0.15
    )
    
    # Find best configuration
    best_idx = df['combined_score'].idxmax()
    best = df.loc[best_idx]
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL CONFIGURATION:")
    print(f"  Method: {best['method']}")
    print(f"  k: {best['k']}")
    print(f"  Silhouette Score: {best['silhouette']:.3f}")
    print(f"  Davies-Bouldin Index: {best['davies_bouldin']:.3f}")
    print(f"  Calinski-Harabasz Score: {best['calinski_harabasz']:.1f}")
    print(f"  Gap Statistic: {best['gap_statistic']:.3f}")
    print(f"  Combined Score: {best['combined_score']:.3f}")
    print(f"{'='*60}\n")
    
    return best, df


def cross_validate_clustering(features, n_clusters, n_splits=5, method='kmeans'):
    """
    Assess clustering stability across data splits using cross-validation.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    n_clusters : int
        Number of clusters
    n_splits : int
        Number of CV folds
    method : str
        Clustering method
    
    Returns:
    --------
    ari_scores : list
        Adjusted Rand Index scores for each fold
    """
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ari_scores = []
    
    # Get reference clustering on full data
    if method == 'kmeans':
        full_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'hierarchical':
        full_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif method == 'gmm':
        full_model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    
    full_labels = full_model.fit_predict(features)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        # Cluster on training fold
        if method == 'kmeans':
            fold_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            fold_model.fit(features[train_idx])
            test_labels = fold_model.predict(features[test_idx])
        elif method == 'hierarchical':
            fold_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            test_labels = fold_model.fit_predict(features[test_idx])
        elif method == 'gmm':
            fold_model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
            fold_model.fit(features[train_idx])
            test_labels = fold_model.predict(features[test_idx])
        
        # Compare with full clustering
        ari = adjusted_rand_score(full_labels[test_idx], test_labels)
        ari_scores.append(ari)
        print(f"  Fold {fold+1}: ARI = {ari:.3f}")
    
    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    print(f"\nClustering Stability (ARI): {mean_ari:.3f} ± {std_ari:.3f}")
    
    if mean_ari > 0.8:
        print("✓ Excellent stability")
    elif mean_ari > 0.6:
        print("✓ Good stability")
    elif mean_ari > 0.4:
        print("⚠️  Moderate stability")
    else:
        print("⚠️  Poor stability - results may not be reliable")
    
    return ari_scores


def plot_clustering_metrics(results_df):
    """Plot clustering evaluation metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Silhouette Score
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        axes[0, 0].plot(method_data['k'], method_data['silhouette'], marker='o', label=method)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score (higher is better)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Davies-Bouldin Index
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        axes[0, 1].plot(method_data['k'], method_data['davies_bouldin'], marker='o', label=method)
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Davies-Bouldin Index')
    axes[0, 1].set_title('Davies-Bouldin Index (lower is better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Calinski-Harabasz Score
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        axes[1, 0].plot(method_data['k'], method_data['calinski_harabasz'], marker='o', label=method)
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Score (higher is better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Gap Statistic
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        axes[1, 1].plot(method_data['k'], method_data['gap_statistic'], marker='o', label=method)
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Gap Statistic')
    axes[1, 1].set_title('Gap Statistic (higher is better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved clustering metrics plot to 'clustering_metrics_comparison.png'")
    plt.show()


def plot_dendrogram(features, max_samples=500):
    """
    Plot hierarchical clustering dendrogram.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    max_samples : int
        Maximum samples to use (for computational efficiency)
    """
    
    # Subsample if needed
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_subset = features[indices]
    else:
        features_subset = features
    
    # Compute linkage
    print("Computing hierarchical clustering linkage...")
    Z = linkage(features_subset, method='ward')
    
    # Plot
    plt.figure(figsize=(15, 8))
    dendrogram(Z, no_labels=True)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.savefig('clustering_dendrogram.png', dpi=150, bbox_inches='tight')
    print("✓ Saved dendrogram to 'clustering_dendrogram.png'")
    plt.show()


def analyze_cluster_separation(features, labels):
    """
    Analyze how well-separated the clusters are.
    
    Computes inter-cluster and intra-cluster distances.
    """
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Compute centroids
    centroids = np.array([features[labels == i].mean(axis=0) for i in unique_labels])
    
    # Inter-cluster distances
    inter_distances = pdist(centroids, metric='euclidean')
    inter_dist_matrix = squareform(inter_distances)
    
    # Intra-cluster distances (average distance to centroid)
    intra_distances = []
    for i in unique_labels:
        cluster_points = features[labels == i]
        centroid = centroids[i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_distances.append(distances.mean())
    
    print("\n--- Cluster Separation Analysis ---")
    print(f"Mean inter-cluster distance: {inter_distances.mean():.3f}")
    print(f"Min inter-cluster distance: {inter_distances.min():.3f}")
    print(f"Max inter-cluster distance: {inter_distances.max():.3f}")
    print(f"\nMean intra-cluster distance: {np.mean(intra_distances):.3f}")
    print(f"Separation ratio (inter/intra): {inter_distances.mean() / np.mean(intra_distances):.3f}")
    
    # Plot inter-cluster distance matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(inter_dist_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Euclidean Distance')
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Cluster ID')
    ax.set_title('Inter-Cluster Distance Matrix')
    
    # Add text annotations
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                text = ax.text(j, i, f'{inter_dist_matrix[i, j]:.1f}',
                             ha='center', va='center', color='white', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('inter_cluster_distances.png', dpi=150, bbox_inches='tight')
    print("✓ Saved inter-cluster distance matrix to 'inter_cluster_distances.png'")
    plt.show()
    
    return inter_dist_matrix, intra_distances
