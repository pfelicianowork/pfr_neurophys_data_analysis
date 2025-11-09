import torch
from torch import nn
from fastai.vision.all import *
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


# Define both architectures
class Autoencoder(nn.Module):
    """Original autoencoder architecture."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )

class ImprovedAutoencoder(nn.Module):
    """Improved autoencoder with BatchNorm and LeakyReLU."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(),
        )


def _validate_feature_matrix(features: np.ndarray, source: str) -> bool:
    """Check for NaNs, infs, and degenerate variance before clustering."""
    if features.ndim != 2:
        print(f"Error: Expected 2D feature matrix from {source}, got shape {features.shape}.")
        return False

    n_samples, n_dims = features.shape
    print(f"{source}: shape=({n_samples}, {n_dims})")

    # Basic sanity checks.
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    if n_nan or n_inf:
        print(f"Error: Found {n_nan} NaNs and {n_inf} infs in feature matrix. Delete 'features.pkl' and rerun extraction.")
        return False

    global_min = float(np.min(features))
    global_max = float(np.max(features))
    global_std = float(np.std(features))
    print(f"{source}: min={global_min:.4f}, max={global_max:.4f}, std={global_std:.4e}")

    if np.isclose(global_std, 0.0):
        print("Error: Feature matrix variance is ~0. Check encoder training or regenerate spectrograms.")
        return False

    # Detect rows that are entirely zero (common corruption mode).
    zero_rows = np.all(np.isclose(features, 0.0), axis=1).sum()
    if zero_rows:
        pct_zero = 100.0 * zero_rows / n_samples
        print(f"Warning: {zero_rows} feature vectors ({pct_zero:.2f}%) are all zeros.")
        if pct_zero > 80:
            print("Aborting: Too many zero rows for reliable clustering. Regenerate features before proceeding.")
            return False

    return True

def analyze_clusters():
    """
    Loads the trained encoder, extracts features, and then evaluates and compares
    different clustering algorithms (K-Means, DBSCAN, GMM) using quantitative metrics.
    """
    # --- 1. Load Input Data and Model ---
    events_path = "detected_events.pkl"
    images_path = Path("all_spectrograms")
    encoder_path = "encoder_model.pkl"
    features_path = "features.pkl"

    if os.path.exists(features_path):
        print("Loading pre-computed features...")
        with open(features_path, "rb") as f:
            features_np = pickle.load(f)
    else:
        print("Could not find pre-computed features. Extracting from images...")
        if not all(os.path.exists(p) for p in [events_path, images_path, encoder_path]):
            print("Error: Missing required files for feature extraction.")
            return

        dblock = DataBlock(
            blocks=(ImageBlock(cls=PILImageBW)),
            get_items=lambda p: p, # Pass the list of files directly
            item_tfms=Resize(128)
        )
        img_files = get_image_files(images_path).sorted()
        dls = dblock.dataloaders(img_files, bs=64, num_workers=0)
        test_dl = dls.test_dl(img_files)

        # Detect which architecture to use
        encoder_state = torch.load(encoder_path, weights_only=False)
        has_batchnorm = any('running_mean' in key or 'running_var' in key for key in encoder_state.keys())
        
        if has_batchnorm:
            print("Detected: ImprovedAutoencoder encoder")
            encoder_model = ImprovedAutoencoder().encoder
        else:
            print("Detected: Original Autoencoder encoder")
            encoder_model = Autoencoder().encoder
        
        encoder_model.load_state_dict(encoder_state)
        encoder_model.eval()

        learn = Learner(dls, encoder_model, loss_func=noop)
        features, _ = learn.get_preds(dl=test_dl)
        features_np = features.numpy()

        print(f"Extracted {features_np.shape[0]} feature vectors. Saving to '{features_path}'...")
        with open(features_path, "wb") as f:
            pickle.dump(features_np, f)

    if not _validate_feature_matrix(features_np, "Feature cache"):
        print("Aborting clustering due to invalid features. Delete 'features.pkl' and rerun feature extraction.")
        return

    # --- 1a. Scale the features ---
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)
    print("Features scaled.")

    # --- 2. Quantitative Evaluation of K-Means ---
    print("\n--- Evaluating K-Means Clustering ---")
    k_range = range(2, 11)
    inertias = []
    silhouette_scores = []

    print("Testing k in range:", list(k_range))
    for k in k_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init='auto').fit(features_scaled)
        inertias.append(kmeans.inertia_)
        # Silhouette score is computationally expensive, so we sample the data
        if features_scaled.shape[0] > 10000:
            sample_indices = np.random.choice(features_scaled.shape[0], 10000, replace=False)
            score = silhouette_score(features_scaled[sample_indices], kmeans.predict(features_scaled[sample_indices]))
        else:
            score = silhouette_score(features_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={score:.3f}")

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('K-Means Elbow Method')

    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('K-Means Silhouette Scores')
    
    plt.suptitle("Quantitative Evaluation of K-Means")
    plt.show()

    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nBest k based on Silhouette Score: {best_k}")

    # --- 3. Exploring Other Clustering Algorithms ---

    # K-Means with the best k
    print(f"\n--- Final K-Means (k={best_k}) ---")
    kmeans_final = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=256, n_init='auto').fit(features_scaled)
    kmeans_labels = kmeans_final.labels_
    print("Cluster distribution:")
    print(pd.Series(kmeans_labels).value_counts().sort_index())

    # DBSCAN
    print("\n--- Evaluating DBSCAN Clustering ---")
    # Note: DBSCAN's `eps` parameter is highly sensitive and may need tuning.
    # We'll start with a common heuristic, but this is the main parameter to adjust.
    dbscan = DBSCAN(eps=15, min_samples=5, n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(features_scaled)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = np.sum(dbscan_labels == -1)
    print(f"DBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points.")
    if n_clusters_dbscan > 1:
        score = silhouette_score(features_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
        print(f"Silhouette Score (excluding noise): {score:.3f}")
        print("Cluster distribution:")
        print(pd.Series(dbscan_labels[dbscan_labels != -1]).value_counts().sort_index())

    # Gaussian Mixture Models (GMM)
    print("\n--- Evaluating GMM Clustering ---")
    # We use the best k from our K-Means analysis as the number of components for GMM
    # Convert to float64 for numerical stability
    gmm = GaussianMixture(n_components=best_k, random_state=42, reg_covar=1e-6).fit(features_scaled.astype(np.float64))
    gmm_labels = gmm.predict(features_scaled)
    gmm_score = silhouette_score(features_scaled, gmm_labels)
    print(f"GMM with {best_k} components: Silhouette Score={gmm_score:.3f}")
    print("Cluster distribution:")
    print(pd.Series(gmm_labels).value_counts().sort_index())

    # --- 4. Save Final Results ---
    # For this example, we'll save the results from the best K-Means model.
    # This could be changed to DBSCAN or GMM based on the results above.
    print(f"\nSaving events with cluster IDs from K-Means (k={best_k})...")
    with open(events_path, "rb") as f:
        events = pickle.load(f)

    for i, event in enumerate(events):
        if i < len(kmeans_labels):
            event['cluster_id'] = kmeans_labels[i]

    output_path = "events_with_clusters_analyzed.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(events, f)
    
    print(f"Successfully saved clustered events to '{output_path}'.")
    print("\nAnalysis complete. You can now visualize the new clusters with 'evaluate_clusters.py'.")


if __name__ == "__main__":
    analyze_clusters()