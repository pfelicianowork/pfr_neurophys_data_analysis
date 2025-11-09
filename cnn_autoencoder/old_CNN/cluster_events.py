
import torch
from torch import nn
from fastai.vision.all import *
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


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
    """Run basic sanity checks on extracted features before clustering."""
    if features.ndim != 2:
        print(f"Error: Expected 2D feature matrix from {source}, got shape {features.shape}.")
        return False

    n_samples, n_dims = features.shape
    print(f"{source}: shape=({n_samples}, {n_dims})")

    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    if n_nan or n_inf:
        print(f"Error: Found {n_nan} NaNs and {n_inf} infs in {source}. Regenerate spectrograms or retrain the encoder.")
        return False

    global_min = float(np.min(features))
    global_max = float(np.max(features))
    global_std = float(np.std(features))
    print(f"{source}: min={global_min:.4f}, max={global_max:.4f}, std={global_std:.4e}")

    if np.isclose(global_std, 0.0):
        print("Error: Features have ~zero variance; clustering would produce empty plots. Investigate encoder outputs before proceeding.")
        return False

    zero_rows = np.all(np.isclose(features, 0.0), axis=1).sum()
    if zero_rows:
        pct_zero = 100.0 * zero_rows / n_samples
        print(f"Warning: {zero_rows} feature vectors ({pct_zero:.2f}%) are entirely zeros.")
        if pct_zero > 80:
            print("Too many zero vectors detected. Aborting clustering until features are regenerated.")
            return False

    return True


def cluster_events():
    """
    Loads the trained encoder, extracts features from spectrograms, and clusters them.
    """
    # --- 2. Load Input Data ---
    events_path = "detected_events.pkl"
    images_path = Path("all_spectrograms")
    encoder_path = "encoder_model.pkl"

    if not all(os.path.exists(p) for p in [events_path, images_path, encoder_path]):
        print("Error: Missing required files. Please ensure the following exist:")
        print(f"- '{events_path}' (from generate_all_spectrograms.py)")
        print(f"- '{images_path}' directory (from generate_all_spectrograms.py)")
        print(f"- '{encoder_path}' (from train_autoencoder.py)")
        return

    print("Loading detected events and image files...")
    with open(events_path, "rb") as f:
        events = pickle.load(f)
    
    # Get a sorted list of image files to ensure order is consistent
    img_files = get_image_files(images_path).sorted()

    if len(events) != len(img_files):
        print(f"Warning: Mismatch between number of events ({len(events)}) and images ({len(img_files)}).")
        print("Please regenerate the data to ensure consistency.")
        # We can proceed but the mapping might be incorrect

    # --- 3. Load Model and Extract Features ---
    print("Loading trained encoder and extracting features...")

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

    # Create a DataBlock for inference (no labels)
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW)),
        get_items=lambda p: p, # Pass the list of files directly
        item_tfms=Resize(128)
    )
    
    # Create a DataLoader for the test set
    dls = dblock.dataloaders(img_files, bs=64, num_workers=0)
    test_dl = dls.test_dl(img_files)

    # Create a Learner for inference
    learn = Learner(dls, encoder_model, loss_func=noop) # No loss function needed for inference

    # Get predictions (the feature vectors)
    features, _ = learn.get_preds(dl=test_dl)
    features_np = features.numpy()
    print(f"Extracted {features_np.shape[0]} feature vectors of size {features_np.shape[1]}.")

    if not _validate_feature_matrix(features_np, "Extracted features"):
        print("Aborting clustering run. Inspect autoencoder training or regenerate the feature cache.")
        return

    # --- 4. Perform Clustering ---
    n_clusters = 5 # This is a hyperparameter you can tune
    print(f"\nPerforming K-Means clustering with k={n_clusters}...")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init='auto')
    cluster_labels = kmeans.fit_predict(features_np)

    # --- 5. Save and Report Results ---
    print("Assigning cluster IDs to events and saving results...")
    
    # Add the cluster ID to each event dictionary
    for i, event in enumerate(events):
        if i < len(cluster_labels):
            event['cluster_id'] = cluster_labels[i]

    # Save the updated events list
    output_path = "events_with_clusters.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(events, f)

    print(f"\nSuccessfully saved events with cluster IDs to '{output_path}'.")

    # Print a summary of the cluster distribution
    print("\n--- Clustering Results ---")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("Number of events per cluster:")
    print(cluster_counts)
    print("\nProcess complete. You can now analyze the events in each cluster.")

if __name__ == "__main__":
    cluster_events()
