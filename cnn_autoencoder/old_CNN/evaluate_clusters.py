
import torch
from torch import nn
from fastai.vision.all import *
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

# --- Define the Autoencoder Architecture ---
# This is needed for fastai to be able to load the learner object.
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x): return self.decoder(self.encoder(x))

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
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x): return self.decoder(self.encoder(x))

# This function needs to be at the top level for the learner to be loaded
def get_x_y(p): return p

def evaluate_model_and_clusters():
    """
    Loads the trained model and clustered events to visualize the results.
    """
    learner_path = Path("autoencoder_learner.pkl")
    events_path = Path("events_with_clusters_analyzed.pkl")
    images_path = Path("all_spectrograms")

    if not all(p.exists() for p in [learner_path, events_path, images_path]):
        print("Error: Missing required files. Please run previous steps first.")
        return

    # --- Part 1: Evaluate Autoencoder Reconstruction ---
    print("--- 1. Visualizing Autoencoder Performance ---")
    
    # Try to load the full model weights
    full_model_path = Path("autoencoder_model.pth")
    if full_model_path.exists():
        print("Loading full model weights from autoencoder_model.pth")
        
        # Detect architecture
        state_dict = torch.load(full_model_path, weights_only=False)
        has_batchnorm = any('running_mean' in key or 'running_var' in key for key in state_dict.keys())
        
        if has_batchnorm:
            print("Detected: ImprovedAutoencoder")
            model = ImprovedAutoencoder()
        else:
            print("Detected: Original Autoencoder")
            model = Autoencoder()
        
        model.load_state_dict(state_dict)
        model.eval()
        model.cpu()
    else:
        print("Warning: Full model weights not found. Attempting to load from learner...")
        # Fallback to loading from learner
        learn = load_learner(learner_path)
        model = learn.model
        model.eval()
        model.cpu()

    # Re-create DataLoaders for show_results using DataBlock for autoencoder
    files = list(get_image_files(images_path))
    
    autoencoder_db = DataBlock(
        blocks=[ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW)],
        get_items=lambda p: files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_x=get_x_y,
        get_y=get_x_y,
        item_tfms=[Resize(128)],
        batch_tfms=[IntToFloatTensor()]  # This normalizes to [0, 1]
    )
    dls = autoencoder_db.dataloaders(images_path, bs=16, num_workers=0)

    print("Comparing original vs reconstructed spectrograms side-by-side...")
    
    try:
        b = dls.one_batch()
        print(f"Successfully fetched one batch. Input tensor shape: {b[0].shape}")
        
        # Debug: Check input statistics - convert to plain tensor first
        input_tensor = b[0].data if hasattr(b[0], 'data') else b[0]
        print(f"Input batch stats: min={float(input_tensor.min()):.4f}, max={float(input_tensor.max()):.4f}, mean={float(input_tensor.mean()):.4f}")
        
        # Get predictions
        with torch.no_grad():
            preds = model(input_tensor.cpu())
        
        # Debug: Check output statistics
        print(f"Output batch stats: min={float(preds.min()):.4f}, max={float(preds.max()):.4f}, mean={float(preds.mean()):.4f}")
        
        # Check if model weights are actually loaded
        print(f"First conv layer weight stats: min={float(model.encoder[0].weight.min()):.4f}, max={float(model.encoder[0].weight.max()):.4f}")
        
        # CRITICAL DIAGNOSTIC: Check if outputs are all the same value (indicates untrained model)
        output_std = float(preds.std())
        print(f"\nCRITICAL DIAGNOSTIC:")
        print(f"  Output std deviation: {output_std:.6f}")
        if output_std < 0.01:
            print("  ⚠️  WARNING: Output has very low variance - model may not be trained!")
            print("  This means all pixels have nearly the same value (likely all ~0.5 from sigmoid)")
            print("  Possible causes:")
            print("    1. Model weights didn't load correctly")
            print("    2. Autoencoder wasn't trained (only initialized)")
            print("    3. Training failed but process completed")
        else:
            print("  ✓ Output has reasonable variance - model appears to be generating different values")
        
        # Additional check: measure reconstruction diversity
        sample_diversity = []
        for i in range(min(5, preds.shape[0])):
            sample_diversity.append(float(preds[i].std()))
        print(f"  Per-sample std deviations: {[f'{x:.4f}' for x in sample_diversity]}")
        if all(s < 0.01 for s in sample_diversity):
            print("  ⚠️  All samples have flat outputs - model definitely not working!")

        # Take 6 examples for comparison
        n_examples = min(6, input_tensor.shape[0])
        originals = input_tensor[:n_examples].detach().cpu()
        reconstructed = preds[:n_examples].detach().cpu()
        
        # Calculate reconstruction errors for these examples
        recon_errors = []
        for i in range(n_examples):
            error = torch.mean((originals[i] - reconstructed[i]) ** 2).item()
            recon_errors.append(error)

        # Create side-by-side comparison
        fig, axes = plt.subplots(n_examples, 3, figsize=(12, n_examples * 2))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_examples):
            # Original
            axes[i, 0].imshow(originals[i].squeeze().numpy(), cmap='viridis', aspect='auto')
            axes[i, 0].set_title('Original' if i == 0 else '')
            axes[i, 0].set_ylabel(f'Example {i+1}', fontsize=10)
            axes[i, 0].axis('off')
            
            # Reconstructed
            axes[i, 1].imshow(reconstructed[i].squeeze().numpy(), cmap='viridis', aspect='auto')
            axes[i, 1].set_title('Reconstructed' if i == 0 else '')
            axes[i, 1].axis('off')
            
            # Difference map (absolute error)
            diff = np.abs(originals[i].squeeze().numpy() - reconstructed[i].squeeze().numpy())
            im = axes[i, 2].imshow(diff, cmap='hot', aspect='auto')
            axes[i, 2].set_title(f'Error' if i == 0 else '')
            axes[i, 2].axis('off')
            
            # Add MSE as text
            axes[i, 2].text(0.5, -0.1, f'MSE: {recon_errors[i]:.4f}', 
                          transform=axes[i, 2].transAxes, ha='center', fontsize=9)
        
        plt.suptitle("Autoencoder Reconstruction Quality Assessment", fontsize=14, y=0.995)
        plt.tight_layout()
        plt.show()
        
        # Calculate overall statistics
        print(f"\nReconstruction Error Statistics (from {n_examples} examples):")
        print(f"  Mean MSE: {np.mean(recon_errors):.4f}")
        print(f"  Std MSE:  {np.std(recon_errors):.4f}")
        print(f"  Min MSE:  {np.min(recon_errors):.4f}")
        print(f"  Max MSE:  {np.max(recon_errors):.4f}")
        
    except Exception as e:
        print(f"\nAn error occurred during reconstruction visualization: {e}")
        import traceback
        traceback.print_exc()
        model = None  # Set to None if loading failed


    # --- Part 2: Load and Prepare Cluster Data ---
    print("\n--- 2. Loading Cluster Data ---")
    with open(events_path, "rb") as f:
        events_with_clusters = pickle.load(f)
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(events_with_clusters)
    # Add the original index to map to the image file name
    df['original_index'] = df.index
    
    print(f"Loaded {len(df)} events with cluster assignments.")

    if 'cluster_id' not in df.columns:
        print("Error: 'cluster_id' not found in events file. Please re-run cluster_events.py")
        return

    cluster_ids = sorted(df['cluster_id'].unique())
    
    # --- Part 3: Load Features for Quantitative Analysis ---
    print("\n--- 3. Loading Features for Quantitative Metrics ---")
    features_path = Path("features.pkl")
    
    if not features_path.exists():
        print("Warning: 'features.pkl' not found. Skipping quantitative metrics.")
        features = None
    else:
        with open(features_path, "rb") as f:
            features = pickle.load(f)
        print(f"Loaded feature matrix: shape {features.shape}")
    
    # --- Part 4: Cluster Quality Metrics ---
    if features is not None:
        print("\n--- 4. Cluster Quality Metrics ---")
        cluster_labels = df['cluster_id'].values
        
        try:
            silhouette = silhouette_score(features, cluster_labels)
            davies_bouldin = davies_bouldin_score(features, cluster_labels)
            calinski = calinski_harabasz_score(features, cluster_labels)
            
            print(f"Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
            print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
            print(f"Calinski-Harabasz Score: {calinski:.1f} (higher is better)")
        except Exception as e:
            print(f"Could not compute cluster quality metrics: {e}")
    
    # --- Part 5: Cluster Statistics Summary ---
    print("\n--- 5. Cluster Statistics Summary ---")
    cluster_stats = df.groupby('cluster_id').size().to_frame(name='count')
    print(cluster_stats)
    
    # Show cluster size distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(cluster_stats.index, cluster_stats['count'])
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Number of Events')
    axes[0].set_title('Cluster Size Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].pie(cluster_stats['count'], labels=cluster_stats.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Cluster Proportions')
    
    plt.tight_layout()
    plt.show()
    
    # --- Part 6: Visualize Feature Space with PCA/t-SNE ---
    if features is not None:
        print("\n--- 6. Visualizing Feature Space ---")
        
        # PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA plot
        scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                 c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=axes[0], label='Cluster ID')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].set_title('PCA: Feature Space Colored by Cluster')
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE plot (if dataset not too large)
        if len(features) < 10000:
            print("Computing t-SNE (this may take a moment)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
            features_tsne = tsne.fit_transform(features)
            
            scatter = axes[1].scatter(features_tsne[:, 0], features_tsne[:, 1],
                                     c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=axes[1], label='Cluster ID')
            axes[1].set_xlabel('t-SNE Dimension 1')
            axes[1].set_ylabel('t-SNE Dimension 2')
            axes[1].set_title('t-SNE: Feature Space Colored by Cluster')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Dataset too large for t-SNE\n(>10,000 samples)', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('t-SNE: Skipped')
        
        plt.tight_layout()
        plt.show()
    
    # --- Part 7: Inter-Cluster Distance Matrix ---
    if features is not None:
        print("\n--- 7. Inter-Cluster Distance Matrix ---")
        
        # Compute cluster centroids
        centroids = []
        for cluster in cluster_ids:
            cluster_mask = cluster_labels == cluster
            centroid = features[cluster_mask].mean(axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        distance_matrix = cdist(centroids, centroids, metric='euclidean')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(distance_matrix, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Euclidean Distance')
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_yticks(range(len(cluster_ids)))
        ax.set_xticklabels(cluster_ids)
        ax.set_yticklabels(cluster_ids)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Cluster ID')
        ax.set_title('Inter-Cluster Distance Matrix')
        
        # Add text annotations
        for i in range(len(cluster_ids)):
            for j in range(len(cluster_ids)):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.1f}', 
                             ha='center', va='center', color='white', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    # --- Part 8: Average Spectrograms per Cluster ---
    print("\n--- 8. Average Spectrograms per Cluster ---")
    for cluster in cluster_ids:
        cluster_df = df[df['cluster_id'] == cluster]
        
        # Load all images for this cluster
        cluster_images = []
        for idx in cluster_df['original_index']:
            img_path = images_path / f"event_{idx:05d}.png"
            if img_path.exists():
                img = np.array(Image.open(img_path).convert('L'))
                cluster_images.append(img)
        
        if cluster_images:
            # Compute average and std
            avg_img = np.mean(cluster_images, axis=0)
            std_img = np.std(cluster_images, axis=0)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            im0 = axes[0].imshow(avg_img, cmap='gray')
            axes[0].set_title(f'Cluster {cluster} - Average')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(std_img, cmap='hot')
            axes[1].set_title(f'Cluster {cluster} - Std Dev')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            plt.suptitle(f'Cluster {cluster} Summary (n={len(cluster_images)})', fontsize=14)
            plt.tight_layout()
            plt.show()
    
    # --- Part 9: Reconstruction Error by Cluster ---
    print("\n--- 9. Reconstruction Error by Cluster ---")
    
    if model is None:
        print("Skipping Part 9: Model failed to load in Part 1.")
    else:
        reconstruction_errors = []
        
        model.eval()
        model.cpu()
        
        print("Computing reconstruction errors for all events...")
        with torch.no_grad():
            for i, idx in enumerate(df['original_index']):
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(df)} events...")
                    
                img_path = images_path / f"event_{idx:05d}.png"
                if img_path.exists():
                    try:
                        # Load image using PIL directly and convert to tensor
                        # Match the exact preprocessing from training
                        img_pil = Image.open(img_path).convert('L')
                        img_pil = img_pil.resize((128, 128))  # Resize to match training
                        img_array = np.array(img_pil, dtype=np.float32) / 255.0  # Normalize to [0,1]
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                        
                        # Get reconstruction
                        reconstructed = model(img_tensor.cpu())
                        
                        # Calculate MSE
                        error = torch.mean((img_tensor - reconstructed) ** 2).item()
                        reconstruction_errors.append(error)
                        
                    except Exception as e:
                        if i < 5:  # Only print first few errors
                            print(f"    Error processing event {idx}: {e}")
                        reconstruction_errors.append(np.nan)
                else:
                    reconstruction_errors.append(np.nan)
        
        df['reconstruction_error'] = reconstruction_errors
        
        # Check if we have valid errors
        valid_errors = df['reconstruction_error'].notna().sum()
        print(f"\nSuccessfully computed {valid_errors}/{len(df)} reconstruction errors.")
        
        if valid_errors > 0:
            # Plot by cluster
            fig, ax = plt.subplots(figsize=(12, 6))
            df.boxplot(column='reconstruction_error', by='cluster_id', ax=ax)
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Mean Squared Error')
            ax.set_title('Reconstruction Error Distribution by Cluster')
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            plt.show()
            
            # Print reconstruction error statistics
            print("\nReconstruction Error Statistics by Cluster:")
            recon_stats = df.groupby('cluster_id')['reconstruction_error'].agg(['mean', 'std', 'min', 'max'])
            print(recon_stats)
        else:
            print("No valid reconstruction errors computed. Skipping plots.")
    
    # --- Part 10: Sample Images from Each Cluster ---
    print("\n--- 10. Sample Images from Each Cluster ---")
    n_examples = 5 # Number of images to show per cluster

    for cluster in cluster_ids:
        print(f"\n--- Displaying examples for Cluster {cluster} ---")
        cluster_df = df[df['cluster_id'] == cluster]
        
        # Get a sample of image paths for this cluster
        sample_events = cluster_df.sample(min(n_examples, len(cluster_df)))
        
        fig, axes = plt.subplots(1, len(sample_events), figsize=(15, 3))
        if len(sample_events) == 1:
            axes = [axes] # Make it iterable
            
        fig.suptitle(f"Examples from Cluster {cluster} (Count: {len(cluster_df)})")
        
        for ax, (_, event_row) in zip(axes, sample_events.iterrows()):
            img_path = images_path/f"event_{event_row['original_index']:05d}.png"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Event {event_row['original_index']}")
                ax.axis('off')
            else:
                ax.set_title(f"Img not found")
                ax.axis('off')
        plt.show()

if __name__ == "__main__":
    evaluate_model_and_clusters()
