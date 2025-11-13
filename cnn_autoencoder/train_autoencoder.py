import numpy as np
from glob import glob
from PIL import Image
def load_encoder(encoder_path, arch='resnet', latent_dim=128, device=None):
    """
    Load a trained encoder model.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if arch == 'resnet':
        from cnn_autoencoder.improved_autoencoder import ResNetAutoencoder
        model = ResNetAutoencoder(latent_dim=latent_dim)
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder = model.encoder
    elif arch == 'vae':
        from cnn_autoencoder.improved_autoencoder import SWR_VAE
        model = SWR_VAE(latent_dim=latent_dim)
        model.encoder_conv.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder = model.encoder_conv
    else:
        raise ValueError(f"Unknown arch: {arch}")
    encoder.eval()
    encoder.to(device)
    return encoder

def encode_spectrograms(encoder, image_dir, n_events=None, img_size=128, device=None):
    """
    Encode all spectrogram images in a directory using the encoder.
    """
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    img_files = sorted(glob(os.path.join(image_dir, 'all_spectrograms', 'event_*.png')))
    if n_events is not None:
        img_files = img_files[:n_events]
    imgs = []
    for f in img_files:
        img = Image.open(f).convert('L').resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs.append(arr)
    imgs = np.stack(imgs)
    imgs = torch.from_numpy(imgs).unsqueeze(1).to(device)
    with torch.no_grad():
        latents = encoder(imgs).cpu().numpy()
    return latents
"""
Train autoencoder for SWR spectrogram analysis (PyTorch-only implementation).
Supports ResNet, VAE, and Attention architectures.
"""

import os
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Import improved architectures - use full package path
try:
    from cnn_autoencoder.improved_autoencoder import ResNetAutoencoder, SWR_VAE, AttentionAutoencoder, vae_loss_function
except ImportError:
    try:
        from .improved_autoencoder import ResNetAutoencoder, SWR_VAE, AttentionAutoencoder, vae_loss_function
    except ImportError:
        from improved_autoencoder import ResNetAutoencoder, SWR_VAE, AttentionAutoencoder, vae_loss_function


class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram images."""
    def __init__(self, files, img_size=128):
        self.files = sorted(files)
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L')
        return self.tf(img)


def train_autoencoder(arch='vae', latent_dim=128, epochs=15, batch_size=64, lr=1e-3, data_dir='.', beta=1.0, device=None, cnn_files_dir=None):
    """
    Train an autoencoder on spectrogram images.
    
    Parameters:
    -----------
    arch : str
        Architecture type: 'resnet', 'vae', or 'attention'
    latent_dim : int
        Dimension of latent space
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float or None
        Learning rate (None defaults to 1e-3)
    data_dir : str
        Directory containing 'all_spectrograms' folder
    beta : float
        Beta parameter for VAE loss
    device : str or None
        Device to use ('cuda' or 'cpu', None auto-detects)
    cnn_files_dir : str or None
        Directory to save models and outputs (default: data_dir/cnn_files)
    
    Returns:
    --------
    model : torch.nn.Module
        Trained model
    """
    # Handle None lr (convert to default)
    if lr is None:
        lr = 1e-3

    # Set cnn_files_dir
    if cnn_files_dir is None:
        cnn_files_dir = os.path.join(data_dir, 'cnn_files')
    os.makedirs(cnn_files_dir, exist_ok=True)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    images_path = os.path.join(data_dir, 'all_spectrograms')
    img_files = glob(os.path.join(images_path, 'event_*.png'))
    if not img_files:
        raise RuntimeError(f"No images found in {images_path}")

    print(f"Found {len(img_files)} images")
    ds = SpectrogramDataset(img_files, img_size=128)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device=='cuda'))
    
    # Create model
    if arch == 'resnet':
        model = ResNetAutoencoder(latent_dim=latent_dim).to(device)
        is_vae = False
    elif arch == 'vae':
        model = SWR_VAE(latent_dim=latent_dim).to(device)
        is_vae = True
    elif arch == 'attention':
        model = AttentionAutoencoder(latent_dim=latent_dim).to(device)
        is_vae = False
    else:
        raise ValueError("arch must be 'resnet', 'vae', or 'attention'")
    
    print(f"Training {arch} autoencoder with latent_dim={latent_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_loss = float('inf')
    history = {'loss': []}
    
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for data in dl:
            data = data.to(device)
            opt.zero_grad()
            
            if is_vae:
                recon, mu, logvar = model(data)
                loss = vae_loss_function(recon, data, mu, logvar, beta=beta)
            else:
                recon = model(data)
                loss = F.mse_loss(recon, data)
            
            loss.backward()
            opt.step()
            running += loss.item() * data.size(0)
        
        epoch_loss = running / len(ds)
        history['loss'].append(epoch_loss)
        print(f"Epoch {ep}/{epochs}  Loss: {epoch_loss:.6f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_path = os.path.join(cnn_files_dir, f"full_model_{arch}.pth")
            encoder_path = os.path.join(cnn_files_dir, f"encoder_model_{arch}.pkl")

            torch.save(model.state_dict(), model_path)

            # Save encoder based on architecture
            if arch == 'vae':
                torch.save(model.encoder_conv.state_dict(), encoder_path)
            else:
                torch.save(model.encoder.state_dict(), encoder_path)

            print(f"✓ Checkpoint saved: {os.path.abspath(model_path)}")
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{arch.upper()} Autoencoder Training')
    plt.legend()
    plt.grid(True)
    history_plot_path = os.path.join(cnn_files_dir, f'training_history_{arch}.png')
    plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved: {os.path.abspath(history_plot_path)}")
    
    # Generate sample reconstructions
    model.eval()
    with torch.no_grad():
        # Get a batch of samples
        sample_batch = next(iter(dl))[:8].to(device)
        
        if is_vae:
            recon_batch, _, _ = model(sample_batch)
        else:
            recon_batch = model(sample_batch)
        
        # Plot
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            # Original
            axes[0, i].imshow(sample_batch[i].cpu().squeeze(), cmap='viridis')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstruction
            axes[1, i].imshow(recon_batch[i].cpu().squeeze(), cmap='viridis')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
    recon_plot_path = os.path.join(cnn_files_dir, f'sample_reconstructions_{arch}.png')
    plt.savefig(recon_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Sample reconstructions saved: {os.path.abspath(recon_plot_path)}")
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Train autoencoder for SWR spectrograms')
    p.add_argument('--arch', default='vae', choices=['resnet','vae','attention'],
                   help='Architecture type')
    p.add_argument('--latent_dim', type=int, default=128,
                   help='Latent dimension size')
    p.add_argument('--epochs', type=int, default=15,
                   help='Number of training epochs')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Learning rate')
    p.add_argument('--data_dir', type=str, default='.',
                   help='Data directory containing all_spectrograms/')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Beta parameter for VAE loss')
    p.add_argument('--cnn_files_dir', type=str, default=None,
                   help='Directory to save models and outputs (default: data_dir/cnn_files)')

    args = p.parse_args()
    train_autoencoder(**vars(args))
