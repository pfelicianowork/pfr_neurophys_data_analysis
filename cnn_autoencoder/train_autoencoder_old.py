"""
IMPROVED: Train advanced autoencoder (ResNet or VAE) for SWR spectrogram analysis.
Supports multiple architecture choices.
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


class ReconstructionCallback(Callback):
    """Callback to monitor reconstruction quality during training."""
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
    
    def after_epoch(self):
        train_loss = float(self.learn.recorder.losses[-1]) if self.learn.recorder.losses else 0
        valid_loss = float(self.learn.recorder.values[-1][0]) if len(self.learn.recorder.values) > 0 else 0
        
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        
        print(f"Epoch {self.epoch+1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        
        # Periodic reconstruction check
        if (self.epoch + 1) % 5 == 0 or self.epoch == 0:
            self.learn.model.eval()
            with torch.no_grad():
                batch = self.dls.valid.one_batch()
                x = batch[0][:4]
                
                # Handle VAE vs regular autoencoder
                if hasattr(self.learn.model, 'get_latent'):
                    recons, _, _ = self.learn.model(x)
                else:
                    recons = self.learn.model(x)
                
                recon_mean = float(recons.mean())
                recon_std = float(recons.std())
                mse = float(((x - recons) ** 2).mean())
                
                print(f"  Reconstruction: mean={recon_mean:.4f}, std={recon_std:.4f}, MSE={mse:.4f}")
                
                if recon_std < 0.01:
                    print(f"  ⚠️  WARNING: Reconstructions have very low variance!")
            
            self.learn.model.train()


class VAELossWrapper(nn.Module):
    """Wrapper to use VAE loss with fastai."""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred, targ):
        recon, mu, logvar = pred
        return vae_loss_function(recon, targ, mu, logvar, self.beta)


def get_x_y(p):
    return p


def train_autoencoder(arch='resnet', latent_dim=128, epochs=15, lr=None, beta=1.0):
    """
    Train improved autoencoder.
    
    Parameters:
    -----------
    arch : str
        Architecture choice: 'resnet', 'vae', or 'attention'
    latent_dim : int
        Dimension of latent space
    epochs : int
        Number of training epochs
    lr : float, optional
        Learning rate (auto-detected if None)
    beta : float
        Beta parameter for VAE (controls KL weight)
    """
    
    data_path = Path("all_spectrograms")
    if not data_path.exists():
        print(f"Error: Directory '{data_path}' not found.")
        print("Run generate_all_spectrograms_improved.py first.")
        return
    
    print("="*80)
    print(f"TRAINING {arch.upper()} AUTOENCODER")
    print("="*80)
    
    # --- Setup DataLoaders ---
    print("\n--- Setting up DataLoaders ---")
    autoencoder_db = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW)),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.15, seed=42),
        get_x=get_x_y,
        get_y=get_x_y,
        item_tfms=Resize(128),
        batch_tfms=[
            IntToFloatTensor(),
        ]
    )
    
    dls = autoencoder_db.dataloaders(data_path, bs=32, num_workers=0)
    num_train = len(dls.train_ds)
    num_valid = len(dls.valid_ds)
    print(f"✓ DataLoaders created:")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_valid}")
    
    # --- Initialize Model ---
    print(f"\n--- Initializing {arch.upper()} Model ---")
    
    if arch == 'resnet':
        model = ResNetAutoencoder(latent_dim=latent_dim)
        loss_func = MSELossFlat()
    elif arch == 'vae':
        model = SWR_VAE(latent_dim=latent_dim)
        loss_func = VAELossWrapper(beta=beta)
    elif arch == 'attention':
        from improved_autoencoder import AttentionAutoencoder
        model = AttentionAutoencoder(latent_dim=latent_dim)
        loss_func = MSELossFlat()
    else:
        print(f"Unknown architecture: {arch}")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"  Latent dimension: {latent_dim}")
    
    # --- Create Learner ---
    learn = Learner(dls, model, loss_func=loss_func)
    recon_cb = ReconstructionCallback()
    learn.add_cb(recon_cb)
    
    # --- Find Learning Rate ---
    if lr is None:
        print("\n--- Finding Optimal Learning Rate ---")
        lr_find_res = learn.lr_find()
        suggested_lr = lr_find_res.valley
        print(f"✓ Suggested learning rate: {suggested_lr:.2e}")
    else:
        suggested_lr = lr
        print(f"\n--- Using specified learning rate: {suggested_lr:.2e} ---")
    
    # --- Training ---
    print(f"\n--- Training for {epochs} epochs ---")
    print()
    
    learn.fit_one_cycle(epochs, suggested_lr)
    
    # --- Post-Training Validation ---
    print("\n--- Post-Training Validation ---")
    learn.model.eval()
    
    with torch.no_grad():
        batch = dls.valid.one_batch()
        x = batch[0][:5]
        
        if arch == 'vae':
            recons, mu, logvar = learn.model(x)
        else:
            recons = learn.model(x)
        
        mse = float(((x - recons) ** 2).mean())
        recon_mean = float(recons.mean())
        recon_std = float(recons.std())
        
        print(f"Final validation metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  Reconstruction mean: {recon_mean:.4f}")
        print(f"  Reconstruction std: {recon_std:.4f}")
        
        # Quality assessment
        if recon_std < 0.01:
            print("\n❌ ERROR: Model did not learn! Reconstructions are flat.")
            print("   Consider:")
            print("   - Checking data quality")
            print("   - Adjusting learning rate")
            print("   - Training longer")
            return False
        elif mse > 0.1:
            print("\n⚠️  WARNING: High reconstruction error.")
            print("   Consider training longer or adjusting architecture.")
        else:
            print("\n✓ Model appears to have trained successfully!")
    
    # --- Save Models ---
    print("\n--- Saving Models ---")
    
    # Save full model
    full_model_path = f"autoencoder_model_{arch}.pth"
    torch.save(learn.model.state_dict(), full_model_path)
    print(f"✓ Full autoencoder saved to '{full_model_path}'")
    
    # Save encoder only
    if arch == 'vae':
        # For VAE, save the encoder conv layers
        encoder_path = f"encoder_model_{arch}.pkl"
        torch.save(learn.model.encoder_conv.state_dict(), encoder_path)
    else:
        encoder_path = f"encoder_model_{arch}.pkl"
        torch.save(learn.model.encoder.state_dict(), encoder_path)
    print(f"✓ Encoder saved to '{encoder_path}'")
    
    # Save learner
    learner_path = f"autoencoder_learner_{arch}.pkl"
    learn.export(learner_path)
    print(f"✓ Learner saved to '{learner_path}'")
    
    # --- Plot Training Curves ---
    print("\n--- Generating Training Plots ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(recon_cb.train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(recon_cb.valid_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Example reconstructions
    with torch.no_grad():
        batch = dls.valid.one_batch()
        x = batch[0][:5]
        
        if arch == 'vae':
            recons, _, _ = learn.model(x)
        else:
            recons = learn.model(x)
        
        # Show side-by-side
        n_show = 3
        for i in range(n_show):
            # Original
            axes[1].imshow(x[i, 0].cpu().numpy(), cmap='viridis', aspect='auto',
                          extent=[i*2, i*2+0.9, 0, 1])
            # Reconstructed
            axes[1].imshow(recons[i, 0].cpu().numpy(), cmap='viridis', aspect='auto',
                          extent=[i*2+1, i*2+1.9, 0, 1])
        
        axes[1].set_xticks([i*2+0.45 for i in range(n_show)] + [i*2+1.45 for i in range(n_show)])
        axes[1].set_xticklabels(['Orig', 'Recon'] * n_show, fontsize=8)
        axes[1].set_yticks([])
        axes[1].set_title('Example Reconstructions')
    
    plt.tight_layout()
    plot_path = f'training_results_{arch}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plots saved to '{plot_path}'")
    plt.close()
    
    # --- Summary ---
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Architecture: {arch.upper()}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Final MSE: {mse:.4f}")
    print(f"\nFiles saved:")
    print(f"  - {full_model_path}")
    print(f"  - {encoder_path}")
    print(f"  - {learner_path}")
    print(f"  - {plot_path}")
    print("\nNext: Run cluster_events_improved.py to cluster with combined features")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train improved autoencoder for SWR analysis')
    parser.add_argument('--arch', type=str, default='resnet', 
                       choices=['resnet', 'vae', 'attention'],
                       help='Architecture to use')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (auto if not specified)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for VAE')
    
    args = parser.parse_args()
    
    train_autoencoder(
        arch=args.arch,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta
    )
