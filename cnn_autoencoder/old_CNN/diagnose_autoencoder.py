"""
Diagnostic script to test if the autoencoder was trained properly.
Run this to check model quality before running full evaluation.
"""

import torch
from torch import nn
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Define the Autoencoder Architecture ---
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
    def forward(self, x): 
        return self.decoder(self.encoder(x))

# Improved model with BatchNorm and LeakyReLU
class ImprovedAutoencoder(nn.Module):
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
    def forward(self, x): 
        return self.decoder(self.encoder(x))

def diagnose_model():
    """Run comprehensive diagnostics on the trained autoencoder."""
    
    print("="*80)
    print("AUTOENCODER DIAGNOSTIC REPORT")
    print("="*80)
    
    # Check if model file exists
    model_path = Path("autoencoder_model.pth")
    if not model_path.exists():
        print("\n❌ ERROR: autoencoder_model.pth not found!")
        print("   Please run train_autoencoder.py first.")
        return
    
    print(f"\n✓ Model file found: {model_path}")
    print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
    
    # Load model
    print("\n--- Loading Model ---")
    
    # Try to detect which architecture was used
    state_dict = torch.load(model_path, weights_only=False)
    
    # Check if BatchNorm layers exist (improved model)
    # BatchNorm layers have keys like "encoder.1.weight", "encoder.1.running_mean", etc.
    has_batchnorm = any('running_mean' in key or 'running_var' in key for key in state_dict.keys())
    
    if has_batchnorm:
        print("Detected: ImprovedAutoencoder (with BatchNorm)")
        model = ImprovedAutoencoder()
    else:
        print("Detected: Original Autoencoder (basic)")
        model = Autoencoder()
    
    try:
        model.load_state_dict(state_dict)
        model.eval()
        model.cpu()
        print("✓ Model loaded successfully")
        
        # DEBUG: Check if model is in eval mode
        print(f"  Model training mode: {model.training}")
        
        # Test with a simple forward pass
        test_input = torch.rand(1, 1, 128, 128)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"  Quick test - Input mean: {float(test_input.mean()):.4f}, Output mean: {float(test_output.mean()):.4f}")
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check weight statistics
    print("\n--- Weight Statistics ---")
    for name, param in model.named_parameters():
        if 'weight' in name:
            w_min = float(param.min())
            w_max = float(param.max())
            w_std = float(param.std())
            print(f"{name:40s} | min={w_min:7.4f}, max={w_max:7.4f}, std={w_std:7.4f}")
    
    # Check if weights are initialized or trained
    encoder_weights = model.encoder[0].weight
    w_std = float(encoder_weights.std())
    print(f"\nFirst encoder layer weight std: {w_std:.4f}")
    if w_std < 0.01:
        print("⚠️  WARNING: Weights have very low variance - model may not be trained!")
    else:
        print("✓ Weights show reasonable variance")
    
    # Test with sample images
    print("\n--- Testing with Sample Images ---")
    images_path = Path("all_spectrograms")
    if not images_path.exists():
        print("❌ ERROR: all_spectrograms directory not found!")
        return
    
    image_files = list(images_path.glob("*.png"))[:5]
    if not image_files:
        print("❌ ERROR: No images found in all_spectrograms/")
        return
    
    print(f"Testing with {len(image_files)} sample images...")
    
    reconstruction_quality = []
    
    with torch.no_grad():
        for img_file in image_files:
            # Load image
            img = Image.open(img_file).convert('L')
            img = img.resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
            
            # Get reconstruction
            recon = model(img_tensor)
            
            # Calculate MSE
            mse = torch.mean((img_tensor - recon) ** 2).item()
            reconstruction_quality.append(mse)
            
            # Check output diversity
            out_std = float(recon.std())
            out_mean = float(recon.mean())
            out_min = float(recon.min())
            out_max = float(recon.max())
            
            print(f"  {img_file.name}: MSE={mse:.4f}, out_std={out_std:.6f}, out_mean={out_mean:.4f}, range=[{out_min:.4f}, {out_max:.4f}]")
    
    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    avg_mse = np.mean(reconstruction_quality)
    avg_std = np.std(reconstruction_quality)
    
    print(f"\nReconstruction MSE: mean={avg_mse:.4f}, std={avg_std:.4f}")
    
    if avg_mse > 0.1:
        print("❌ POOR: High reconstruction error - model not learning well")
        print("   Recommendation: Retrain with more epochs or adjust learning rate")
    elif avg_mse > 0.01:
        print("⚠️  FAIR: Moderate reconstruction error - model partially trained")
        print("   Recommendation: Consider training for more epochs")
    else:
        print("✓ GOOD: Low reconstruction error - model trained successfully")
    
    # Check for flat outputs
    print("\n--- Checking for Common Issues ---")
    
    with torch.no_grad():
        test_img = torch.rand(1, 1, 128, 128) * 0.5 + 0.25  # Random test image
        test_out = model(test_img)
        test_out_std = float(test_out.std())
        test_out_mean = float(test_out.mean())
        
        print(f"Random input reconstruction: mean={test_out_mean:.4f}, std={test_out_std:.4f}")
        
        if test_out_std < 0.01:
            print("❌ CRITICAL: Model outputs are nearly constant!")
            print("   This means the model is NOT trained.")
            print("   All outputs are approximately {:.4f}".format(test_out_mean))
            print("\n   ACTION REQUIRED:")
            print("   1. Delete autoencoder_model.pth")
            print("   2. Run train_autoencoder.py and watch for errors")
            print("   3. Verify training loss decreases")
        elif abs(test_out_mean - 0.5) < 0.01 and test_out_std < 0.05:
            print("⚠️  WARNING: Model outputs centered around 0.5 with low variance")
            print("   Model may not have trained properly")
        else:
            print("✓ Model outputs show reasonable variation")
    
    # Visualize a reconstruction
    print("\n--- Creating Visual Comparison ---")
    img_file = image_files[0]
    img = Image.open(img_file).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        recon = model(img_tensor)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array, cmap='viridis', aspect='auto')
    axes[0].set_title('Original Spectrogram')
    axes[0].axis('off')
    
    recon_np = recon.squeeze().numpy()
    axes[1].imshow(recon_np, cmap='viridis', aspect='auto')
    axes[1].set_title('Reconstructed Spectrogram')
    axes[1].axis('off')
    
    diff = np.abs(img_array - recon_np)
    im = axes[2].imshow(diff, cmap='hot', aspect='auto')
    axes[2].set_title(f'Absolute Error (MSE={reconstruction_quality[0]:.4f})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.suptitle('Autoencoder Reconstruction Test', fontsize=14)
    plt.tight_layout()
    plt.savefig('autoencoder_diagnostic.png', dpi=150, bbox_inches='tight')
    print("✓ Visual comparison saved to 'autoencoder_diagnostic.png'")
    plt.show()
    
    print("\n" + "="*80)
    print("Diagnostic complete. Check the visualization above.")
    print("="*80)

if __name__ == "__main__":
    diagnose_model()
