# See main workspace for full docstring and comments

import torch
from torch import nn
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np

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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_x_y(p):
    return p

class ReconstructionCallback(Callback):
    def __init__(self):
        self.losses = []
    def after_epoch(self):
        self.losses.append(float(self.learn.recorder.losses[-1]))
        train_loss = float(self.learn.recorder.losses[-1])
        valid_loss = float(self.learn.recorder.values[-1][0]) if len(self.learn.recorder.values) > 0 else 0
        print(f"Epoch {self.epoch+1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        if (self.epoch + 1) % 5 == 0 or self.epoch == 0:
            self.learn.model.eval()
            with torch.no_grad():
                batch = self.dls.valid.one_batch()
                x = batch[0][:4]
                recons = self.learn.model(x)
                recon_mean = float(recons.mean())
                recon_std = float(recons.std())
                print(f"  Reconstruction check: mean={recon_mean:.4f}, std={recon_std:.4f}")
                if recon_std < 0.01:
                    print(f"  ⚠️ WARNING: Reconstructions have very low variance!")
            self.learn.model.train()

def train_autoencoder():
    data_path = Path("../all_spectrograms")
    if not data_path.exists():
        print(f"Error: Directory '{data_path}' not found.")
        return
    print("="*80)
    print("TRAINING IMPROVED AUTOENCODER")
    print("="*80)
    print("\n--- Setting up DataLoaders ---")
    autoencoder_db = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW)),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.1, seed=42),
        get_x=get_x_y,
        get_y=get_x_y,
        item_tfms=Resize(128)
    )
    dls = autoencoder_db.dataloaders(data_path, bs=32, num_workers=0)
    num_train = len(dls.train_ds)
    num_valid = len(dls.valid_ds)
    print(f"✓ DataLoaders created:")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_valid}")
    print("\n--- Initializing Model ---")
    model = ImprovedAutoencoder()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    learn = Learner(dls, model, loss_func=MSELossFlat())
    recon_cb = ReconstructionCallback()
    learn.add_cb(recon_cb)
    print("\n--- Finding Learning Rate ---")
    lr_find_res = learn.lr_find()
    suggested_lr = lr_find_res.valley
    print(f"✓ Suggested learning rate: {suggested_lr:.2e}")
    print("\n--- Training ---")
    print("Training for 10 epochs...")
    print()
    learn.fit_one_cycle(10, suggested_lr)
    print("\n--- Post-Training Validation ---")
    learn.model.eval()
    with torch.no_grad():
        batch = dls.valid.one_batch()
        x = batch[0][:5]
        recons = learn.model(x)
        mse = float(((x - recons) ** 2).mean())
        recon_mean = float(recons.mean())
        recon_std = float(recons.std())
        print(f"Final validation check:")
        print(f"  MSE: {mse:.4f}")
        print(f"  Reconstruction mean: {recon_mean:.4f}")
        print(f"  Reconstruction std: {recon_std:.4f}")
        if recon_std < 0.01:
            print("\n❌ ERROR: Model did not learn! Reconstructions are flat.")
            print("   Training may have failed.")
            return
        elif mse > 0.1:
            print("\n⚠️  WARNING: High reconstruction error.")
            print("   Consider training longer or adjusting architecture.")
        else:
            print("\n✓ Model appears to have trained successfully!")
    print("\n--- Saving Models ---")
    full_model_path = "autoencoder_model.pth"
    torch.save(learn.model.state_dict(), full_model_path)
    print(f"✓ Full autoencoder saved to '{full_model_path}'")
    encoder_path = "encoder_model.pkl"
    torch.save(learn.model.encoder.state_dict(), encoder_path)
    print(f"✓ Encoder saved to '{encoder_path}'")
    learner_path = "autoencoder_learner.pkl"
    learn.export(learner_path)
    print(f"✓ Learner saved to '{learner_path}'")
    print("\n--- Training Loss Curve ---")
    if len(recon_cb.losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(recon_cb.losses)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
        print("✓ Training loss plot saved to 'training_loss.png'")
        plt.close()
    print("\n" + "="*80)
    print("Training complete!")
    print("Next: Run diagnose_autoencoder.py to verify the model")
    print("="*80)

if __name__ == "__main__":
    train_autoencoder()
