
import torch
from torch import nn
from fastai.vision.all import *
import os

# --- 1. Autoencoder Model Definition ---
class Autoencoder(nn.Module):
    """
    A simple Convolutional Autoencoder.
    The encoder compresses the image into a latent vector, and the decoder
    reconstructs the image from that vector.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> 16x64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64x16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 128x8x8
            nn.Flatten(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(), # Use Sigmoid for output in [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# For an autoencoder, the input and target are the same image path.
def get_x_y(p):
    return p

def train_autoencoder():
    """
    Trains the autoencoder on the spectrogram images.
    """
    data_path = Path("all_spectrograms")
    if not data_path.exists():
        print(f"Error: Directory '{data_path}' not found. Please run generate_all_spectrograms.py first.")
        return

    print("--- Setting up DataLoaders for Autoencoder ---")
    
    # --- 2. DataBlock Setup ---
    # For an autoencoder, the input and target are the same.
    # We tell the DataBlock this by setting blocks=(ImageBlock, ImageBlock).
    autoencoder_db = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW)), # Use PILImageBW for grayscale
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.1, seed=42), # 10% validation set
        get_x=get_x_y,
        get_y=get_x_y,
        item_tfms=Resize(128) # Resize all images to 128x128
    )

    dls = autoencoder_db.dataloaders(data_path, bs=32, num_workers=0)
    print("DataLoaders created. One batch shape:")
    dls.show_batch()

    # --- 3. Learner Setup and Training ---
    print("\n--- Initializing and Training the Autoencoder ---")
    model = Autoencoder()
    
    # The loss function measures the difference between the original and reconstructed image.
    # MSE (Mean Squared Error) is a standard choice for this.
    learn = Learner(dls, model, loss_func=MSELossFlat())

    print("Finding initial learning rate...")
    lr_find_res = learn.lr_find()
    print(f"Suggested learning rate: {lr_find_res.valley:.2e}")

    print("\nTraining for 5 epochs...")
    learn.fit_one_cycle(5, lr_find_res.valley)

    print("\n--- Saving the trained models ---")
    
    # Save the full autoencoder model (both encoder and decoder)
    full_model_path = "autoencoder_model.pth"
    torch.save(learn.model.state_dict(), full_model_path)
    print(f"Full autoencoder model saved to '{full_model_path}'.")
    
    # Save encoder separately for feature extraction
    encoder_path = "encoder_model.pkl"
    torch.save(learn.model.encoder.state_dict(), encoder_path)
    print(f"Encoder model state dictionary saved to '{encoder_path}'.")

    # Save the full learner for easy evaluation later
    learner_path = "autoencoder_learner.pkl"
    learn.export(learner_path)
    print(f"Full learner saved to '{learner_path}'.")
    print("\nNext step: Use the trained encoder to extract features and cluster the events.")

if __name__ == "__main__":
    train_autoencoder()
