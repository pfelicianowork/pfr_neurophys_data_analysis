"""
Improved autoencoder architectures for SWR spectrogram analysis.
Includes ResNet-style autoencoder and Variational Autoencoder (VAE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.leaky_relu(out, 0.2)
        return out


class ResNetAutoencoder(nn.Module):
    """
    ResNet-style autoencoder with residual blocks for better training.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Initial conv block: 128x128 -> 64x64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            
            # Residual blocks
            ResidualBlock(32, 64, stride=2),  # 64x64 -> 32x32
            ResidualBlock(64, 64),
            
            ResidualBlock(64, 128, stride=2),  # 32x32 -> 16x16
            ResidualBlock(128, 128),
            
            ResidualBlock(128, 256, stride=2),  # 16x16 -> 8x8
            ResidualBlock(256, 256),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),
            
            # Upsample blocks
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Extract features without decoding."""
        return self.encoder(x)


class SWR_VAE(nn.Module):
    """
    Variational Autoencoder for learning structured SWR representations.
    VAEs learn a probabilistic latent space which can be better for clustering.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 256 * 8 * 8)
        
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        h = self.decoder_fc(z)
        return self.decoder_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_latent(self, x):
        """Get latent representation (mean) for clustering."""
        mu, _ = self.encode(x)
        return mu


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Parameters:
    -----------
    recon_x : torch.Tensor
        Reconstructed images
    x : torch.Tensor
        Original images
    mu : torch.Tensor
        Mean of latent distribution
    logvar : torch.Tensor
        Log variance of latent distribution
    beta : float
        Weight for KL divergence (beta-VAE)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


class AttentionBlock(nn.Module):
    """Self-attention mechanism for capturing long-range dependencies."""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        # Compute attention
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        return self.gamma * out + x


class AttentionAutoencoder(nn.Module):
    """
    Autoencoder with self-attention for capturing temporal dependencies
    in spectrograms.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            AttentionBlock(64),  # Attention after feature extraction
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


# Training helper functions

def train_vae_epoch(model, dataloader, optimizer, device, beta=1.0):
    """Train VAE for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon, mu, logvar = model(data)
        
        recon_loss = F.mse_loss(recon, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_recon_loss / n_samples, total_kl_loss / n_samples


def validate_vae(model, dataloader, device, beta=1.0):
    """Validate VAE."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = vae_loss_function(recon, data, mu, logvar, beta)
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)
