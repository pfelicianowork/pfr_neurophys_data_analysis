# CNN Autoencoder Module

This folder contains scripts for training and diagnosing a convolutional autoencoder for SWR spectrogram images.

## Files
- `train_autoencoder_fixed.py`: Train the improved autoencoder on spectrogram images.
- `diagnose_autoencoder.py`: Run diagnostics to verify model quality after training.

## Usage

1. **Train the Autoencoder**

   ```bash
   python train_autoencoder_fixed.py
   ```
   - Expects images in `../all_spectrograms/` (relative to this folder).
   - Produces `autoencoder_model.pth`, `encoder_model.pkl`, and `autoencoder_learner.pkl`.

2. **Diagnose the Model**

   ```bash
   python diagnose_autoencoder.py
   ```
   - Checks model weights, output statistics, and creates a visual comparison.

3. **Integration**
   - Use the saved encoder for feature extraction and clustering in your main pipeline.

## Requirements
- Python 3.8+
- fastai
- torch
- matplotlib
- numpy
- PIL

Install requirements with:
```bash
pip install fastai torch matplotlib numpy pillow
```
