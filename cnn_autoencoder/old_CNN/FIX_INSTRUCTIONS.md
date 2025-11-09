# Fix Instructions for Blank Reconstructions

## Problem Diagnosed
The autoencoder is producing blank (all-zero) reconstructions because the full model weights weren't being saved during training. This causes:
1. Reconstructed images appear blank
2. Error maps just show the original (because error = |original - 0|)
3. All NaN values in Part 9 reconstruction error statistics

## Solution Implemented

### 1. Updated `train_autoencoder.py`
Now saves **both**:
- `autoencoder_model.pth` - **Full model weights (encoder + decoder)**
- `encoder_model.pkl` - Encoder only (for feature extraction)
- `autoencoder_learner.pkl` - FastAI learner object

### 2. Updated `evaluate_clusters.py` 
**Part 1 improvements:**
- Loads full model weights from `autoencoder_model.pth` first
- Falls back to learner if weights file missing
- **Added debug prints** to diagnose the issue:
  - Input batch statistics
  - Output batch statistics  
  - Model weight statistics
- Better error handling

**Part 9 improvements:**
- Fixed image preprocessing to match training exactly
- Loads images directly with PIL (not through FastAI)
- Proper normalization (÷255)
- Progress indicator every 100 images
- Error handling with limited error printing
- Checks for valid errors before plotting

## Required Action

**You MUST retrain the autoencoder** to generate the full model weights:

```powershell
cd "C:\Users\PedWKS\OneDrive - Massachusetts Institute of Technology\pfr_neurophys_data_analysis"
python train_autoencoder.py
```

This will create `autoencoder_model.pth` which contains the complete trained model.

## After Retraining

Run the evaluation script:
```powershell
python evaluate_clusters.py
```

You should now see:
- **Part 1:** Proper reconstructions with meaningful error maps
- Debug output showing non-zero min/max/mean for both inputs and outputs
- **Part 9:** Valid reconstruction error statistics (no NaN values)

## Verification Checklist

After retraining, check that you have:
- [ ] `autoencoder_model.pth` exists (new file, ~several MB)
- [ ] Debug prints show output stats with non-zero values
- [ ] Reconstructed spectrograms are visible (not blank)
- [ ] Error maps show focused patterns (not just original)
- [ ] Part 9 statistics show numeric values (not NaN)

## If Issues Persist

Check the debug output from Part 1:
- If **input stats** show min/max near 0 and 1: ✓ Input loading works
- If **output stats** show all zeros: ✗ Model still not loaded correctly
- If **weight stats** show all zeros: ✗ Model file corrupted or wrong architecture

Contact for help if problems continue after retraining.
