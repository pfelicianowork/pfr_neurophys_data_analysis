# Quick Start Guide - Improved SWR Classification Pipeline

## Complete File List

### Core Modules (Import these)
1. ✅ `feature_extraction.py` - Extract biological features
2. ✅ `improved_autoencoder.py` - Advanced architectures (ResNet, VAE, Attention)
3. ✅ `advanced_clustering.py` - Optimal clustering algorithms
4. ✅ `biological_validation.py` - Validation and quality scoring

### Pipeline Scripts (Run these)
5. ✅ `generate_all_spectrograms_improved.py` - Step 1: Data preparation
6. ✅ `train_autoencoder_improved.py` - Step 2: Train autoencoder
7. ✅ `cluster_events_improved.py` - Step 3: Cluster with combined features
8. ✅ `evaluate_clusters_improved.py` - Step 4: Comprehensive evaluation

### Documentation
9. ✅ `README_IMPROVED_PIPELINE.md` - Full documentation
10. ✅ `QUICK_START_GUIDE.md` - This file

---

## Step-by-Step Workflow

### Step 1: Generate Spectrograms and Extract Features (5-10 minutes)

```bash
python generate_all_spectrograms_improved.py
```

**What it does:**
- Detects SWR events from your raw LFP data
- Generates spectrogram images for each event
- Extracts 30+ biological features (duration, frequency, power, MUA coupling, etc.)

**Output files:**
- `all_spectrograms/` directory with images
- `detected_events.pkl` - Event metadata
- `biological_features.pkl` - Extracted biological features

---

### Step 2: Train Autoencoder (10-30 minutes depending on GPU)

**Option A: ResNet-style (Recommended for most cases)**
```bash
python train_autoencoder_improved.py --arch resnet --latent_dim 128 --epochs 15
```

**Option B: VAE (Better for clustering, probabilistic latent space)**
```bash
python train_autoencoder_improved.py --arch vae --latent_dim 64 --epochs 20 --beta 1.0
```

**Option C: Attention-based (For temporal dependencies)**
```bash
python train_autoencoder_improved.py --arch attention --latent_dim 128 --epochs 15
```

**What it does:**
- Trains chosen architecture on your spectrograms
- Monitors reconstruction quality
- Saves encoder for feature extraction

**Output files:**
- `autoencoder_model_{arch}.pth` - Full trained model
- `encoder_model_{arch}.pkl` - Encoder for feature extraction
- `autoencoder_learner_{arch}.pkl` - Learner for inference
- `training_results_{arch}.png` - Training curves

**Tips:**
- Start with ResNet - it's most robust
- If clustering results are poor, try VAE
- Watch the reconstruction std - should be > 0.01

---

### Step 3: Cluster Events (5-15 minutes)

**Recommended: Use combined features**
```bash
python cluster_events_improved.py --arch resnet --ae_weight 0.7 --bio_weight 0.3
```

**Alternative: Only autoencoder features**
```bash
python cluster_events_improved.py --arch resnet --no_combined
```

**Advanced: Adjust clustering parameters**
```bash
python cluster_events_improved.py \
    --arch resnet \
    --ae_weight 0.6 \
    --bio_weight 0.4 \
    --k_min 3 \
    --k_max 10
```

**What it does:**
- Extracts autoencoder features from spectrograms
- Combines with biological features (weighted)
- Tests multiple clustering methods (K-Means, Hierarchical, GMM)
- Finds optimal k using 4 metrics
- Cross-validates stability

**Output files:**
- `autoencoder_features_{arch}.pkl` - Cached features
- `events_with_clusters_{type}.pkl` - Events with cluster labels
- `clustering_info_{type}.pkl` - Clustering metrics and results
- `clustering_metrics_comparison.png` - Metric plots
- `clustering_dendrogram.png` - Hierarchical view
- `inter_cluster_distances.png` - Separation analysis

**Interpreting results:**
- **Silhouette > 0.5**: Good separation
- **ARI > 0.7**: Stable clustering
- **Gap statistic**: Higher is better

---

### Step 4: Evaluate and Validate (2-5 minutes)

```bash
python evaluate_clusters_improved.py --feature_type combined
```

**What it does:**
- Biological validation (duration, frequency, MUA coupling)
- Statistical comparisons between clusters
- Feature space visualization (PCA, t-SNE)
- Average spectrograms per cluster
- Sample events per cluster

**Output files:**
- `cluster_validation_report_{type}.txt` - Text report
- `feature_space_{type}.png` - PCA/t-SNE visualization
- `cluster_{type}_characteristics_boxplots.png` - Feature distributions
- `cluster_{id}_avg_spectrogram.png` - Average per cluster
- `cluster_{id}_samples.png` - Sample events per cluster

**Key metrics to check:**
- **Quality Score ≥ 80%**: Excellent - likely true SWRs
- **Duration in range**: Should be 15-500ms
- **Ripple band %**: Should be > 60% in 125-250 Hz
- **MUA correlation**: Should be positive

---

## Common Usage Patterns

### Pattern 1: Quick Default Run

```bash
# Step 1
python generate_all_spectrograms_improved.py

# Step 2
python train_autoencoder_improved.py --arch resnet

# Step 3
python cluster_events_improved.py --arch resnet

# Step 4
python evaluate_clusters_improved.py
```

### Pattern 2: Optimize Clustering

Try different feature weights to see what works best:

```bash
# Heavily favor learned features
python cluster_events_improved.py --ae_weight 0.8 --bio_weight 0.2

# Balanced
python cluster_events_improved.py --ae_weight 0.5 --bio_weight 0.5

# Heavily favor biological features
python cluster_events_improved.py --ae_weight 0.3 --bio_weight 0.7

# Compare results
python evaluate_clusters_improved.py --feature_type combined
```

### Pattern 3: Compare Architectures

```bash
# Train all three
python train_autoencoder_improved.py --arch resnet
python train_autoencoder_improved.py --arch vae
python train_autoencoder_improved.py --arch attention

# Cluster with each
python cluster_events_improved.py --arch resnet
python cluster_events_improved.py --arch vae
python cluster_events_improved.py --arch attention

# Evaluate and compare
python evaluate_clusters_improved.py --feature_type combined
```

---

## Troubleshooting

### Issue: "Encoder model not found"
**Solution:** Run `train_autoencoder_improved.py` first

### Issue: "Biological features not found"
**Solution:** Run `generate_all_spectrograms_improved.py` first

### Issue: Low silhouette score (< 0.3)
**Try:**
1. Adjust feature weights: `--ae_weight 0.5 --bio_weight 0.5`
2. Use only autoencoder: `--no_combined`
3. Try VAE architecture: `--arch vae`

### Issue: Poor biological quality scores
**Try:**
1. Increase bio weight: `--bio_weight 0.5`
2. Check SWR detection parameters in Step 1
3. Review sample events in evaluation

### Issue: Model not learning (flat reconstructions)
**Try:**
1. Lower learning rate: `--lr 0.0001`
2. Train longer: `--epochs 25`
3. Check data quality in `all_spectrograms/`

---

## Understanding the Output

### Clustering Quality Indicators

**✓ Good clustering:**
- Silhouette > 0.5
- Davies-Bouldin < 1.5
- ARI > 0.7
- Quality scores > 70%

**⚠️ Needs improvement:**
- Silhouette < 0.3
- ARI < 0.5
- Quality scores < 60%

### Biological Validation

**For each cluster, check:**
1. **Duration**: Should be 15-500ms for SWRs
2. **Frequency**: Should peak at 125-250 Hz
3. **MUA correlation**: Should be positive (> 0)
4. **Inter-event interval**: Not too short (> 100ms)

---

## Advanced Tips

### 1. Feature Weight Optimization

```python
# Try a grid search
for ae_w in [0.5, 0.6, 0.7, 0.8]:
    bio_w = 1.0 - ae_w
    os.system(f"python cluster_events_improved.py --ae_weight {ae_w} --bio_weight {bio_w}")
```

### 2. Custom Feature Selection

Edit `feature_extraction.py` to add your own features:
```python
def extract_swr_features(event, lfp_array, mua_vec, fs, region_lfp=None):
    features = {}
    # Add your custom features here
    features['my_custom_metric'] = compute_my_metric(event)
    return features
```

### 3. Export for Further Analysis

```python
import pickle
import pandas as pd

# Load results
with open('events_with_clusters_combined.pkl', 'rb') as f:
    events = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(events)

# Export to CSV
df.to_csv('clustered_events.csv', index=False)

# Export to MATLAB
from scipy.io import savemat
savemat('clustered_events.mat', {'events': df.to_dict('list')})
```

---

## Performance Benchmarks

**Typical runtime on a modern laptop (NVIDIA RTX GPU):**
- Step 1 (1000 events): ~5 minutes
- Step 2 (15 epochs): ~10-20 minutes
- Step 3 (k range 2-12): ~5-10 minutes
- Step 4 (evaluation): ~2-3 minutes

**Total: ~25-40 minutes for complete pipeline**

---

## Next Steps After Clustering

1. **Verify biological plausibility** - Check validation report
2. **Visualize results** - Look at average spectrograms and samples
3. **Statistical analysis** - Compare clusters quantitatively
4. **Behavioral correlation** - Link clusters to behavioral states
5. **Temporal analysis** - Examine cluster occurrence over time
6. **Multi-session analysis** - Apply to other recordings

---

## Getting Help

**Common error messages:**

- `FileNotFoundError`: Run previous step first
- `Model architecture mismatch`: Ensure latent_dim matches training
- `Out of memory`: Reduce batch size in code
- `Poor reconstruction`: Check data quality, adjust learning rate

**For best results:**
1. Start with default parameters
2. Check each step's output before proceeding
3. Review validation metrics carefully
4. Iterate on feature weights if needed

---

## Citation

If you use this pipeline, consider citing:
- Deep learning methods for neuroscience
- Clustering validation techniques
- Biological feature engineering approaches
