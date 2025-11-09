# Improved SWR Classification Pipeline

## Overview

This improved pipeline combines **deep learning (autoencoders)** with **biological feature engineering** to achieve more robust and interpretable classification of Sharp Wave-Ripple (SWR) events.

## Key Improvements

### 1. **Rich Feature Extraction** (`feature_extraction.py`)
- **Temporal features**: Duration, peak timing, inter-event intervals
- **Spectral features**: Ripple power, peak frequency, spectral entropy, spectral centroid
- **MUA features**: Multi-unit activity max, mean, integral, MUA-ripple correlation
- **Waveform features**: Kurtosis, skewness, peak-to-peak amplitude, RMS
- **Phase coupling**: Slow oscillation phase, phase-amplitude coupling
- **Spatial features**: Multi-channel power distribution, cross-channel correlation
- **Quality metrics**: Signal-to-noise ratio estimates

### 2. **Advanced Autoencoder Architectures** (`improved_autoencoder.py`)
- **ResNet-style Autoencoder**: Residual connections for better gradient flow
- **Variational Autoencoder (VAE)**: Probabilistic latent space for better clustering
- **Attention Autoencoder**: Self-attention mechanism for temporal dependencies

### 3. **Optimal Cluster Selection** (`advanced_clustering.py`)
- Tests multiple clustering algorithms: K-Means, Hierarchical, GMM, DBSCAN
- Evaluates using 4 metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz, Gap Statistic
- Cross-validation for stability assessment
- Automated optimal k selection

### 4. **Biological Validation** (`biological_validation.py`)
- Validates clusters against known SWR properties
- Statistical comparisons between clusters (ANOVA, t-tests)
- Quality scoring based on biological plausibility
- Comprehensive visualization suite

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Preparation & Feature Extraction                    │
│    generate_all_spectrograms_improved.py                   │
│                                                             │
│    Input:  Raw LFP and spike data                          │
│    Output: - Spectrograms (images)                         │
│            - Biological features (pkl)                      │
│            - Detected events (pkl)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Autoencoder Training                                     │
│    train_autoencoder_improved.py                           │
│                                                             │
│    Options: --arch resnet|vae|attention                    │
│            --latent_dim 128                                 │
│            --epochs 15                                      │
│                                                             │
│    Output: - Trained autoencoder models                    │
│            - Encoder for feature extraction                 │
│            - Training curves                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Combined Feature Clustering                              │
│    cluster_events_improved.py                              │
│                                                             │
│    Options: --ae_weight 0.7                                │
│            --bio_weight 0.3                                 │
│            --k_min 2 --k_max 12                            │
│                                                             │
│    Process: - Extract autoencoder features                 │
│             - Combine with biological features              │
│             - Find optimal k (multiple methods)             │
│             - Cross-validate stability                      │
│                                                             │
│    Output: - Clustered events (pkl)                        │
│            - Clustering metrics and plots                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Comprehensive Evaluation                                 │
│    evaluate_clusters_improved.py                           │
│                                                             │
│    Analysis: - Biological validation                       │
│              - Statistical comparisons                      │
│              - Feature space visualization                  │
│              - Average spectrograms per cluster             │
│              - Sample events per cluster                    │
│                                                             │
│    Output: - Validation report (txt)                       │
│            - Comprehensive plots                            │
│            - Quality scores per cluster                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Step 1: Generate spectrograms and extract features

```bash
python generate_all_spectrograms_improved.py
```

This will:
- Detect SWR events from your LFP data
- Generate spectrograms for each event
- Extract 30+ biological features
- Save everything for subsequent steps

### Step 2: Train an autoencoder

Choose an architecture and train:

```bash
# ResNet-style autoencoder (recommended)
python train_autoencoder_improved.py --arch resnet --latent_dim 128 --epochs 15

# Or try VAE for better clustering
python train_autoencoder_improved.py --arch vae --latent_dim 64 --epochs 20 --beta 1.0

# Or attention-based
python train_autoencoder_improved.py --arch attention --latent_dim 128 --epochs 15
```

### Step 3: Cluster with combined features

```bash
# Use combined features (autoencoder + biological)
python cluster_events_improved.py --arch resnet --ae_weight 0.7 --bio_weight 0.3

# Or use autoencoder features only
python cluster_events_improved.py --arch resnet --no_combined

# Specify k range
python cluster_events_improved.py --k_min 3 --k_max 10
```

### Step 4: Evaluate results

```bash
python evaluate_clusters_improved.py --feature_type combined
```

## Feature Weighting

The pipeline allows you to control the relative importance of autoencoder vs biological features:

```bash
# Heavy emphasis on learned features (70%)
python cluster_events_improved.py --ae_weight 0.7 --bio_weight 0.3

# Balanced
python cluster_events_improved.py --ae_weight 0.5 --bio_weight 0.5

# Heavy emphasis on biological features (70%)
python cluster_events_improved.py --ae_weight 0.3 --bio_weight 0.7
```

**Recommendation**: Start with 70/30 (autoencoder/biological) and adjust based on validation results.

## Interpreting Results

### Clustering Metrics

- **Silhouette Score** (higher is better, range [-1, 1])
  - \> 0.5: Good separation
  - 0.25-0.5: Moderate separation
  - < 0.25: Poor separation

- **Davies-Bouldin Index** (lower is better)
  - < 1.0: Excellent
  - 1.0-2.0: Good
  - \> 2.0: Poor

- **Cross-validation ARI** (higher is better, range [0, 1])
  - \> 0.8: Excellent stability
  - 0.6-0.8: Good stability
  - < 0.6: Poor stability

### Biological Quality Scores

Each cluster receives a quality score based on:
- Duration within expected range (15-500ms)
- Peak frequency in ripple band (125-250 Hz)
- Positive MUA-ripple correlation

**Interpretation**:
- ≥ 80%: Excellent - Likely true SWRs
- 60-80%: Good - Consistent with SWRs
- < 60%: Poor - May contain artifacts

## File Structure

```
project/
├── feature_extraction.py              # Biological feature extraction
├── improved_autoencoder.py            # Advanced architectures
├── advanced_clustering.py             # Optimal clustering methods
├── biological_validation.py           # Validation and interpretation
│
├── generate_all_spectrograms_improved.py  # Step 1
├── train_autoencoder_improved.py          # Step 2
├── cluster_events_improved.py             # Step 3
├── evaluate_clusters_improved.py          # Step 4
│
├── all_spectrograms/                  # Generated spectrogram images
├── detected_events.pkl                # Detected SWR events
├── biological_features.pkl            # Extracted features
├── autoencoder_model_*.pth           # Trained models
├── events_with_clusters_*.pkl        # Clustered results
└── cluster_validation_report_*.txt   # Final report
```

## Troubleshooting

### Model not learning (flat reconstructions)

1. Check data quality: `validate_biological_features()`
2. Reduce learning rate: `--lr 0.0001`
3. Train longer: `--epochs 25`
4. Try different architecture: `--arch vae`

### Poor clustering (low Silhouette score)

1. Try different feature weights: `--ae_weight 0.5 --bio_weight 0.5`
2. Use only autoencoder features: `--no_combined`
3. Adjust k range: `--k_min 2 --k_max 8`
4. Try VAE for better latent structure

### Clusters don't make biological sense

1. Check biological validation scores
2. Increase biological feature weight: `--bio_weight 0.5`
3. Examine individual clusters in evaluation output
4. Consider stricter SWR detection parameters

## Advanced Usage

### Custom Feature Selection

Edit `feature_extraction.py` to add domain-specific features:

```python
def extract_swr_features(event, lfp_array, mua_vec, fs, region_lfp=None):
    features = {}
    
    # Add your custom features here
    features['custom_metric'] = compute_custom_metric(event)
    
    return features
```

### Ensemble Clustering

Combine multiple clustering methods:

```python
from advanced_clustering import find_optimal_clusters

best, results_df = find_optimal_clusters(
    features,
    methods=['kmeans', 'hierarchical', 'gmm', 'dbscan']
)
```

### Export for Further Analysis

```python
import pandas as pd

# Load clustered events
with open('events_with_clusters_combined.pkl', 'rb') as f:
    events = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(events)

# Export to CSV
df.to_csv('clustered_swr_events.csv', index=False)
```

## Citation

If you use this improved pipeline in your research, please cite the relevant methods:

- Autoencoder architecture improvements
- VAE for unsupervised learning
- Silhouette analysis for optimal k
- Biological feature engineering for neuroscience

## Future Improvements

Potential enhancements:
1. **Contrastive learning** for better feature representations
2. **Semi-supervised learning** with expert labels
3. **Hierarchical clustering** with automatic depth selection
4. **Real-time classification** for closed-loop experiments
5. **Multi-region synchronization** features
6. **Transfer learning** across animals/sessions

## Support

For issues or questions:
1. Check the validation report for cluster quality
2. Review the biological validation metrics
3. Examine the feature space visualizations
4. Consider adjusting feature weights or architecture
