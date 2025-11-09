"""
Biological validation and interpretation of SWR clusters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def validate_clusters_biologically(events_df, feature_names=None):
    """
    Validate clusters using known SWR biological properties.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame with events and cluster assignments
    feature_names : list, optional
        List of feature names for detailed analysis
    
    Returns:
    --------
    validation_results : dict
        Dictionary of validation metrics per cluster
    """
    
    if 'cluster_id' not in events_df.columns:
        print("Error: No cluster_id column found")
        return None
    
    validation_results = {}
    cluster_ids = sorted(events_df['cluster_id'].unique())
    
    print("\n" + "="*70)
    print("BIOLOGICAL VALIDATION REPORT")
    print("="*70)
    
    for cluster_id in cluster_ids:
        cluster_events = events_df[events_df['cluster_id'] == cluster_id]
        n_events = len(cluster_events)
        
        print(f"\n{'─'*70}")
        print(f"CLUSTER {cluster_id} (n={n_events} events)")
        print(f"{'─'*70}")
        
        results = {}
        results['n_events'] = n_events
        
        # 1. Duration Analysis
        if 'duration' in cluster_events.columns:
            durations = cluster_events['duration'].values
            results['duration_mean'] = np.mean(durations)
            results['duration_std'] = np.std(durations)
            results['duration_cv'] = np.std(durations) / (np.mean(durations) + 1e-10)
            results['duration_range'] = (np.min(durations), np.max(durations))
            
            # Check if durations are in expected range (15-500ms)
            in_range = np.sum((durations >= 0.015) & (durations <= 0.5))
            results['duration_in_range_pct'] = 100 * in_range / len(durations)
            
            print(f"\n1. Duration Analysis:")
            print(f"   Mean: {results['duration_mean']*1000:.1f} ms")
            print(f"   Std:  {results['duration_std']*1000:.1f} ms")
            print(f"   CV:   {results['duration_cv']:.3f}")
            print(f"   Range: {results['duration_range'][0]*1000:.1f} - {results['duration_range'][1]*1000:.1f} ms")
            print(f"   In expected range (15-500ms): {results['duration_in_range_pct']:.1f}%")
            
            if results['duration_cv'] < 0.3:
                print(f"   ✓ Low variability - consistent durations")
            elif results['duration_cv'] > 0.6:
                print(f"   ⚠️  High variability - heterogeneous durations")
        
        # 2. Ripple Frequency Analysis
        if 'ripple_peak_freq' in cluster_events.columns:
            peak_freqs = cluster_events['ripple_peak_freq'].values
            peak_freqs = peak_freqs[peak_freqs > 0]  # Remove zeros
            
            if len(peak_freqs) > 0:
                results['peak_freq_mean'] = np.mean(peak_freqs)
                results['peak_freq_std'] = np.std(peak_freqs)
                
                # Check if in ripple band (125-250 Hz)
                in_ripple_band = np.sum((peak_freqs >= 125) & (peak_freqs <= 250))
                results['ripple_band_pct'] = 100 * in_ripple_band / len(peak_freqs)
                
                print(f"\n2. Ripple Frequency Analysis:")
                print(f"   Mean peak frequency: {results['peak_freq_mean']:.1f} Hz")
                print(f"   Std: {results['peak_freq_std']:.1f} Hz")
                print(f"   In ripple band (125-250 Hz): {results['ripple_band_pct']:.1f}%")
                
                if results['ripple_band_pct'] > 80:
                    print(f"   ✓ Excellent - most events in canonical ripple band")
                elif results['ripple_band_pct'] > 50:
                    print(f"   ✓ Good - majority in ripple band")
                else:
                    print(f"   ⚠️  WARNING - <50% events in ripple frequency band")
        
        # 3. MUA-Ripple Correlation Analysis
        if 'ripple_mua_correlation' in cluster_events.columns:
            mua_corrs = cluster_events['ripple_mua_correlation'].values
            mua_corrs = mua_corrs[~np.isnan(mua_corrs)]
            
            if len(mua_corrs) > 0:
                results['mean_mua_corr'] = np.mean(mua_corrs)
                results['std_mua_corr'] = np.std(mua_corrs)
                results['positive_corr_pct'] = 100 * np.sum(mua_corrs > 0) / len(mua_corrs)
                
                print(f"\n3. MUA-Ripple Coupling:")
                print(f"   Mean correlation: {results['mean_mua_corr']:.3f}")
                print(f"   Std: {results['std_mua_corr']:.3f}")
                print(f"   Positive correlations: {results['positive_corr_pct']:.1f}%")
                
                if results['mean_mua_corr'] > 0.3:
                    print(f"   ✓ Strong positive coupling - characteristic of true SWRs")
                elif results['mean_mua_corr'] > 0:
                    print(f"   ✓ Positive coupling present")
                else:
                    print(f"   ⚠️  WARNING - Negative mean correlation (unusual for SWRs)")
        
        # 4. Ripple Power Analysis
        if 'ripple_power' in cluster_events.columns:
            powers = cluster_events['ripple_power'].values
            powers = powers[powers > 0]
            
            if len(powers) > 0:
                results['mean_ripple_power'] = np.mean(powers)
                results['std_ripple_power'] = np.std(powers)
                
                print(f"\n4. Ripple Power:")
                print(f"   Mean: {results['mean_ripple_power']:.3f}")
                print(f"   Std:  {results['std_ripple_power']:.3f}")
        
        # 5. Inter-Event Interval Analysis
        if 'start_time' in cluster_events.columns:
            start_times = sorted(cluster_events['start_time'].values)
            if len(start_times) > 1:
                ieis = np.diff(start_times)
                results['median_iei'] = np.median(ieis)
                results['mean_iei'] = np.mean(ieis)
                results['min_iei'] = np.min(ieis)
                
                # Check for suspiciously short IEIs (<100ms)
                short_ieis = np.sum(ieis < 0.1)
                results['short_iei_pct'] = 100 * short_ieis / len(ieis)
                
                print(f"\n5. Inter-Event Intervals:")
                print(f"   Median: {results['median_iei']:.3f} s")
                print(f"   Mean:   {results['mean_iei']:.3f} s")
                print(f"   Min:    {results['min_iei']*1000:.1f} ms")
                print(f"   IEIs < 100ms: {results['short_iei_pct']:.1f}%")
                
                if results['short_iei_pct'] > 10:
                    print(f"   ⚠️  Many short IEIs - possible detection artifacts")
        
        # 6. Spectral Characteristics
        if 'spec_entropy' in cluster_events.columns:
            entropies = cluster_events['spec_entropy'].values
            entropies = entropies[entropies > 0]
            
            if len(entropies) > 0:
                results['mean_spec_entropy'] = np.mean(entropies)
                results['std_spec_entropy'] = np.std(entropies)
                
                print(f"\n6. Spectral Entropy:")
                print(f"   Mean: {results['mean_spec_entropy']:.3f}")
                print(f"   Std:  {results['std_spec_entropy']:.3f}")
        
        # 7. Overall Quality Assessment
        print(f"\n7. Quality Assessment:")
        quality_score = 0
        max_score = 0
        
        if 'duration_in_range_pct' in results:
            max_score += 1
            if results['duration_in_range_pct'] > 80:
                quality_score += 1
                print(f"   ✓ Duration within expected range")
            else:
                print(f"   ✗ Duration outside expected range")
        
        if 'ripple_band_pct' in results:
            max_score += 1
            if results['ripple_band_pct'] > 60:
                quality_score += 1
                print(f"   ✓ Peak frequencies in ripple band")
            else:
                print(f"   ✗ Peak frequencies outside ripple band")
        
        if 'mean_mua_corr' in results:
            max_score += 1
            if results['mean_mua_corr'] > 0:
                quality_score += 1
                print(f"   ✓ Positive MUA-ripple coupling")
            else:
                print(f"   ✗ Negative MUA-ripple coupling")
        
        if max_score > 0:
            results['quality_score'] = quality_score / max_score
            print(f"\n   Overall Quality Score: {quality_score}/{max_score} ({results['quality_score']*100:.0f}%)")
            
            if results['quality_score'] >= 0.8:
                print(f"   ✓✓ Excellent - Likely true SWRs")
            elif results['quality_score'] >= 0.6:
                print(f"   ✓ Good - Consistent with SWRs")
            else:
                print(f"   ⚠️  Poor - May contain artifacts or non-SWR events")
        
        validation_results[f'cluster_{cluster_id}'] = results
    
    print("\n" + "="*70 + "\n")
    
    return validation_results


def compare_clusters_statistically(events_df, feature_cols=None):
    """
    Perform statistical tests to compare clusters.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame with events and features
    feature_cols : list, optional
        List of feature columns to compare
    
    Returns:
    --------
    comparison_results : dict
        Statistical comparison results
    """
    
    if feature_cols is None:
        # Use common biological features
        feature_cols = [col for col in events_df.columns if any(
            key in col for key in ['duration', 'ripple', 'mua', 'power', 'freq']
        )]
    
    cluster_ids = sorted(events_df['cluster_id'].unique())
    comparison_results = {}
    
    print("\n" + "="*70)
    print("STATISTICAL CLUSTER COMPARISON")
    print("="*70)
    
    for feature in feature_cols:
        if feature not in events_df.columns:
            continue
        
        print(f"\n{feature}:")
        
        # Collect data per cluster
        cluster_data = []
        for cid in cluster_ids:
            data = events_df[events_df['cluster_id'] == cid][feature].values
            data = data[~np.isnan(data)]
            cluster_data.append(data)
        
        # ANOVA test
        if len(cluster_data) > 1 and all(len(d) > 0 for d in cluster_data):
            try:
                f_stat, p_val = stats.f_oneway(*cluster_data)
                comparison_results[feature] = {
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
                
                print(f"  ANOVA: F={f_stat:.3f}, p={p_val:.4f}", end="")
                if p_val < 0.001:
                    print(" ***")
                elif p_val < 0.01:
                    print(" **")
                elif p_val < 0.05:
                    print(" *")
                else:
                    print(" (ns)")
                
                # Pairwise comparisons if significant
                if p_val < 0.05 and len(cluster_ids) <= 10:
                    print("  Pairwise comparisons:")
                    for i, cid1 in enumerate(cluster_ids):
                        for cid2 in cluster_ids[i+1:]:
                            data1 = cluster_data[i]
                            data2 = cluster_data[cluster_ids.index(cid2)]
                            t_stat, p_val_pair = stats.ttest_ind(data1, data2)
                            if p_val_pair < 0.05:
                                print(f"    Cluster {cid1} vs {cid2}: p={p_val_pair:.4f} *")
            except Exception as e:
                print(f"  Could not perform ANOVA: {e}")
    
    print("\n" + "="*70 + "\n")
    
    return comparison_results


def plot_cluster_characteristics(events_df, save_prefix='cluster'):
    """
    Create comprehensive visualization of cluster characteristics.
    """
    
    cluster_ids = sorted(events_df['cluster_id'].unique())
    n_clusters = len(cluster_ids)
    
    # Figure 1: Distribution of key features
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    features_to_plot = [
        ('duration', 'Duration (s)', False),
        ('ripple_peak_freq', 'Peak Frequency (Hz)', False),
        ('ripple_power', 'Ripple Power', True),
        ('ripple_mua_correlation', 'Ripple-MUA Correlation', False)
    ]
    
    for ax, (feature, label, use_log) in zip(axes.flat, features_to_plot):
        if feature in events_df.columns:
            data = [events_df[events_df['cluster_id'] == cid][feature].dropna().values 
                    for cid in cluster_ids]
            
            if use_log:
                data = [np.log10(d + 1) for d in data]
                label = f'Log10({label})'
            
            bp = ax.boxplot(data, labels=cluster_ids, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel(label)
            ax.set_title(f'Distribution of {label}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_characteristics_boxplots.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved boxplots to '{save_prefix}_characteristics_boxplots.png'")
    plt.show()
    
    # Figure 2: Cluster size and quality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cluster sizes
    cluster_sizes = events_df['cluster_id'].value_counts().sort_index()
    axes[0].bar(cluster_sizes.index, cluster_sizes.values, color=plt.cm.Set3(np.linspace(0, 1, n_clusters)))
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Number of Events')
    axes[0].set_title('Cluster Sizes')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Duration vs ripple power scatter
    if 'duration' in events_df.columns and 'ripple_power' in events_df.columns:
        for cid in cluster_ids:
            cluster_data = events_df[events_df['cluster_id'] == cid]
            axes[1].scatter(
                cluster_data['duration']*1000,
                cluster_data['ripple_power'],
                alpha=0.6,
                s=30,
                label=f'Cluster {cid}'
            )
        axes[1].set_xlabel('Duration (ms)')
        axes[1].set_ylabel('Ripple Power')
        axes[1].set_title('Duration vs Ripple Power')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_summary.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved summary plot to '{save_prefix}_summary.png'")
    plt.show()


def export_validation_report(validation_results, events_df, filename='cluster_validation_report.txt'):
    """
    Export detailed validation report to text file.
    """
    
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SWR CLUSTER VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total events: {len(events_df)}\n")
        f.write(f"Number of clusters: {len(events_df['cluster_id'].unique())}\n")
        f.write(f"Cluster sizes: {dict(events_df['cluster_id'].value_counts().sort_index())}\n\n")
        
        # Per-cluster details
        for cluster_name, results in validation_results.items():
            f.write(f"\n{cluster_name.upper()}\n")
            f.write("-"*70 + "\n")
            
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Exported validation report to '{filename}'")
