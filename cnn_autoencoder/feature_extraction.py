"""
Feature extraction module for SWR analysis.
Extracts biological, spectral, temporal, and spatial features from detected events.
"""

import numpy as np
import scipy.stats
from scipy.signal import hilbert, butter, filtfilt, welch
from scipy.stats import kurtosis, skew
import warnings

warnings.filterwarnings('ignore')


def extract_swr_features(event, lfp_array, mua_vec, fs, region_lfp= None):
    """
    Extract comprehensive SWR features beyond spectrograms.
    
    Parameters:
    -----------
    event : dict
        Event dictionary containing timing and detection info
    lfp_array : np.ndarray
        LFP signal array (1D or 2D if multichannel)
    mua_vec : np.ndarray
        Multi-unit activity vector
    fs : float
        Sampling frequency
    region_lfp : dict, optional
        Dictionary containing multi-channel LFP data
    
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    
    features = {}
    
    # --- 1. Temporal Features ---
    features['duration'] = event['duration']
    features['start_time'] = event['start_time']
    features['end_time'] = event['end_time']
    
    # Peak timing (normalized within event)
    if 'peak_time' in event:
        features['peak_time_normalized'] = (event['peak_time'] - event['start_time']) / event['duration']
    else:
        features['peak_time_normalized'] = 0.5  # Default to middle
    
    # --- 2. Ripple Band Features (125-250 Hz) ---
    ripple_power = event.get('ripple_power', 0)
    features['ripple_power'] = ripple_power
    features['ripple_power_log'] = np.log1p(ripple_power)
    features['ripple_peak_freq'] = event.get('peak_frequency', 0)
    features['ripple_amplitude'] = event.get('ripple_amplitude', 0)
    
    # Ripple frequency band percentage
    if 'peak_frequency' in event:
        peak_freq = event['peak_frequency']
        features['in_ripple_band'] = 1.0 if (125 <= peak_freq <= 250) else 0.0
    else:
        features['in_ripple_band'] = 0.0
    
    # --- 3. MUA Features ---
    start_idx = int(event['start_time'] * fs)
    end_idx = int(event['end_time'] * fs)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(mua_vec), end_idx)
    
    if start_idx < end_idx:
        mua_segment = mua_vec[start_idx:end_idx]
        features['mua_max'] = np.max(mua_segment)
        features['mua_mean'] = np.mean(mua_segment)
        features['mua_std'] = np.std(mua_segment)
        features['mua_integral'] = np.trapz(mua_segment)
        features['mua_peak_time'] = np.argmax(mua_segment) / len(mua_segment)
    else:
        features['mua_max'] = 0.0
        features['mua_mean'] = 0.0
        features['mua_std'] = 0.0
        features['mua_integral'] = 0.0
        features['mua_peak_time'] = 0.5
    
    # --- 4. Ripple-MUA Coupling ---
    if 'ripple_envelope' in event and len(event['ripple_envelope']) > 0:
        ripple_env = event['ripple_envelope'][start_idx:end_idx]
        if len(ripple_env) == len(mua_segment) and len(mua_segment) > 1:
            try:
                corr = np.corrcoef(ripple_env, mua_segment)[0, 1]
                features['ripple_mua_correlation'] = corr if not np.isnan(corr) else 0.0
            except:
                features['ripple_mua_correlation'] = 0.0
        else:
            features['ripple_mua_correlation'] = 0.0
    else:
        features['ripple_mua_correlation'] = 0.0
    
    # --- 5. Spectral Features from Spectrogram ---
    if 'spectrogram' in event and event['spectrogram'] is not None:
        spec = event['spectrogram']
        
        # Ensure spectrogram is valid
        if isinstance(spec, np.ndarray) and spec.size > 0:
            # Normalize spectrogram
            spec_norm = spec / (np.sum(spec) + 1e-10)
            
            # Spectral entropy
            spec_flat = spec_norm.flatten()
            spec_flat = spec_flat[spec_flat > 0]
            features['spec_entropy'] = -np.sum(spec_flat * np.log(spec_flat + 1e-10))
            
            # Peak power
            features['spec_peak_power'] = np.max(spec)
            features['spec_mean_power'] = np.mean(spec)
            
            # Spectral centroid (frequency domain)
            freq_profile = np.sum(spec, axis=1)
            if np.sum(freq_profile) > 0:
                features['spec_centroid'] = np.sum(np.arange(len(freq_profile)) * freq_profile) / np.sum(freq_profile)
            else:
                features['spec_centroid'] = 0.0
            
            # Spectral spread
            if np.sum(freq_profile) > 0:
                centroid = features['spec_centroid']
                features['spec_spread'] = np.sqrt(np.sum(((np.arange(len(freq_profile)) - centroid)**2) * freq_profile) / np.sum(freq_profile))
            else:
                features['spec_spread'] = 0.0
        else:
            features['spec_entropy'] = 0.0
            features['spec_peak_power'] = 0.0
            features['spec_mean_power'] = 0.0
            features['spec_centroid'] = 0.0
            features['spec_spread'] = 0.0
    else:
        features['spec_entropy'] = 0.0
        features['spec_peak_power'] = 0.0
        features['spec_mean_power'] = 0.0
        features['spec_centroid'] = 0.0
        features['spec_spread'] = 0.0
    
    # --- 6. Waveform Shape Features ---
    if lfp_array.ndim == 1:
        lfp_segment = lfp_array[start_idx:end_idx]
    else:
        # Use first channel if multichannel
        lfp_segment = lfp_array[0, start_idx:end_idx]
    
    if len(lfp_segment) > 3:
        features['lfp_kurtosis'] = kurtosis(lfp_segment)
        features['lfp_skewness'] = skew(lfp_segment)
        features['lfp_peak_to_peak'] = np.ptp(lfp_segment)
        features['lfp_std'] = np.std(lfp_segment)
        features['lfp_rms'] = np.sqrt(np.mean(lfp_segment**2))
    else:
        features['lfp_kurtosis'] = 0.0
        features['lfp_skewness'] = 0.0
        features['lfp_peak_to_peak'] = 0.0
        features['lfp_std'] = 0.0
        features['lfp_rms'] = 0.0
    
    # --- 7. Phase-Amplitude Coupling (Slow Oscillation - Ripple) ---
    try:
        # Need enough data around event for filtering
        margin = int(1.0 * fs)  # 1 second margin
        extended_start = max(0, start_idx - margin)
        extended_end = min(len(lfp_array) if lfp_array.ndim == 1 else lfp_array.shape[1], end_idx + margin)
        
        if lfp_array.ndim == 1:
            extended_lfp = lfp_array[extended_start:extended_end]
        else:
            extended_lfp = lfp_array[0, extended_start:extended_end]
        
        if len(extended_lfp) > 2 * margin:
            # Filter for slow oscillation (1-10 Hz)
            b, a = butter(3, [1, 10], btype='band', fs=fs)
            slow_filt = filtfilt(b, a, extended_lfp)
            
            # Get phase
            analytic = hilbert(slow_filt)
            slow_phase = np.angle(analytic)
            
            # Phase at event peak
            event_offset = start_idx - extended_start
            event_length = end_idx - start_idx
            peak_idx = event_offset + event_length // 2
            
            if 0 <= peak_idx < len(slow_phase):
                features['slow_phase_at_peak'] = slow_phase[peak_idx]
                
                # Phase locking value
                event_phases = slow_phase[event_offset:event_offset + event_length]
                features['phase_locking_value'] = np.abs(np.mean(np.exp(1j * event_phases)))
            else:
                features['slow_phase_at_peak'] = 0.0
                features['phase_locking_value'] = 0.0
        else:
            features['slow_phase_at_peak'] = 0.0
            features['phase_locking_value'] = 0.0
    except Exception as e:
        features['slow_phase_at_peak'] = 0.0
        features['phase_locking_value'] = 0.0
    
    # --- 8. Multi-Channel Spatial Features ---
    if region_lfp is not None:
        spatial_features = extract_multichannel_features(event, region_lfp, fs)
        features.update(spatial_features)
    
    # --- 9. Signal Quality Metrics ---
    if len(lfp_segment) > 10:
        # Signal-to-noise ratio estimate
        signal_power = np.mean(lfp_segment**2)
        noise_est = np.median(np.abs(np.diff(lfp_segment))) / 0.6745  # MAD estimate
        features['snr_estimate'] = signal_power / (noise_est**2 + 1e-10)
    else:
        features['snr_estimate'] = 0.0
    
    return features


def extract_multichannel_features(event, region_lfp, fs):
    """
    Extract features across multiple recording channels.
    
    Parameters:
    -----------
    event : dict
        Event dictionary
    region_lfp : dict
        Dictionary with region names as keys and multi-channel arrays as values
    fs : float
        Sampling frequency
    
    Returns:
    --------
    features : dict
        Dictionary of spatial features
    """
    
    features = {}
    
    start_idx = int(event['start_time'] * fs)
    end_idx = int(event['end_time'] * fs)
    
    # Use CA1 channels (or whichever region is primary)
    if 'CA1' in region_lfp:
        channels = region_lfp['CA1']
        
        # Ensure indices are valid
        start_idx = max(0, start_idx)
        end_idx = min(channels.shape[1], end_idx)
        
        if channels.shape[0] > 1 and start_idx < end_idx:
            # Extract segments from all channels
            channel_segments = channels[:, start_idx:end_idx]
            
            # Compute power for each channel
            channel_powers = np.sum(channel_segments**2, axis=1)
            
            features['spatial_power_variance'] = np.var(channel_powers)
            features['spatial_power_mean'] = np.mean(channel_powers)
            features['max_channel_idx'] = np.argmax(channel_powers) / channels.shape[0]  # Normalized
            features['power_gradient'] = np.max(np.abs(np.diff(channel_powers))) if len(channel_powers) > 1 else 0.0
            
            # Cross-channel correlation
            if channels.shape[0] >= 2:
                seg0 = channel_segments[0, :]
                seg1 = channel_segments[1, :]
                if len(seg0) > 1 and len(seg1) > 1:
                    try:
                        corr = np.corrcoef(seg0, seg1)[0, 1]
                        features['channel_correlation'] = corr if not np.isnan(corr) else 0.0
                    except:
                        features['channel_correlation'] = 0.0
                else:
                    features['channel_correlation'] = 0.0
            else:
                features['channel_correlation'] = 0.0
            
            # Spatial coherence across all channel pairs
            if channels.shape[0] > 2:
                coherences = []
                for i in range(min(3, channels.shape[0])):
                    for j in range(i+1, min(3, channels.shape[0])):
                        try:
                            corr = np.corrcoef(channel_segments[i, :], channel_segments[j, :])[0, 1]
                            if not np.isnan(corr):
                                coherences.append(corr)
                        except:
                            pass
                features['mean_spatial_coherence'] = np.mean(coherences) if coherences else 0.0
            else:
                features['mean_spatial_coherence'] = 0.0
        else:
            # Single channel or invalid indices
            features['spatial_power_variance'] = 0.0
            features['spatial_power_mean'] = 0.0
            features['max_channel_idx'] = 0.0
            features['power_gradient'] = 0.0
            features['channel_correlation'] = 0.0
            features['mean_spatial_coherence'] = 0.0
    else:
        # No multi-channel data
        features['spatial_power_variance'] = 0.0
        features['spatial_power_mean'] = 0.0
        features['max_channel_idx'] = 0.0
        features['power_gradient'] = 0.0
        features['channel_correlation'] = 0.0
        features['mean_spatial_coherence'] = 0.0
    
    return features


def batch_extract_features(events, lfp_array, mua_vec, fs, region_lfp=None, verbose=True):
    """
    Extract features for all events in batch.
    
    Parameters:
    -----------
    events : list
        List of event dictionaries
    lfp_array : np.ndarray
        LFP signal array
    mua_vec : np.ndarray
        MUA vector
    fs : float
        Sampling frequency
    region_lfp : dict, optional
        Multi-channel LFP data
    verbose : bool
        Print progress
    
    Returns:
    --------
    feature_matrix : np.ndarray
        Matrix of extracted features (n_events x n_features)
    feature_names : list
        List of feature names
    """
    
    all_features = []
    feature_names = None
    
    for i, event in enumerate(events):
        if verbose and i % 100 == 0:
            print(f"Extracting features: {i}/{len(events)}...")
        
        features = extract_swr_features(event, lfp_array, mua_vec, fs, region_lfp)
        
        if feature_names is None:
            feature_names = sorted(features.keys())
        
        # Create feature vector in consistent order
        feature_vector = [features[name] for name in feature_names]
        all_features.append(feature_vector)
    
    feature_matrix = np.array(all_features)
    
    if verbose:
        print(f"Extracted {feature_matrix.shape[1]} features from {feature_matrix.shape[0]} events")
    
    return feature_matrix, feature_names


def validate_biological_features(feature_matrix, feature_names):
    """
    Validate that biological features are in expected ranges.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Feature matrix
    feature_names : list
        List of feature names
    """
    
    print("\n--- Biological Feature Validation ---")
    
    for i, name in enumerate(feature_names):
        values = feature_matrix[:, i]
        
        # Check for NaN/Inf
        n_nan = np.isnan(values).sum()
        n_inf = np.isinf(values).sum()
        
        if n_nan > 0 or n_inf > 0:
            print(f"⚠️  {name}: {n_nan} NaNs, {n_inf} Infs")
        
        # Check ranges for key features
        if 'ripple_peak_freq' in name:
            out_of_range = np.sum((values < 100) | (values > 300))
            if out_of_range > len(values) * 0.5:
                print(f"⚠️  {name}: {out_of_range}/{len(values)} values outside typical ripple range (100-300 Hz)")
        
        if 'duration' in name:
            out_of_range = np.sum((values < 0.015) | (values > 0.5))
            if out_of_range > len(values) * 0.1:
                print(f"⚠️  {name}: {out_of_range}/{len(values)} values outside typical SWR duration (15-500ms)")
    
    print("Validation complete.\n")
