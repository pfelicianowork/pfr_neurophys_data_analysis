"""
Feature extraction module for SWR analysis.
Extracts biological, spectral, temporal, and spatial features from detected events.
"""

import numpy as np
# scipy.stats was removed (unused import)
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import kurtosis, skew
import warnings

warnings.filterwarnings('ignore')


# --- Helpers: LFP normalization and utility conversions ---
def zscore_lfp(lfp_array):
    """Z-score normalize LFP per-channel (or single-channel).

    Returns a copy of the array normalized along channels when applicable.
    """
    if lfp_array is None:
        return lfp_array
    lfp = np.array(lfp_array, copy=True)
    # 1D: single channel
    if lfp.ndim == 1:
        mean = np.nanmean(lfp)
        std = np.nanstd(lfp) + 1e-10
        return (lfp - mean) / std
    # 2D: channels x time
    mean = np.nanmean(lfp, axis=1, keepdims=True)
    std = np.nanstd(lfp, axis=1, keepdims=True) + 1e-10
    return (lfp - mean) / std


def ms_to_samples(ms, fs):
    """Convert milliseconds to samples given sampling frequency fs (Hz)."""
    return int(np.round(ms / 1000.0 * fs))


def extract_swr_features(event, lfp_array, mua_vec, fs, region_lfp=None, extension_ms=0):
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
    features['duration'] = event.get('duration', 0.0)
    features['start_time'] = event.get('start_time', 0.0)
    features['end_time'] = event.get('end_time', features['start_time'] + features['duration'])
    
    # Peak timing (normalized within event)
    if 'peak_time' in event:
        features['peak_time_normalized'] = (event['peak_time'] - event['start_time']) / event['duration']
    else:
        features['peak_time_normalized'] = 0.5  # Default to middle
    

    # Calculate extension in samples and event indices
    extension_samples = ms_to_samples(extension_ms, fs)
    start_idx = int(features['start_time'] * fs) - extension_samples
    end_idx = int(features['end_time'] * fs) + extension_samples
    start_idx = max(0, start_idx)
    # For LFP, use length check if available
    lfp_len = len(lfp_array) if lfp_array is not None and getattr(lfp_array, 'ndim', 1) == 1 else (lfp_array.shape[1] if lfp_array is not None else 0)
    end_idx = min(lfp_len, end_idx) if lfp_len else end_idx

    # Extract the LFP segment for this event (with extension)
    if lfp_array is not None:
        if getattr(lfp_array, 'ndim', 1) == 1:
            lfp_seg = lfp_array[start_idx:end_idx]
        else:
            lfp_seg = lfp_array[0, start_idx:end_idx]
    else:
        lfp_seg = np.array([])

    # Filter into ripple band (125-250 Hz)
    if len(lfp_seg) > 4:
        try:
            b, a = butter(3, [125, 250], btype='band', fs=fs)
            ripple_filt = filtfilt(b, a, lfp_seg)
            analytic = hilbert(ripple_filt)
            envelope = np.abs(analytic)
            envelope_power = envelope ** 2
            # Normalize envelope power (z-score)
            env_mean = np.mean(envelope_power)
            env_std = np.std(envelope_power) + 1e-10
            envelope_power_norm = (envelope_power - env_mean) / env_std

            # Ripple power: mean of normalized envelope power
            features['ripple_power'] = float(np.mean(envelope_power_norm))

            # Ripple amplitude: max of envelope
            features['ripple_amplitude'] = float(np.max(envelope))

            # Envelope asymmetry
            peak_idx = np.argmax(envelope)
            rise_time = peak_idx / fs
            decay_time = (len(envelope) - peak_idx - 1) / fs
            if (decay_time + rise_time) > 0:
                features['envelope_asymmetry'] = float((decay_time - rise_time) / (decay_time + rise_time))
            else:
                features['envelope_asymmetry'] = 0.0

            # Envelope kurtosis
            features['envelope_kurtosis'] = float(kurtosis(envelope_power_norm))

            # Envelope area
            features['envelope_area'] = float(np.trapz(envelope_power_norm))

            # Envelope curvature (second derivative)
            if len(envelope_power_norm) > 2:
                curvature = np.diff(envelope_power_norm, n=2)
                features['envelope_curvature'] = float(np.mean(np.abs(curvature)))
            else:
                features['envelope_curvature'] = 0.0

            # Peak frequency from spectrogram if available (legacy)
            if 'spectrogram' in event and event['spectrogram'] is not None:
                spec = event['spectrogram']
                freqs = event.get('spectrogram_freqs')
                if freqs is not None and isinstance(spec, np.ndarray) and spec.size > 0:
                    if spec.ndim == 2:
                        freq_profile = np.mean(spec, axis=1)
                    elif spec.ndim == 1:
                        freq_profile = spec
                    else:
                        freq_profile = None
                    if freq_profile is not None and len(freq_profile) == len(freqs):
                        peak_freq_idx = np.argmax(freq_profile)
                        features['ripple_peak_freq'] = float(freqs[peak_freq_idx])
                    else:
                        features['ripple_peak_freq'] = float(event.get('peak_frequency', 150))
                else:
                    features['ripple_peak_freq'] = float(event.get('peak_frequency', 150))
            else:
                features['ripple_peak_freq'] = float(event.get('peak_frequency', 150))

            # In ripple band
            peak_freq = features['ripple_peak_freq']
            features['in_ripple_band'] = 1.0 if (125 <= peak_freq <= 250) else 0.0

            # --- 9. Slow Oscillation (0.5-4 Hz) Phase Features ---
            # Compute phase at event peak and at ripple peak
            try:
                b_slow, a_slow = butter(3, [0.5, 4], btype='band', fs=fs)
                slow_filt = filtfilt(b_slow, a_slow, lfp_seg)
                analytic_slow = hilbert(slow_filt)
                slow_phase = np.angle(analytic_slow)
                # Phase at event peak (center of segment)
                event_peak_idx = len(slow_phase) // 2
                features['slow_wave_phase_at_event_peak'] = float(slow_phase[event_peak_idx]) if len(slow_phase) > 0 else 0.0
                # Phase at ripple peak (use envelope from ripple band)
                if len(envelope) == len(slow_phase):
                    ripple_peak_idx = int(np.argmax(envelope))
                    if 0 <= ripple_peak_idx < len(slow_phase):
                        features['slow_wave_phase_at_ripple_peak'] = float(slow_phase[ripple_peak_idx])
                    else:
                        features['slow_wave_phase_at_ripple_peak'] = 0.0
                else:
                    features['slow_wave_phase_at_ripple_peak'] = 0.0
            except Exception:
                features['slow_wave_phase_at_event_peak'] = 0.0
                features['slow_wave_phase_at_ripple_peak'] = 0.0

        except Exception:
            features['ripple_power'] = 0.0
            features['ripple_amplitude'] = 0.0
            features['envelope_asymmetry'] = 0.0
            features['envelope_kurtosis'] = 0.0
            features['envelope_area'] = 0.0
            features['envelope_curvature'] = 0.0
            features['ripple_peak_freq'] = float(event.get('peak_frequency', 150))
            features['in_ripple_band'] = 0.0
            features['slow_wave_phase_at_event_peak'] = 0.0
            features['slow_wave_phase_at_ripple_peak'] = 0.0
    else:
        features['ripple_power'] = 0.0
        features['ripple_amplitude'] = 0.0
        features['envelope_asymmetry'] = 0.0
        features['envelope_kurtosis'] = 0.0
        features['envelope_area'] = 0.0
        features['envelope_curvature'] = 0.0
        features['ripple_peak_freq'] = float(event.get('peak_frequency', 150))
        features['in_ripple_band'] = 0.0
        features['slow_wave_phase_at_event_peak'] = 0.0
        features['slow_wave_phase_at_ripple_peak'] = 0.0
    
    # --- 3. MUA Features ---
    # Calculate extension in samples
    extension_samples = ms_to_samples(extension_ms, fs)
    start_idx = int(features['start_time'] * fs) - extension_samples
    end_idx = int(features['end_time'] * fs) + extension_samples

    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(mua_vec), end_idx)

    if start_idx < end_idx:
        mua_segment = mua_vec[start_idx:end_idx]
        features['mua_max'] = float(np.max(mua_segment))
        features['mua_mean'] = float(np.mean(mua_segment))
        features['mua_std'] = float(np.std(mua_segment))
        features['mua_integral'] = float(np.trapz(mua_segment))
        features['mua_peak_time'] = float(np.argmax(mua_segment) / max(1, len(mua_segment)))
    else:
        mua_segment = np.array([])
        features['mua_max'] = 0.0
        features['mua_mean'] = 0.0
        features['mua_std'] = 0.0
        features['mua_integral'] = 0.0
        features['mua_peak_time'] = 0.5
    
    # --- NEW: Sharp-wave (0.5-20 Hz) features ---
    try:
        # Prefer using the averaged / first channel if multichannel provided
        if lfp_array is None:
            sw_seg = np.array([])
        else:
            if getattr(lfp_array, 'ndim', 1) == 1:
                lfp_for_sw = lfp_array
            else:
                # Take first channel as canonical slow-wave reference
                lfp_for_sw = lfp_array[0]

            # Band for sharp-wave: use 0.5-20 Hz (avoid exact 0 Hz DC)
            low_sw, high_sw = 0.5, 20.0
            b_sw, a_sw = butter(3, [low_sw / (fs/2), high_sw / (fs/2)], btype='band')
            sw_filtered = filtfilt(b_sw, a_sw, lfp_for_sw)
            sw_seg = sw_filtered[start_idx:end_idx] if end_idx > start_idx and end_idx <= len(sw_filtered) else sw_filtered[start_idx: min(len(sw_filtered), end_idx)]

        if len(sw_seg) > 2:
            # Analytic envelope
            sw_env = np.abs(hilbert(sw_seg))
            features['sharpwave_peak_amp'] = float(np.max(sw_seg))
            features['sharpwave_peak_amp_abs'] = float(np.max(np.abs(sw_seg)))
            features['sharpwave_env_mean'] = float(np.mean(sw_env))
            features['sharpwave_area'] = float(np.trapz(np.abs(sw_seg)))

            # Sharp-wave symmetry: (decay_time - rise_time) / (decay_time + rise_time)
            peak_idx = int(np.argmax(np.abs(sw_seg)))
            rise_time = peak_idx / fs
            decay_time = (len(sw_seg) - peak_idx - 1) / fs
            if (decay_time + rise_time) > 0:
                features['sharpwave_symmetry'] = float((decay_time - rise_time) / (decay_time + rise_time))
            else:
                features['sharpwave_symmetry'] = 0.0

            # Sharp-wave curvature (second derivative)
            if len(sw_seg) > 2:
                curvature = np.diff(sw_seg, n=2)
                features['sharpwave_curvature'] = float(np.mean(np.abs(curvature)))
            else:
                features['sharpwave_curvature'] = 0.0

            # Sharp-wave kurtosis
            from scipy.stats import kurtosis as _kurtosis
            features['sharpwave_kurtosis'] = float(_kurtosis(sw_seg)) if len(sw_seg) > 3 else 0.0

            # slope (linear fit)
            try:
                from scipy import stats as _stats
                tvec = np.arange(len(sw_seg)) / fs
                slope = float(_stats.linregress(tvec, sw_seg).slope)
                features['sharpwave_slope'] = slope
            except Exception:
                features['sharpwave_slope'] = 0.0

            # polarity: +1 (positive deflection) or -1 (negative)
            features['sharpwave_polarity'] = float(np.sign(np.mean(sw_seg)))

            # peak latency relative to event (fraction)
            features['sharpwave_peak_latency'] = float(peak_idx) / max(1, len(sw_seg))

            # attach envelope into the event dict for downstream use (PAC, plotting)
            try:
                event['sharpwave_envelope'] = sw_env.tolist()
            except Exception:
                event['sharpwave_envelope'] = np.array(sw_env)
        else:
            features['sharpwave_peak_amp'] = 0.0
            features['sharpwave_peak_amp_abs'] = 0.0
            features['sharpwave_env_mean'] = 0.0
            features['sharpwave_mean_power'] = 0.0
            features['sharpwave_area'] = 0.0
            features['sharpwave_slope'] = 0.0
            features['sharpwave_polarity'] = 0.0
            features['sharpwave_peak_latency'] = 0.5
            event['sharpwave_envelope'] = []
    except Exception:
        # fail-safe: populate zeros so feature vector keeps consistent length
        features['sharpwave_peak_amp'] = 0.0
        features['sharpwave_peak_amp_abs'] = 0.0
        features['sharpwave_env_mean'] = 0.0
        features['sharpwave_mean_power'] = 0.0
        features['sharpwave_area'] = 0.0
        features['sharpwave_slope'] = 0.0
        features['sharpwave_polarity'] = 0.0
        features['sharpwave_peak_latency'] = 0.5
        event['sharpwave_envelope'] = []
    # --- NEW: Sharp-wave phase at ripple peak and basic PAC (Modulation Index) ---
    try:
        # Need both sharp-wave phase and ripple envelope aligned to the event
        sw_phase = None
        rip_seg = None
        if 'sharpwave_envelope' in event and getattr(event['sharpwave_envelope'], '__len__', None) and len(event['sharpwave_envelope']) >= (end_idx - start_idx):
            # If we have stored sw_filtered above, compute phase from sw_seg; else try to reconstruct
            if 'sw_seg' in locals() and len(sw_seg) == (end_idx - start_idx):
                sw_phase = np.angle(hilbert(sw_seg))
            else:
                # fallback: compute phase from envelope by Hilbert on smoothed envelope (least preferred)
                try:
                    sw_env = np.array(event['sharpwave_envelope'])
                    sw_phase = np.angle(hilbert(sw_env))
                except Exception:
                    sw_phase = None

        if 'ripple_envelope' in event and getattr(event['ripple_envelope'], '__len__', None) and len(event['ripple_envelope']) > 0:
            rip_env = np.array(event['ripple_envelope'])
            rip_seg = rip_env[start_idx:end_idx] if len(rip_env) >= end_idx else rip_env[start_idx: min(len(rip_env), end_idx)]

        # Phase at ripple peak (sharp-wave phase)
        if sw_phase is not None and rip_seg is not None and len(rip_seg) == len(sw_phase) and len(rip_seg) > 0:
            peak_idx = int(np.argmax(rip_seg))
            if 0 <= peak_idx < len(sw_phase):
                features['sharpwave_phase_at_ripple_peak'] = float(sw_phase[peak_idx])
            else:
                features['sharpwave_phase_at_ripple_peak'] = 0.0

            # Compute Tort-style Modulation Index (MI) with 18 phase bins
            try:
                n_bins = 18
                bins = np.linspace(-np.pi, np.pi, n_bins + 1)
                inds = np.digitize(sw_phase, bins) - 1
                mean_amp = np.zeros(n_bins, dtype=float)
                for b in range(n_bins):
                    mask = inds == b
                    if np.any(mask):
                        mean_amp[b] = np.mean(rip_seg[mask])
                    else:
                        mean_amp[b] = 0.0
                mean_amp += 1e-10
                mean_amp /= mean_amp.sum()
                uniform = np.ones(n_bins) / n_bins
                kl = np.sum(mean_amp * np.log(mean_amp / uniform))
                mi = kl / np.log(n_bins)
                features['sharpwave_ripple_modulation_index'] = float(mi)
            except Exception:
                features['sharpwave_ripple_modulation_index'] = 0.0
        else:
            features['sharpwave_phase_at_ripple_peak'] = 0.0
            features['sharpwave_ripple_modulation_index'] = 0.0
    except Exception:
        features['sharpwave_phase_at_ripple_peak'] = 0.0
        features['sharpwave_ripple_modulation_index'] = 0.0
    
    # --- 4. Ripple-MUA Coupling & contextual envelope features ---
    if 'ripple_envelope' in event and hasattr(event['ripple_envelope'], '__len__') and len(event['ripple_envelope']) > 0:
        env = np.array(event['ripple_envelope'])
        # event envelope segment
        env_seg = env[start_idx:end_idx] if len(env) >= end_idx else env[start_idx: min(len(env), end_idx)]
        if len(env_seg) == len(mua_segment) and len(mua_segment) > 1:
            try:
                corr = np.corrcoef(env_seg, mua_segment)[0, 1]
                features['ripple_mua_correlation'] = float(corr) if not np.isnan(corr) else 0.0
            except Exception:
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
            # Frequency chirp computed from spectrogram (Hz per second)
            features['frequency_chirp'] = float(event.get('frequency_chirp', np.nan))
        else:
            features['spec_entropy'] = 0.0
            features['spec_peak_power'] = 0.0
            features['spec_mean_power'] = 0.0
            features['spec_centroid'] = 0.0
            features['spec_spread'] = 0.0
            features['frequency_chirp'] = 0.0
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
    except Exception:
        features['slow_phase_at_peak'] = 0.0
        features['phase_locking_value'] = 0.0
    
    # --- 8. Multi-Channel Spatial Features ---
    if region_lfp is not None:
        spatial_features = extract_multichannel_features(event, region_lfp, fs, extension_ms=extension_ms)
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


def extract_multichannel_features(event, region_lfp, fs, extension_ms=0):
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
    
    extension_samples = ms_to_samples(extension_ms, fs)
    start_idx = int(event['start_time'] * fs) - extension_samples
    end_idx = int(event['end_time'] * fs) + extension_samples
    
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
                    except Exception:
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
                        except Exception:
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


def batch_extract_features(events, lfp_array, mua_vec, fs, region_lfp=None, verbose=True, normalize_lfp=True, extension_ms=0):
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
    
    # Optionally normalize LFP and region_lfp
    if normalize_lfp and lfp_array is not None:
        lfp_array = zscore_lfp(lfp_array)
    if normalize_lfp and region_lfp is not None:
        norm_region = {}
        for k, v in region_lfp.items():
            norm_region[k] = zscore_lfp(v)
        region_lfp = norm_region

    all_features = []
    feature_names = None

    for i, event in enumerate(events):
        if verbose and i % 100 == 0:
            print(f"Extracting features: {i}/{len(events)}...")

        features = extract_swr_features(event, lfp_array, mua_vec, fs, region_lfp, extension_ms=extension_ms)

        if feature_names is None:
            feature_names = sorted(features.keys())

        # Create feature vector in consistent order, force scalar conversion and debug non-scalars
        feature_vector = []
        for j, name in enumerate(feature_names):
            val = features[name]
            if isinstance(val, (list, tuple, np.ndarray)):
                print(f"Non-scalar feature at event {i}, feature '{name}': type={type(val)}, value={val}")
            try:
                feature_vector.append(float(val))
            except Exception:
                feature_vector.append(np.nan)
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
        
        # Check ranges for key features (always show, not just warnings)
        if 'ripple_peak_freq' in name:
            in_range = np.sum((values >= 100) & (values <= 300))
            out_of_range = len(values) - in_range
            if out_of_range > len(values) * 0.5:
                print(f"⚠️  {name}: {in_range}/{len(values)} in range (100-300 Hz), {out_of_range}/{len(values)} outside")
            else:
                print(f"✓  {name}: {in_range}/{len(values)} in range (100-300 Hz), {out_of_range}/{len(values)} outside")
        
        if 'duration' in name:
            in_range = np.sum((values >= 0.015) & (values <= 0.5))
            out_of_range = len(values) - in_range
            if out_of_range > len(values) * 0.1:
                print(f"⚠️  {name}: {in_range}/{len(values)} in range (15-500ms), {out_of_range}/{len(values)} outside")
            else:
                print(f"✓  {name}: {in_range}/{len(values)} in range (15-500ms), {out_of_range}/{len(values)} outside")
        
        if 'ripple_power' in name and 'log' not in name:
            # Show distribution stats for ripple power
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"ℹ️  {name}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{np.min(values):.3f}, {np.max(values):.3f}]")
        
        if 'mua_max' in name or 'mua_mean' in name:
            # Show distribution stats for MUA features
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"ℹ️  {name}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{np.min(values):.3f}, {np.max(values):.3f}]")
    
    print("Validation complete.\n")
