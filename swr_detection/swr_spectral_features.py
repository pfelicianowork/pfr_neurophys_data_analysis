"""
swr_spectral_features.py

Utilities for computing and analyzing spectrograms and spectral features of SWR/ripple events.

Functions:
- compute_event_spectral_features
- batch_compute_spectral_features
- average_spectrograms_by_type

Intended for use with SWRHMMDetector and event dictionaries.
"""

import numpy as np
import warnings
from scipy.signal import spectrogram
from scipy.interpolate import interp1d
from scipy.stats import linregress
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, VBox, Output
from IPython.display import display, clear_output
from ipywidgets import interact, IntSlider

# Multitaper spectral estimation using MNE
try:
    from mne.time_frequency import psd_array_multitaper
except ImportError:
    psd_array_multitaper = None
    print("Warning: mne not installed. Multitaper spectral analysis will not work.")

# Optional: Advanced CWT with pyfftw and Dask for parallel processing
try:
    import pyfftw
    import dask
    from dask.diagnostics import ProgressBar
    from multiprocessing import cpu_count
    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False
    warnings.warn("pyfftw or dask not installed. Advanced CWT will fall back to scipy.")

def mtspectrum(lfp_seg, fs, window_size, fpass=(100, 250), average=True, bandwidth=10, ntapers=5, pad=0, **kwargs):
    """
    Multitaper spectral estimation using MNE.
    Returns: S (power), f (freqs), Serr (None), options (dict)
    """
    if psd_array_multitaper is None:
        raise ImportError("mne is required for multitaper spectral analysis.")
    S, f = psd_array_multitaper(
        lfp_seg, sfreq=fs, fmin=fpass[0], fmax=fpass[1],
        bandwidth=bandwidth, adaptive=False, normalization='length', verbose=False
    )
    Serr = None
    options = {'bandwidth': bandwidth, 'ntapers': ntapers, 'pad': pad, 'fpass': fpass}
    return S, f, Serr, options


def _cwt_single_freq_fftw(data_fft, scale, freq, fs, freqs_fft, extend_len, N):
    """
    Compute CWT for a single frequency using pyfftw (for parallel execution).
    
    CORRECTED: Proper Morlet wavelet normalization and scale-frequency relationship.
    
    Parameters:
    -----------
    data_fft : np.ndarray
        FFT of the data
    scale : float
        Wavelet scale (dimensionless)
    freq : float
        Target frequency in Hz
    fs : float
        Sampling rate in Hz
    freqs_fft : np.ndarray
        Frequency vector in Hz (from np.fft.fftfreq)
    extend_len : int
        Boundary extension length
    N : int
        Original data length
        
    Returns:
    --------
    cwt_coef : np.ndarray, shape (N,)
        CWT coefficients for this frequency
    """
    # Morlet wavelet in frequency domain
    # Central frequency f0 = 6 (standard for neuroscience applications)
    f0 = 6.0
    
    # Morlet wavelet: ψ(ω) = π^(-1/4) * exp(-0.5 * (scale*ω - f0)^2)
    # For digital signals: ω_normalized = 2π * f / fs
    # So: scale * ω_normalized = scale * 2π * f / fs
    # We want this to equal f0 when f = target_freq
    # Therefore: scale = f0 * fs / (2π * target_freq)
    
    # Compute wavelet in frequency domain
    psi_fft = np.zeros_like(freqs_fft, dtype='complex128')
    
    # Only compute for positive frequencies
    positive_mask = freqs_fft > 0
    
    # Normalized frequency: ω = 2π * f_Hz / fs (in radians per sample)
    omega_normalized = 2 * np.pi * freqs_fft[positive_mask] / fs
    
    # Wavelet function in frequency domain with proper normalization
    # π^(-1/4) for amplitude normalization
    # sqrt(scale) for energy normalization across scales
    psi_fft[positive_mask] = (np.pi ** -0.25) * np.sqrt(scale) * \
                              np.exp(-0.5 * (scale * omega_normalized - f0) ** 2)
    
    # Convolution in frequency domain (element-wise multiplication)
    conv_fft = data_fft * psi_fft
    
    # Inverse FFT using pyfftw
    out = pyfftw.zeros_aligned(len(conv_fft), dtype='complex128')
    ifft = pyfftw.FFTW(out, out, direction='FFTW_BACKWARD', flags=['FFTW_ESTIMATE'])
    out[:] = conv_fft
    ifft(normalise_idft=True)
    
    # Extract valid region (remove boundary extensions)
    # Return absolute value (magnitude) for power analysis
    return np.abs(out[extend_len:extend_len + N])


def _cwt_optimized(lfp_seg, fs, freqs, method='full', boundary='mirror', n_workers=None, verbose=False):
    """
    Optimized CWT implementation using pyfftw and Dask for parallel processing.
    
    CORRECTED: Proper frequency-scale relationship and validation.
    
    Parameters:
    -----------
    lfp_seg : np.ndarray
        LFP segment to analyze
    fs : float
        Sampling frequency
    freqs : np.ndarray
        Frequencies to analyze (in Hz)
    method : str, optional
        'full' (standard FFT)
    boundary : str, optional
        Boundary handling: 'mirror', 'zeros', or 'periodic'
    n_workers : int, optional
        Number of parallel workers (default: CPU count)
    verbose : bool, optional
        Show progress bar
    
    Returns:
    --------
    cwt_matrix : np.ndarray
        CWT power (freqs × time)
    """
    if not HAS_PYFFTW:
        # Fallback to scipy
        from scipy.signal import morlet2
        try:
            from scipy.signal import cwt
            # Correct scale-frequency relationship for Morlet wavelet
            # scale = f0 * fs / (2 * pi * freq) where f0 = 6
            widths = (6.0 * fs) / (2 * np.pi * freqs)
            cwt_matrix = cwt(lfp_seg, morlet2, widths, w=6.0)
            return np.abs(cwt_matrix)
        except ImportError:
            raise ImportError("scipy.signal.cwt not available. Install pyfftw/dask for optimized CWT or use STFT.")
    
    if n_workers is None:
        n_workers = cpu_count()
    
    N = len(lfp_seg)
    
    # Validate frequency range
    nyquist = fs / 2.0
    if np.any(freqs > nyquist):
        import warnings
        warnings.warn(f"Some frequencies exceed Nyquist limit ({nyquist:.1f} Hz). Filtering...")
        freqs = freqs[freqs <= nyquist]
    
    if np.any(freqs <= 0):
        import warnings
        warnings.warn("Removing non-positive frequencies...")
        freqs = freqs[freqs > 0]
    
    if len(freqs) == 0:
        raise ValueError("No valid frequencies after Nyquist filtering!")
    
    # Boundary extension (10% or full length, whichever is smaller)
    extend_len = min(int(np.ceil(N * 0.1)), N)
    if boundary == 'mirror':
        data = np.hstack((np.flip(lfp_seg[:extend_len]), lfp_seg, np.flip(lfp_seg[-extend_len:])))
    elif boundary == 'zeros':
        data = np.hstack((np.zeros(extend_len), lfp_seg, np.zeros(extend_len)))
    elif boundary == 'periodic':
        data = np.hstack((lfp_seg[-extend_len:], lfp_seg, lfp_seg[:extend_len]))
    else:
        data = lfp_seg
        extend_len = 0
    
    # Compute FFT of data ONCE (shared across all frequencies)
    data_fft = pyfftw.zeros_aligned(data.shape[0], dtype='complex128')
    fft_sig = pyfftw.FFTW(data_fft, data_fft, direction='FFTW_FORWARD', flags=['FFTW_ESTIMATE'])
    data_fft[:] = data
    fft_sig()
    
    # Frequency vector in Hz (for wavelet construction)
    freqs_fft = np.fft.fftfreq(data.shape[0], d=1.0/fs)
    
    # ========== PARALLEL EXECUTION WITH DASK ==========
    task_list = []
    f0 = 6.0  # Morlet central frequency (standard)
    
    for freq in freqs:
        # Scale for this frequency: scale = f0 / (2π * f_normalized)
        # where f_normalized = f / fs (frequency as fraction of sampling rate)
        scale = f0 / (2 * np.pi * (freq / fs))
        
        # Create delayed task for each frequency
        task = dask.delayed(_cwt_single_freq_fftw)(
            data_fft, scale, freq, fs, freqs_fft, extend_len, N
        )
        task_list.append(task)
    
    # Execute all tasks in parallel
    if verbose:
        print(f"Computing CWT for {len(freqs)} frequencies using {n_workers} workers...")
        print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        print(f"Nyquist limit: {nyquist:.1f} Hz")
        with ProgressBar():
            results = dask.compute(*task_list, num_workers=n_workers)
    else:
        results = dask.compute(*task_list, num_workers=n_workers)
    
    # Stack results into matrix (freqs × time)
    cwt_matrix = np.stack(results, axis=0)
    
    return cwt_matrix


def compute_event_spectral_features(event, lfp_array, fs, nperseg=256, noverlap=None, freq_range=(100, 250), n_bins=20, pad_to=None, smoothing_sigma=1.5, pre_ms=200, post_ms=200, target_freq_bins=None, use_optimized_cwt=True, n_workers=None, verbose=False):
    """
    Compute high-res, smoothed, frequency-masked, and padded spectrogram for a single event.
    Always uses ±pre_ms/post_ms (in ms) window around event peak.
    
    Parameters:
    -----------
    target_freq_bins : int or None
        If specified, limit frequency bins to this number. If None, keep all frequencies in range.
    use_optimized_cwt : bool, optional
        If True and pyfftw is available, use optimized parallel CWT implementation
    n_workers : int, optional
        Number of parallel workers for CWT (default: all CPUs)
    verbose : bool, optional
        Show CWT progress bar
    """
    import numpy as np
    from scipy.signal import spectrogram
    from scipy.ndimage import gaussian_filter1d
    
    # Determine event window
    t0 = event.get('basic_start_time', event.get('start_time'))
    t1 = event.get('basic_end_time', event.get('end_time'))
    if t0 is None or t1 is None:
        raise ValueError("Event missing start/end time.")
    
    # Use peak time if available, else center
    peak_time = event.get('peak_time', 0.5 * (t0 + t1))
    win_start = peak_time - pre_ms/1000.0
    win_end = peak_time + post_ms/1000.0
    idx0 = int(max(0, win_start * fs))
    idx1 = int(min(len(lfp_array), win_end * fs))
    lfp_seg = lfp_array[idx0:idx1]
    
    spec_method = event.get('spec_method', 'stft') if 'spec_method' in event else 'stft'
    
    if spec_method == 'cwt' and use_optimized_cwt and HAS_PYFFTW:
        # --- Optimized PARALLEL CWT with pyfftw + Dask ---
        min_freq, max_freq = freq_range
        
        # CORRECTED: Better frequency selection for ripple analysis
        if target_freq_bins:
            num_freqs = target_freq_bins
        else:
            # Adaptive: ~2 Hz resolution for ripple band (125-250 Hz)
            num_freqs = max(20, int((max_freq - min_freq) / 2.0))
        
        # Use linear spacing for uniform resolution
        freqs = np.linspace(min_freq, max_freq, num_freqs)
        
        # Validate against Nyquist
        nyquist = fs / 2.0
        freqs = freqs[freqs < nyquist]
        
        if len(freqs) == 0:
            raise ValueError(f"No valid frequencies below Nyquist ({nyquist:.1f} Hz)!")
        
        Sxx = _cwt_optimized(
            lfp_seg, fs, freqs, 
            method='full', 
            boundary='mirror',
            n_workers=n_workers,
            verbose=verbose
        )
        f = freqs
        
        # Time axis resampling
        t_norm = np.linspace(0, 1, Sxx.shape[1])
        t_target = np.linspace(0, 1, n_bins)
        Sxx_resampled = np.zeros((Sxx.shape[0], n_bins))
        
        for fi in range(Sxx.shape[0]):
            f_interp = interp1d(t_norm, Sxx[fi], kind='linear', bounds_error=False, fill_value=float(Sxx[fi][0]))
            Sxx_resampled[fi] = f_interp(t_target)
        
        # Apply smoothing
        if smoothing_sigma is not None and smoothing_sigma > 0:
            import scipy.ndimage
            Sxx_resampled = scipy.ndimage.gaussian_filter(Sxx_resampled, sigma=[0, smoothing_sigma])
        
        return Sxx_resampled, f, t_target
        
    elif spec_method == 'cwt':
        # --- Standard CWT (scipy fallback) ---
        from scipy.signal import morlet2
        try:
            from scipy.signal import cwt as scipy_cwt
        except ImportError:
            raise ImportError("scipy.signal.cwt not available. Use STFT or install pyfftw for optimized CWT.")
        
        min_freq, max_freq = freq_range
        num_freqs = target_freq_bins if target_freq_bins else max(20, int((max_freq - min_freq) / 2.0))
        freqs = np.linspace(min_freq, max_freq, num_freqs)
        
        # CORRECTED: Proper scale calculation
        widths = (6.0 * fs) / (2 * np.pi * freqs)  # f0 = 6 for standard Morlet
        
        cwt_matrix = scipy_cwt(lfp_seg, morlet2, widths, w=6.0)
        Sxx = np.abs(cwt_matrix)
        f = freqs
        
        # Resample time axis to n_bins
        t_norm = np.linspace(0, 1, Sxx.shape[1])
        t_target = np.linspace(0, 1, n_bins)
        Sxx_resampled = np.zeros((Sxx.shape[0], n_bins))
        
        for fi in range(Sxx.shape[0]):
            f_interp = interp1d(t_norm, Sxx[fi], kind='linear', bounds_error=False, fill_value=float(Sxx[fi][0]))
            Sxx_resampled[fi] = f_interp(t_target)
        
        if smoothing_sigma is not None and smoothing_sigma > 0:
            import scipy.ndimage
            Sxx_resampled = scipy.ndimage.gaussian_filter(Sxx_resampled, sigma=[0, smoothing_sigma])
        
        return Sxx_resampled, f, t_target
    
    else:
        # --- STFT (default) ---
        if noverlap is None:
            noverlap = nperseg // 2
        
        f, t, Sxx = spectrogram(lfp_seg, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
        
        # Frequency masking
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Optional: limit frequency bins
        if target_freq_bins is not None and len(f) > target_freq_bins:
            f_indices = np.linspace(0, len(f)-1, target_freq_bins, dtype=int)
            f = f[f_indices]
            Sxx = Sxx[f_indices, :]
        
        # Resample time axis
        t_norm = np.linspace(0, 1, Sxx.shape[1])
        t_target = np.linspace(0, 1, n_bins)
        Sxx_resampled = np.zeros((Sxx.shape[0], n_bins))
        
        for fi in range(Sxx.shape[0]):
            f_interp = interp1d(t_norm, Sxx[fi], kind='linear', bounds_error=False, fill_value=float(Sxx[fi][0]))
            Sxx_resampled[fi] = f_interp(t_target)
        
        # Apply smoothing
        if smoothing_sigma is not None and smoothing_sigma > 0:
            import scipy.ndimage
            Sxx_resampled = scipy.ndimage.gaussian_filter(Sxx_resampled, sigma=[0, smoothing_sigma])
        
        return Sxx_resampled, f, t_target


def batch_compute_spectral_features(detector, lfp_array, fs, use='basic', n_bins=20, freq_range=(100, 250), pad_to=None, pre_ms=200, post_ms=200, target_freq_bins=None, smoothing_sigma=1.5, use_optimized_cwt=True, n_workers=None, verbose=False, normalize_psd=False):
    """
    Compute and attach spectrograms to all events in detector.swr_events.
    Supports STFT (default), CWT, and multi-taper spectral estimation.
    
    Parameters:
    -----------
    use_optimized_cwt : bool, optional
        If True and pyfftw is available, use optimized parallel CWT
    n_workers : int, optional
        Number of parallel workers for CWT (default: all CPUs)
    verbose : bool, optional
        Show CWT progress bar
    method: 'stft' (default) or 'multitaper'
    multitaper_kwargs: dict, optional arguments for multitaper
    target_freq_bins : int or None
        If specified, limit frequency bins to this number. If None, keep all frequencies in range.
    smoothing_sigma : float, standard deviation for Gaussian smoothing (in bins)
    normalize_psd : bool, optional
        If True, normalize multitaper PSD to sum to 1 (useful for comparison)
    """
    n_with_spec = 0
    method = getattr(detector, 'spectral_method', 'stft') if hasattr(detector, 'spectral_method') else 'stft'
    multitaper_kwargs = getattr(detector, 'multitaper_kwargs', {}) if hasattr(detector, 'multitaper_kwargs') else {}
    
    for event in detector.swr_events:
        try:
            if method == 'multitaper':
                # Extract event window
                t0 = event.get('basic_start_time', event.get('start_time'))
                t1 = event.get('basic_end_time', event.get('end_time'))
                peak_time = event.get('peak_time', 0.5 * (t0 + t1))
                win_start = peak_time - pre_ms/1000.0
                win_end = peak_time + post_ms/1000.0
                idx0 = int(max(0, win_start * fs))
                idx1 = int(min(len(lfp_array), win_end * fs))
                lfp_seg = lfp_array[idx0:idx1]
                
                # Check if segment is too short
                if len(lfp_seg) < 8:
                    event['spectrogram'] = None
                    event['spectrogram_freqs'] = None
                    event['spectrogram_times'] = None
                    event['spectrogram_error'] = 'Segment too short for multitaper'
                    continue
                
                # Compute single PSD for entire event (not time-resolved)
                S, f, _, _ = mtspectrum(
                    lfp_seg, fs=fs, window_size=len(lfp_seg)/fs,
                    average=True,
                    fpass=multitaper_kwargs.get('fpass', freq_range),
                    bandwidth=multitaper_kwargs.get('bandwidth', 10),
                    ntapers=multitaper_kwargs.get('ntapers', 5),
                    pad=multitaper_kwargs.get('pad', 0)
                )
                
                # Optional: Normalize PSD
                if normalize_psd and np.sum(S) > 0:
                    S = S / np.sum(S)
                
                # Store as 1D PSD (NOT time-resolved spectrogram)
                event['spectrogram'] = S  # Shape: (n_freqs,)
                event['spectrogram_freqs'] = f
                event['spectrogram_times'] = None  # No time axis for single PSD
                event['spectrogram_method'] = 'multitaper'
                
            else:
                # Use compute_event_spectral_features (supports STFT/CWT)
                spec, freqs, t_norm = compute_event_spectral_features(
                    event, lfp_array, fs, n_bins=n_bins, freq_range=freq_range, pad_to=pad_to, 
                    smoothing_sigma=smoothing_sigma,
                    pre_ms=pre_ms, post_ms=post_ms, target_freq_bins=target_freq_bins,
                    use_optimized_cwt=use_optimized_cwt, n_workers=n_workers, verbose=verbose
                )
                event['spectrogram'] = spec
                event['spectrogram_freqs'] = freqs
                event['spectrogram_times'] = t_norm
                event['spectrogram_method'] = event.get('spec_method', 'stft')
            n_with_spec += 1
        except Exception as e:
            event['spectrogram'] = None
            event['spectrogram_freqs'] = None
            event['spectrogram_times'] = None
            event['spectrogram_error'] = str(e)
    return n_with_spec


def average_spectrograms_by_type(detector, event_type, n_bins=100):
    """
    Average resampled spectrograms for all events of a given type.
    Returns (mean_spec, freqs, norm_time)
    """
    specs = []
    freqs_list = []
    for e in detector.swr_events:
        if e.get('event_type') == event_type and 'spectrogram' in e:
            spec = e['spectrogram']
            freqs = e['spectrogram_freqs']
            specs.append(spec)
            freqs_list.append(freqs)
    if not specs:
        return None, None, None

    # Find max frequency dimension
    max_freq_dim = max(s.shape[0] for s in specs)
    # Pad all spectrograms to max_freq_dim
    padded_specs = []
    for spec in specs:
        pad_width = max_freq_dim - spec.shape[0]
        if pad_width > 0:
            spec_padded = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
        else:
            spec_padded = spec
        # Resample time axis
        current_time_bins = spec_padded.shape[1]
        t_current = np.linspace(0, 1, current_time_bins)
        t_target = np.linspace(0, 1, n_bins)
        spec_resampled = np.zeros((max_freq_dim, n_bins))
        for fi in range(max_freq_dim):
            f_interp = interp1d(t_current, spec_padded[fi], kind='linear', bounds_error=False, fill_value=float(spec_padded[fi][0]))
            spec_resampled[fi] = f_interp(t_target)
        padded_specs.append(spec_resampled)

    # Average
    mean_spec = np.mean(np.stack(padded_specs, axis=0), axis=0)
    # Use the most common frequency vector, padded if needed
    from collections import Counter
    freq_tuples = [tuple(np.round(f, 2)) for f in freqs_list]
    most_common_freq = Counter(freq_tuples).most_common(1)[0][0]
    freqs = np.array(most_common_freq)
    if len(freqs) < max_freq_dim:
        freqs = np.pad(freqs, (0, max_freq_dim - len(freqs)), mode='edge')
    norm_time = np.linspace(0, 1, n_bins)
    return mean_spec, freqs, norm_time

# --- Widget for browsing single-event spectrograms with event classification in the legend ---
def event_spectrogram_browser(detector, event_type=None):
    # Filter events with spectrograms
    if event_type is not None:
        events = [e for e in detector.swr_events if e.get('event_type') == event_type and 'spectrogram' in e]
    else:
        events = [e for e in detector.swr_events if 'spectrogram' in e]
    
    # Additional validation: ensure spectrograms are 2D with valid shape
    valid_events = []
    for e in events:
        spec = e.get('spectrogram')
        if spec is not None and isinstance(spec, np.ndarray) and spec.ndim == 2 and spec.shape[0] > 0 and spec.shape[1] > 0:
            valid_events.append(e)
    
    events = valid_events
    
    if not events:
        print("No events with valid 2D spectrograms found.")
        return

    out = Output()
    idx_slider = IntSlider(value=0, min=0, max=len(events)-1, description='Event #')

    def plot_event(idx):
        with out:
            clear_output(wait=True)
            event = events[idx]
            spec = event['spectrogram']
            freqs = event['spectrogram_freqs']
            t_norm = event['spectrogram_times']
            evt_type = event.get('event_type', 'unknown')
            eid = event.get('event_id', idx)
            
            # Additional shape validation before plotting
            if spec.shape[0] == 0 or spec.shape[1] == 0:
                print(f"Event {eid} has invalid spectrogram shape: {spec.shape}")
                return
            
            plt.figure(figsize=(7, 4))
            plt.pcolormesh(t_norm, freqs, spec, shading='auto')
            plt.xlabel('Normalized Time (0=start, 1=end)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Event {eid} | Type: {evt_type}')
            plt.colorbar(label='Power')
            plt.tight_layout()
            plt.show()

    def on_change(change):
        plot_event(idx_slider.value)

    idx_slider.observe(on_change, names='value')
    display(VBox([idx_slider, out]))
    plot_event(0)

# Usage example (after batch_compute_spectral_features):
# event_spectrogram_browser(detector, event_type=None)  # or specify event_type='ripple_only'

def compute_event_spectral_features_from_trace(event, fs, n_bins=100):
    """
    Compute spectrogram and features from a per-event LFP trace.
    Assumes event['raw_trace'] or event['ripple_trace'] is present.
    Adds results to event dict.
    """
    lfp_seg = event.get('raw_trace') or event.get('ripple_trace')
    if lfp_seg is None or len(lfp_seg) < 16:
        return
    from scipy.signal import spectrogram
    from scipy.interpolate import interp1d
    import numpy as np

    f, t, Sxx = spectrogram(lfp_seg, fs=fs, nperseg=64, noverlap=32)
    Sxx = np.abs(Sxx)
    t_norm = np.linspace(0, 1, Sxx.shape[1])
    t_target = np.linspace(0, 1, n_bins)
    Sxx_resampled = np.zeros((Sxx.shape[0], n_bins))
    for fi in range(Sxx.shape[0]):
        f_interp = interp1d(t_norm, Sxx[fi], kind='linear', bounds_error=False, fill_value=float(Sxx[fi][0]))
        Sxx_resampled[fi] = f_interp(t_target)
    psd = np.mean(Sxx, axis=1)
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.ones_like(psd)/len(psd)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    from scipy.stats import linregress
    if np.any(psd[1:] > 0):
        log_f = np.log10(f[1:])
        log_psd = np.log10(psd[1:])
        slope, _, _, _, _ = linregress(log_f, log_psd)
    else:
        slope = np.nan
    event['spectrogram'] = Sxx_resampled
    event['spectrogram_freqs'] = f
    event['spectrogram_times'] = t_target
    event['spectral_entropy'] = float(spectral_entropy)
    event['spectral_slope'] = float(slope)

# --- Widget for visualizing single event LFP and spectrogram, aligned at peak, with ±200ms padding ---
def event_lfp_and_spectrogram_widget(detector, lfp_array, fs, pre_ms=200, post_ms=200, event_type=None, freq_range=(100, 250), ripple_trace=None, ripple_alpha=0.7, ripple_scale=0.5, normalize_psd=False):
    """
    Interactive widget to visualize raw LFP and spectrogram for single events, aligned at peak amplitude,
    with ±pre_ms and post_ms (in ms) padding around the event.
    
    Parameters:
    -----------
    freq_range : tuple, optional
        Default frequency range (min_freq, max_freq) in Hz for the slider. Default is (100, 250).
    normalize_psd : bool, optional
        If True, normalize multitaper PSD to sum to 1
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from ipywidgets import Button, HBox, VBox, IntText, Output, FloatRangeSlider, Checkbox
    from IPython.display import display, clear_output

    # Filter events with valid spectrograms
    def has_valid_spec(e):
        spec = e.get('spectrogram')
        freqs = e.get('spectrogram_freqs')
        # Accept both 2D (stft/cwt) and 1D (multitaper) spectrograms
        if spec is None or freqs is None:
            return False
        if isinstance(spec, np.ndarray):
            if spec.ndim == 2 and spec.shape[0] > 0 and spec.shape[1] > 0:
                return True
            if spec.ndim == 1 and spec.shape[0] > 0:  # multitaper: 1D PSD
                return True
        return False

    if event_type is not None:
        events = [e for e in detector.swr_events if e.get('event_type') == event_type and has_valid_spec(e)]
    else:
        events = [e for e in detector.swr_events if has_valid_spec(e)]
    if not events:
        print("No events with valid spectrograms found.")
        return

    # Get global min/max frequency for slider
    all_freqs = np.concatenate([np.atleast_1d(e['spectrogram_freqs']) for e in events if e['spectrogram_freqs'] is not None])
    min_freq = float(np.min(all_freqs))
    max_freq = float(np.max(all_freqs))
    default_freq_range = (max(min_freq, freq_range[0]), min(max_freq, freq_range[1]))

    out = Output()
    event_idx = IntText(value=0, description='Event #:', min=0, max=len(events)-1)
    prev_btn = Button(description='Previous')
    next_btn = Button(description='Next')
    freq_slider = FloatRangeSlider(
        value=default_freq_range, min=min_freq, max=max_freq, step=1,
        description='Freq Range (Hz):', continuous_update=False, layout={'width': '400px'}
    )
    normalize_checkbox = Checkbox(value=normalize_psd, description='Normalize PSD')

    def plot_event():
        with out:
            clear_output(wait=True)
            idx = event_idx.value
            freq_range_slider = freq_slider.value
            do_normalize = normalize_checkbox.value

            if idx < 0 or idx >= len(events):
                print(f"Invalid event index: {idx}")
                return

            event = events[idx]
            peak_time = event.get('peak_time')
            if peak_time is None:
                t0 = event.get('basic_start_time', event.get('start_time'))
                t1 = event.get('basic_end_time', event.get('end_time'))
                if t0 is not None and t1 is not None:
                    peak_time = 0.5 * (t0 + t1)
                else:
                    print(f"Event {idx} missing timing info.")
                    return
            
            # Extract LFP segment
            win_start = peak_time - pre_ms/1000.0
            win_end = peak_time + post_ms/1000.0
            idx0 = int(max(0, win_start * fs))
            idx1 = int(min(len(lfp_array), win_end * fs))
            lfp_seg = lfp_array[idx0:idx1]
            # Create time axis centered at peak (t=0)
            t_lfp = np.arange(len(lfp_seg)) / fs - pre_ms/1000.0
            
            # Get spectrogram data
            spec = event.get('spectrogram')
            freqs = np.atleast_1d(event.get('spectrogram_freqs'))
            t_norm = event.get('spectrogram_times')
            method = event.get('spectrogram_method', detector.spectral_method if hasattr(detector, 'spectral_method') else 'stft')

            if spec is None or freqs is None:
                print(f"Event {idx} missing spectrogram data.")
                return

            # ===== FIX: Handle multitaper 1D PSD properly =====
            if method == 'multitaper' and spec.ndim == 1:
                # Multitaper: plot PSD vs. frequency
                fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False, gridspec_kw={'height_ratios': [1, 1.5]})
                ax1, ax2 = axes
                
                # Top: Raw LFP
                ax1.plot(t_lfp, lfp_seg, color='k', lw=1)
                ax1.axvline(0, color='r', linestyle='--', lw=1, label='Peak')
                ax1.set_ylabel('LFP (µV)', fontweight='bold')
                ax1.set_title(f'Event {idx} ({event.get("event_type", "Unclassified")}): Raw LFP', fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Bottom: PSD
                # Ensure matching lengths
                n = min(len(freqs), len(spec))
                freqs_plot = freqs[:n]
                spec_plot = spec[:n]
                
                # Optional normalization
                if do_normalize and np.sum(spec_plot) > 0:
                    spec_plot = spec_plot / np.sum(spec_plot)
                    ylabel = 'Normalized Power (a.u.)'
                else:
                    ylabel = 'Power Spectral Density'
                
                # Frequency masking
                freq_mask = (freqs_plot >= freq_range_slider[0]) & (freqs_plot <= freq_range_slider[1])
                
                ax2.plot(freqs_plot[freq_mask], spec_plot[freq_mask], color='b', lw=2, label='Multitaper PSD')
                
                # Find and mark peak frequency
                peak_freq_idx = np.argmax(spec_plot[freq_mask])
                peak_freq = freqs_plot[freq_mask][peak_freq_idx]
                peak_power = spec_plot[freq_mask][peak_freq_idx]
                ax2.axvline(peak_freq, color='r', linestyle='--', lw=1.5, alpha=0.7, label=f'Peak: {peak_freq:.1f} Hz')
                ax2.plot(peak_freq, peak_power, 'ro', markersize=8)
                
                ax2.set_ylabel(ylabel, fontweight='bold')
                ax2.set_xlabel('Frequency (Hz)', fontweight='bold')
                ax2.set_title('Multitaper Power Spectrum', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
            else:
                # STFT or CWT: 2D spectrogram
                freq_mask = (freqs >= freq_range_slider[0]) & (freqs <= freq_range_slider[1])
                spec_disp = spec[freq_mask, :]
                freqs_disp = freqs[freq_mask]
                t_spec = t_norm * (win_end - win_start) - pre_ms/1000.0
                
                fig, ax = plt.subplots(figsize=(12, 6))
                pcm = ax.pcolormesh(t_spec, freqs_disp, spec_disp, shading='auto', cmap='magma')
                ax.axvline(0, color='r', linestyle='--', lw=1.5, label='Peak', alpha=0.8)
                ax.set_ylabel('Frequency (Hz)', fontweight='bold')
                ax.set_xlabel('Time (s) relative to peak', fontweight='bold')
                fig.colorbar(pcm, ax=ax, label='Power')

                # Mark peak amplitude in time and frequency on the spectrogram
                if spec_disp.size > 0:
                    peak_idx_flat = np.argmax(spec_disp)
                    peak_freq_idx, peak_time_idx = np.unravel_index(peak_idx_flat, spec_disp.shape)
                    peak_freq_val = freqs_disp[peak_freq_idx]
                    peak_time_val = t_spec[peak_time_idx]
                    ax.plot(peak_time_val, peak_freq_val, 'rx', markersize=14, markeredgewidth=2, label='Peak Power')

                # Normalize LFP for overlay
                lfp_seg_norm = (lfp_seg - np.min(lfp_seg)) / (np.max(lfp_seg) - np.min(lfp_seg)) if np.max(lfp_seg) != np.min(lfp_seg) else np.zeros_like(lfp_seg)
                lfp_seg_scaled = lfp_seg_norm * (freqs_disp[-1] - freqs_disp[0]) + freqs_disp[0]
                ax.plot(t_lfp, lfp_seg_scaled, color='white', lw=1.5, label='Normalized LFP', alpha=0.9)

                # Overlay ripple-filtered trace if provided
                if ripple_trace is not None:
                    ripple_seg = ripple_trace[idx0:idx1]
                    ripple_norm = (ripple_seg - np.min(ripple_seg)) / (np.max(ripple_seg) - np.min(ripple_seg)) if np.max(ripple_seg) != np.min(ripple_seg) else np.zeros_like(ripple_seg)
                    ripple_scaled = ripple_norm * (freqs_disp[-1] - freqs_disp[0]) * ripple_scale + freqs_disp[0]
                    ax.plot(t_lfp, ripple_scaled, color='orange', lw=1.5, alpha=ripple_alpha, label='Ripple-filtered')

                ax.legend(loc='upper right', fontsize=10)
                event_class = event.get('event_type', 'Unclassified')
                ax.set_title(f'Event {idx} ({event_class}): Spectrogram with LFP Overlay', fontweight='bold')
                plt.tight_layout()
                plt.show()

    def on_prev(b):
        if event_idx.value > 0:
            event_idx.value -= 1
            plot_event()

    def on_next(b):
        if event_idx.value < len(events) - 1:
            event_idx.value += 1
            plot_event()

    def on_idx_change(change):
        plot_event()

    def on_freq_change(change):
        plot_event()
    
    def on_normalize_change(change):
        plot_event()

    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    event_idx.observe(on_idx_change, names='value')
    freq_slider.observe(on_freq_change, names='value')
    normalize_checkbox.observe(on_normalize_change, names='value')

    nav_controls = HBox([prev_btn, next_btn, event_idx])
    controls = VBox([nav_controls, freq_slider, normalize_checkbox, out])
    display(controls)
    plot_event()
