def _cwt_optimized_gpu_batched(lfp_batch, fs, freqs, boundary='mirror', max_freqs_per_batch=50, verbose=False):
    """
    Batched GPU-accelerated CWT for multiple events.
    lfp_batch: shape (n_events, n_samples)
    Returns: np.ndarray, shape (n_events, n_freqs, n_samples)
    """
    import cupy as cp
    from cupyx.scipy.fft import fft, ifft

    n_events, N = lfp_batch.shape
    n_freqs = len(freqs)
    # Boundary extension (same for all events)
    extend_len = min(int(np.ceil(N * 0.1)), N)
    if boundary == 'mirror':
        data = np.concatenate([np.flip(lfp_batch[:, :extend_len], axis=1), lfp_batch, np.flip(lfp_batch[:, -extend_len:], axis=1)], axis=1)
    elif boundary == 'zeros':
        data = np.concatenate([np.zeros((n_events, extend_len)), lfp_batch, np.zeros((n_events, extend_len))], axis=1)
    else:
        data = lfp_batch

    data_gpu = cp.asarray(data, dtype='complex128')
    data_fft_gpu = fft(data_gpu, axis=1)
    freqs_fft = np.fft.fftfreq(data.shape[1], d=1.0/fs)
    freqs_fft_gpu = cp.asarray(freqs_fft)

    results = cp.zeros((n_events, n_freqs, N), dtype='float64')
    f0 = 6.0

    for i, freq in enumerate(freqs):
        scale = f0 / (2 * np.pi * (freq / fs))
        psi_fft = cp.zeros_like(freqs_fft_gpu, dtype='complex128')
        positive_mask = freqs_fft_gpu > 0
        omega_normalized = 2 * cp.pi * freqs_fft_gpu[positive_mask] / fs
        psi_fft[positive_mask] = (cp.pi ** -0.25) * cp.sqrt(scale) * cp.exp(-0.5 * (scale * omega_normalized - f0) ** 2)
        conv_fft = data_fft_gpu * psi_fft
        result = ifft(conv_fft, axis=1)
        results[:, i, :] = cp.abs(result[:, extend_len:extend_len+N])
    return cp.asnumpy(results)
"""
GPU-Accelerated CWT Spectral Analysis using CuPy

Provides CUDA-accelerated implementations of continuous wavelet transform
for rapid spectrogram computation. Maintains mathematical equivalence to CPU version.

Requirements:
- cupy-cuda11x or cupy-cuda12x (matching system CUDA version)
"""

import numpy as np
import warnings


def _cwt_single_freq_gpu(data_fft_gpu, scale, freq, fs, freqs_fft_gpu, extend_len, N):
    """
    Compute CWT for a single frequency using CuPy (GPU-accelerated).
    
    CORRECTED: Proper Morlet wavelet normalization and scale-frequency relationship.
    
    Parameters:
    -----------
    data_fft_gpu : cupy.ndarray
        FFT of the data (on GPU)
    scale : float
        Wavelet scale (dimensionless)
    freq : float
        Target frequency in Hz
    fs : float
        Sampling rate in Hz
    freqs_fft_gpu : cupy.ndarray
        Frequency vector in Hz (from np.fft.fftfreq, on GPU)
    extend_len : int
        Boundary extension length
    N : int
        Original data length
        
    Returns:
    --------
    cwt_coef : cupy.ndarray, shape (N,)
        CWT coefficients for this frequency (absolute value)
    """
    import cupy as cp
    from cupyx.scipy.fft import ifft
    
    # Morlet wavelet in frequency domain
    # Central frequency f0 = 6 (standard for neuroscience applications)
    f0 = 6.0
    
    # Morlet wavelet: ψ(ω) = π^(-1/4) * exp(-0.5 * (scale*ω - f0)^2)
    # For digital signals: ω_normalized = 2π * f / fs
    # We want scale * ω_normalized = f0 when f = target_freq
    # Therefore: scale = f0 * fs / (2π * target_freq)
    
    # Compute wavelet in frequency domain
    psi_fft = cp.zeros_like(freqs_fft_gpu, dtype='complex128')
    
    # Only compute for positive frequencies
    positive_mask = freqs_fft_gpu > 0
    
    # Normalized frequency: ω = 2π * f_Hz / fs (in radians per sample)
    omega_normalized = 2 * cp.pi * freqs_fft_gpu[positive_mask] / fs
    
    # Wavelet function in frequency domain with proper normalization
    # π^(-1/4) for amplitude normalization
    # sqrt(scale) for energy normalization across scales
    psi_fft[positive_mask] = (cp.pi ** -0.25) * cp.sqrt(scale) * \
                              cp.exp(-0.5 * (scale * omega_normalized - f0) ** 2)
    
    # Convolution in frequency domain (element-wise multiplication)
    conv_fft = data_fft_gpu * psi_fft
    
    # Inverse FFT using CuPy
    result = ifft(conv_fft)
    
    # Extract valid region (remove boundary extensions)
    # Return absolute value (magnitude) for power analysis
    return cp.abs(result[extend_len:extend_len + N])


def _cwt_optimized_gpu(lfp_seg, fs, freqs, boundary='mirror', max_freqs_per_batch=50, verbose=False):
    """
    GPU-accelerated CWT implementation using CuPy with automatic batching.
    
    CORRECTED: Proper frequency-scale relationship and validation.
    Maintains mathematical equivalence to CPU version (_cwt_optimized).
    
    Parameters:
    -----------
    lfp_seg : np.ndarray
        LFP segment to analyze (on CPU)
    fs : float
        Sampling frequency
    freqs : np.ndarray
        Frequencies to analyze (in Hz, on CPU)
    boundary : str, optional
        Boundary handling: 'mirror', 'zeros', or 'periodic'
    max_freqs_per_batch : int, optional
        Max frequencies to process simultaneously (tune based on GPU VRAM)
    verbose : bool, optional
        Show progress
    
    Returns:
    --------
    cwt_matrix : np.ndarray
        CWT power (freqs × time), returned on CPU
    """
    try:
        import cupy as cp
        from cupyx.scipy.fft import fft
    except ImportError:
        raise ImportError(
            "CuPy not installed. Install with: pip install cupy-cuda11x (or cupy-cuda12x)"
        )
    
    N = len(lfp_seg)
    
    # Validate frequency range
    nyquist = fs / 2.0
    if np.any(freqs > nyquist):
        warnings.warn(
            f"Frequencies above Nyquist ({nyquist:.1f} Hz) detected. Clipping to valid range."
        )
        freqs = freqs[freqs <= nyquist]
    
    if np.any(freqs <= 0):
        warnings.warn("Non-positive frequencies detected. Removing invalid frequencies.")
        freqs = freqs[freqs > 0]
    
    if len(freqs) == 0:
        raise ValueError("No valid frequencies remaining after validation")
    
    # Boundary extension (10% or full length, whichever is smaller)
    extend_len = min(int(np.ceil(N * 0.1)), N)
    if boundary == 'mirror':
        data = np.hstack((np.flip(lfp_seg[:extend_len]), lfp_seg, np.flip(lfp_seg[-extend_len:])))
    elif boundary == 'zeros':
        data = np.hstack((np.zeros(extend_len), lfp_seg, np.zeros(extend_len)))
    elif boundary == 'periodic':
        data = lfp_seg  # Periodic boundary = no extension needed
    else:
        raise ValueError(f"Invalid boundary mode: {boundary}")
    
    # Transfer data to GPU ONCE (shared across all frequencies)
    data_gpu = cp.asarray(data, dtype='complex128')
    data_fft_gpu = fft(data_gpu)
    
    # Frequency vector in Hz (for wavelet construction)
    freqs_fft = np.fft.fftfreq(data.shape[0], d=1.0/fs)
    freqs_fft_gpu = cp.asarray(freqs_fft)
    
    # Process frequencies in batches to prevent OOM
    cwt_results = []
    f0 = 6.0  # Morlet central frequency (standard)
    
    for i in range(0, len(freqs), max_freqs_per_batch):
        batch_freqs = freqs[i:i + max_freqs_per_batch]
        batch_results = []
        
        for freq in batch_freqs:
            # Compute scale: scale = f0 / (2π * freq_normalized)
            # where freq_normalized = freq / fs
            scale = f0 / (2 * np.pi * (freq / fs))
            
            result = _cwt_single_freq_gpu(
                data_fft_gpu, scale, freq, fs, freqs_fft_gpu, extend_len, N
            )
            batch_results.append(result)
        
        # Transfer batch back to CPU to free GPU memory
        cwt_results.extend([cp.asnumpy(r) for r in batch_results])
        
        # Clear GPU cache
        cp.get_default_memory_pool().free_all_blocks()
        
        if verbose and i > 0:
            print(f"GPU: Processed {min(i+max_freqs_per_batch, len(freqs))}/{len(freqs)} frequencies")
    
    # Stack results into matrix (freqs × time)
    cwt_matrix = np.stack(cwt_results, axis=0)
    
    return cwt_matrix


def cwt_batch_gpu(lfp_segments, fs, freqs, boundary='mirror', max_freqs_per_batch=50, 
                  event_batch_size=100, verbose=False):
    """
    Process multiple LFP segments in batches using GPU.
    
    Parameters:
    -----------
    lfp_segments : list of np.ndarray
        List of LFP segments to process
    fs : float
        Sampling frequency
    freqs : np.ndarray
        Frequencies to analyze
    boundary : str
        Boundary handling method
    max_freqs_per_batch : int
        Frequencies per GPU batch
    event_batch_size : int
        Number of events to process before clearing GPU cache
    verbose : bool
        Show progress
    
    Returns:
    --------
    results : list of np.ndarray
        List of CWT matrices (one per segment)
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy not installed. Cannot use GPU batching.")
    
    results = []
    n_events = len(lfp_segments)
    
    for i, segment in enumerate(lfp_segments):
        cwt_matrix = _cwt_optimized_gpu(
            segment, fs, freqs,
            boundary=boundary,
            max_freqs_per_batch=max_freqs_per_batch,
            verbose=False
        )
        results.append(cwt_matrix)
        
        # Clear GPU cache periodically
        if (i + 1) % event_batch_size == 0:
            cp.get_default_memory_pool().free_all_blocks()
            if verbose:
                print(f"GPU: Processed {i+1}/{n_events} events (cache cleared)")
    
    if verbose:
        print(f"GPU: Completed {n_events} events")
    
    return results


if __name__ == "__main__":
    # Test GPU functionality
    print("Testing GPU CWT implementation...")
    
    from .gpu_utils import check_gpu_availability, print_gpu_info
    
    print_gpu_info()
    
    has_cupy, has_cuda, _, _ = check_gpu_availability()
    
    if not (has_cupy and has_cuda):
        print("\n❌ GPU not available. Cannot run test.")
    else:
        print("\n✅ Running GPU CWT test...")
        
        # Create synthetic 150 Hz signal
        fs = 1000.0
        duration = 0.4
        t = np.arange(0, duration, 1/fs)
        signal = np.sin(2 * np.pi * 150 * t) + 0.3 * np.random.randn(len(t))
        
        freqs = np.linspace(100, 250, 60)
        
        # Run GPU CWT
        import time
        start = time.time()
        cwt_result = _cwt_optimized_gpu(signal, fs, freqs, verbose=True)
        elapsed = time.time() - start
        
        # Find peak
        power_avg = np.mean(cwt_result, axis=1)
        peak_idx = np.argmax(power_avg)
        detected_freq = freqs[peak_idx]
        
        print(f"\nGPU CWT completed in {elapsed:.3f}s")
        print(f"Expected frequency: 150 Hz")
        print(f"Detected frequency: {detected_freq:.1f} Hz")
        print(f"Error: {abs(detected_freq - 150):.2f} Hz")
        
        if abs(detected_freq - 150) < 5:
            print("✅ GPU CWT test PASSED")
        else:
            print("⚠️ GPU CWT test shows unexpected error")
