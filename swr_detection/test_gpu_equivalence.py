"""
GPU/CPU Equivalence Test for CWT Spectral Analysis

Validates that GPU implementation produces mathematically identical results
to the CPU version within numerical precision.

Phase 6: Equivalence Testing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_synthetic_lfp(duration=0.4, fs=1000.0, ripple_freq=150, noise_level=0.3, seed=42):
    """
    Generate synthetic LFP with known ripple frequency.
    
    Parameters:
    -----------
    duration : float
        Signal duration in seconds
    fs : float
        Sampling frequency
    ripple_freq : float
        Ripple frequency in Hz
    noise_level : float
        Noise amplitude relative to signal
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    signal : np.ndarray
        Synthetic LFP signal
    t : np.ndarray
        Time vector
    """
    np.random.seed(seed)
    t = np.arange(0, duration, 1/fs)
    
    # Pure ripple component
    ripple = np.sin(2 * np.pi * ripple_freq * t)
    
    # Add multiple frequency components for complexity
    ripple += 0.3 * np.sin(2 * np.pi * (ripple_freq * 1.5) * t)  # Harmonic
    ripple += 0.2 * np.sin(2 * np.pi * 120 * t)  # Lower frequency
    
    # Add noise
    noise = noise_level * np.random.randn(len(t))
    signal = ripple + noise
    
    return signal, t


def test_cwt_equivalence(signal, fs, freqs, tolerance=0.01, verbose=True):
    """
    Test GPU vs CPU CWT equivalence.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input LFP signal
    fs : float
        Sampling frequency
    freqs : np.ndarray
        Frequencies to analyze
    tolerance : float
        Acceptable relative error (default 1%)
    verbose : bool
        Print detailed results
    
    Returns:
    --------
    passed : bool
        True if test passes
    correlation : float
        Pearson correlation between GPU and CPU results
    max_error : float
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # === DEBUG: Print CuPy/CUDA status at script startup ===
    try:
        import cupy
        print("DEBUG: CuPy version:", cupy.__version__)
        try:
            print("DEBUG: CUDA runtime version:", cupy.cuda.runtime.runtimeGetVersion())
            print("DEBUG: CUDA driver version:", cupy.cuda.runtime.driverGetVersion())
            print("DEBUG: GPU name:", cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())
        except Exception as e:
            print("DEBUG: CUDA ERROR:", e)
    except Exception as e:
        print("DEBUG: CuPy import failed:", e)
        Maximum relative error
    """
    # Import both implementations
    try:
        from swr_detection.swr_spectral_features import _cwt_optimized
        from swr_detection.swr_spectral_features_gpu import _cwt_optimized_gpu, _cwt_optimized_gpu_batched
        from swr_detection.gpu_utils import check_gpu_availability
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False, 0.0, np.inf

    # Check GPU availability
    has_cupy, has_cuda, _, _ = check_gpu_availability()
    if not (has_cupy and has_cuda):
        print("❌ GPU not available. Cannot run equivalence test.")
        return False, 0.0, np.inf

    # Run CPU version
    if verbose:
        print("Running CPU CWT...")
    try:
        cwt_cpu = _cwt_optimized(
            signal, fs, freqs,
            boundary='mirror',
            n_workers=4,
            verbose=False
        )
    except Exception as e:
        print(f"❌ CPU CWT failed: {e}")
        return False, 0.0, np.inf

    # Run GPU version (single)
    if verbose:
        print("Running GPU CWT (single event)...")
    try:
        cwt_gpu = _cwt_optimized_gpu(
            signal, fs, freqs,
            boundary='mirror',
            max_freqs_per_batch=50,
            verbose=False
        )
    except Exception as e:
        print(f"❌ GPU CWT failed: {e}")
        return False, 0.0, np.inf

    # Run GPU version (batched)
    if verbose:
        print("Running GPU CWT (batched)...")
    try:
        import numpy as np
        signal_batch = np.expand_dims(signal, axis=0)  # shape (1, n_samples)
        cwt_gpu_batched = _cwt_optimized_gpu_batched(
            signal_batch, fs, freqs,
            boundary='mirror',
            max_freqs_per_batch=50,
            verbose=False
        )[0]
    except Exception as e:
        print(f"❌ GPU CWT (batched) failed: {e}")
        cwt_gpu_batched = None
    
    # Validate shapes
    if cwt_cpu.shape != cwt_gpu.shape:
        print(f"❌ Shape mismatch: CPU {cwt_cpu.shape} vs GPU {cwt_gpu.shape}")
        return False, 0.0, np.inf

    # Compare single-event GPU
    cpu_flat = cwt_cpu.flatten()
    gpu_flat = cwt_gpu.flatten()
    correlation, _ = pearsonr(cpu_flat, gpu_flat)
    relative_error = np.abs(cwt_gpu - cwt_cpu) / (np.abs(cwt_cpu) + 1e-10)
    max_error = np.max(relative_error)
    mean_error = np.mean(relative_error)
    cpu_power = np.mean(cwt_cpu, axis=1)
    gpu_power = np.mean(cwt_gpu, axis=1)
    cpu_peak_idx = np.argmax(cpu_power)
    gpu_peak_idx = np.argmax(gpu_power)
    cpu_peak_freq = freqs[cpu_peak_idx]
    gpu_peak_freq = freqs[gpu_peak_idx]
    freq_error = abs(cpu_peak_freq - gpu_peak_freq)
    passed = (correlation > 0.99) and (max_error < tolerance)

    # Compare batched GPU if available
    if cwt_gpu_batched is not None:
        gpu_batched_flat = cwt_gpu_batched.flatten()
        corr_batched, _ = pearsonr(cpu_flat, gpu_batched_flat)
        rel_err_batched = np.abs(cwt_gpu_batched - cwt_cpu) / (np.abs(cwt_cpu) + 1e-10)
        max_err_batched = np.max(rel_err_batched)
        mean_err_batched = np.mean(rel_err_batched)
        gpu_batched_power = np.mean(cwt_gpu_batched, axis=1)
        gpu_batched_peak_idx = np.argmax(gpu_batched_power)
        gpu_batched_peak_freq = freqs[gpu_batched_peak_idx]
        freq_err_batched = abs(cpu_peak_freq - gpu_batched_peak_freq)
        passed_batched = (corr_batched > 0.99) and (max_err_batched < tolerance)
    else:
        corr_batched = max_err_batched = mean_err_batched = freq_err_batched = None
        passed_batched = False

    if verbose:
        print("\n" + "=" * 70)
        print("GPU/CPU EQUIVALENCE TEST RESULTS")
        print("=" * 70)
        print(f"Signal shape: {signal.shape}")
        print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz ({len(freqs)} bins)")
        print(f"CWT output shape: {cwt_cpu.shape}")
        print(f"\n📊 Numerical Comparison (Single Event GPU):")
        print(f"   Correlation: {correlation:.6f}")
        print(f"   Max relative error: {max_error:.6f} ({max_error*100:.4f}%)")
        print(f"   Mean relative error: {mean_error:.6f} ({mean_error*100:.4f}%)")
        print(f"\n🎯 Peak Frequency Detection:")
        print(f"   CPU peak: {cpu_peak_freq:.2f} Hz")
        print(f"   GPU peak: {gpu_peak_freq:.2f} Hz")
        print(f"   Difference: {freq_error:.2f} Hz")
        print(f"\n✅ Test {'PASSED' if passed else '❌ FAILED'}")
        if not passed:
            if correlation <= 0.99:
                print(f"   ⚠️ Correlation too low ({correlation:.6f} < 0.99)")
            if max_error >= tolerance:
                print(f"   ⚠️ Max error too high ({max_error:.6f} >= {tolerance:.6f})")
        print("-" * 70)
        if cwt_gpu_batched is not None:
            print(f"\n📊 Numerical Comparison (Batched GPU):")
            print(f"   Correlation: {corr_batched:.6f}")
            print(f"   Max relative error: {max_err_batched:.6f} ({max_err_batched*100:.4f}%)")
            print(f"   Mean relative error: {mean_err_batched:.6f} ({mean_err_batched*100:.4f}%)")
            print(f"\n🎯 Peak Frequency Detection:")
            print(f"   CPU peak: {cpu_peak_freq:.2f} Hz")
            print(f"   Batched GPU peak: {gpu_batched_peak_freq:.2f} Hz")
            print(f"   Difference: {freq_err_batched:.2f} Hz")
            print(f"\n✅ Batched Test {'PASSED' if passed_batched else '❌ FAILED'}")
            if not passed_batched:
                if corr_batched <= 0.99:
                    print(f"   ⚠️ Correlation too low ({corr_batched:.6f} < 0.99)")
                if max_err_batched >= tolerance:
                    print(f"   ⚠️ Max error too high ({max_err_batched:.6f} >= {tolerance:.6f})")
        print("=" * 70)

    return passed and passed_batched, correlation, max_error


def plot_comparison(signal, fs, freqs, cwt_cpu, cwt_gpu):
    """
    Visualize CPU vs GPU results side-by-side.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    t = np.arange(len(signal)) / fs * 1000  # Convert to ms
    
    # CPU spectrogram
    im1 = axes[0, 0].pcolormesh(t, freqs, cwt_cpu, shading='auto', cmap='viridis')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_title('CPU CWT', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], label='Power')
    
    # GPU spectrogram
    im2 = axes[0, 1].pcolormesh(t, freqs, cwt_gpu, shading='auto', cmap='viridis')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_title('GPU CWT', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], label='Power')
    
    # Difference map
    difference = np.abs(cwt_gpu - cwt_cpu)
    im3 = axes[1, 0].pcolormesh(t, freqs, difference, shading='auto', cmap='Reds')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_title('Absolute Difference', fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], label='|GPU - CPU|')
    
    # Power spectrum comparison
    cpu_power = np.mean(cwt_cpu, axis=1)
    gpu_power = np.mean(cwt_gpu, axis=1)
    axes[1, 1].plot(freqs, cpu_power, 'b-', lw=2, label='CPU', alpha=0.7)
    axes[1, 1].plot(freqs, gpu_power, 'r--', lw=2, label='GPU', alpha=0.7)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Average Power')
    axes[1, 1].set_title('Power Spectrum Comparison', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_comprehensive_tests():
    """
    Run multiple test scenarios.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE GPU EQUIVALENCE TESTING")
    print("=" * 70)
    
    # Check GPU availability first
    from swr_detection.gpu_utils import print_gpu_info
    print_gpu_info()
    
    test_cases = [
        {
            'name': 'Standard ripple (150 Hz, 400ms)',
            'duration': 0.4,
            'fs': 1000.0,
            'ripple_freq': 150,
            'freq_range': (100, 250),
            'n_freqs': 60
        },
        {
            'name': 'Short event (50ms)',
            'duration': 0.05,
            'fs': 1000.0,
            'ripple_freq': 175,
            'freq_range': (100, 250),
            'n_freqs': 40
        },
        {
            'name': 'High frequency resolution (150 bins)',
            'duration': 0.4,
            'fs': 1000.0,
            'ripple_freq': 180,
            'freq_range': (100, 250),
            'n_freqs': 150
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_cases)}: {case['name']}")
        print(f"{'='*70}")
        
        # Generate signal
        signal, _ = generate_synthetic_lfp(
            duration=case['duration'],
            fs=case['fs'],
            ripple_freq=case['ripple_freq']
        )
        
        # Generate frequency vector
        freqs = np.linspace(case['freq_range'][0], case['freq_range'][1], case['n_freqs'])
        
        # Run test
        passed, corr, error = test_cwt_equivalence(
            signal, case['fs'], freqs, tolerance=0.01, verbose=True
        )
        
        results.append({
            'name': case['name'],
            'passed': passed,
            'correlation': corr,
            'max_error': error
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL TESTS")
    print("=" * 70)
    n_passed = sum(r['passed'] for r in results)
    print(f"Tests passed: {n_passed}/{len(results)}")
    
    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"{status} | {r['name']}")
        print(f"     Correlation: {r['correlation']:.6f}, Max Error: {r['max_error']:.6f}")
    
    print("=" * 70)
    
    return all(r['passed'] for r in results)


if __name__ == "__main__":
    import sys
    
    # Run comprehensive tests
    all_passed = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
