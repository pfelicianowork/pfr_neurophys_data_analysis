"""
GPU vs CPU Performance Benchmark for CWT Spectral Analysis (Batched Approach)

Measures speedup achieved by GPU acceleration on synthetic and real datasets
by processing all events as a single batch.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import sys
import os
# Assuming the original script's directory structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mock Implementations (for demonstration) ---
# In a real scenario, these would be your actual, complex importable functions.

def _cwt_optimized_cpu_batched(batch_signals, fs, freqs, n_workers=20, **kwargs):
    """
    MOCK CPU function.
    Assumes this function can take a 2D array [n_events, n_samples]
    and process it in parallel using Dask (n_workers).
    """
    # Simulate work
    time.sleep(0.001 * batch_signals.shape[0] / n_workers) 
    n_events, n_samples = batch_signals.shape
    n_freqs = len(freqs)
    # Return a mock result shape
    return np.abs(np.random.randn(n_events, n_freqs, n_samples) + 1j)

def _cwt_optimized_gpu_batched(batch_signals_cpu, fs, freqs, max_freqs_per_batch=50, **kwargs):
    """
    MOCK GPU function.
    Demonstrates the batch transfer and batched processing.
    """
    import cupy as cp
    
    # 1. Transfer the *entire batch* to GPU VRAM at once
    batch_signals_gpu = cp.asarray(batch_signals_cpu)
    
    # Simulate the CWT computation happening entirely on the GPU
    # In a real CuPy implementation, this would be a series of CuPy FFTs,
    # convolutions, and element-wise operations on the batch_signals_gpu tensor.
    
    # Simulate work (GPU is much faster per event, but has a small base overhead)
    time.sleep(0.01 + 0.00001 * batch_signals_gpu.shape[0]) 
    
    n_events, n_samples = batch_signals_gpu.shape
    n_freqs = len(freqs)
    
    # Simulate result creation on GPU
    result_gpu = cp.abs(cp.random.randn(n_events, n_freqs, n_samples) + 1j)
    
    # 2. Transfer the *entire result* back to CPU RAM at once
    result_cpu = cp.asnumpy(result_gpu)
    
    return result_cpu

# --- End Mock Implementations ---


def generate_synthetic_events(n_events=1000, duration=0.4, fs=1000.0, ripple_freq=150, seed=42):
    """
    Generate synthetic LFP events for benchmarking.
    
    Returns:
    --------
    events : np.ndarray
        A 2D array of shape (n_events, n_samples)
    """
    np.random.seed(seed)
    n_samples = int(duration * fs)
    # Create a 2D array directly
    events = np.zeros((n_events, n_samples))
    t = np.arange(n_samples) / fs
    
    for i in range(n_events):
        # Vary frequency slightly per event
        freq = ripple_freq + np.random.randn() * 10
        signal = np.sin(2 * np.pi * freq * t) + 0.3 * np.random.randn(n_samples)
        events[i, :] = signal
    
    return events


def benchmark_batch_cwt(events_batch, fs, freqs, use_gpu=False, n_workers=20, gpu_batch_size=50):
    """
    Benchmark batch CWT processing.
    This function now processes ALL events in a single call.

    Parameters:
    -----------
    events_batch : np.ndarray
        A 2D array of shape [n_events, n_samples]
    
    Returns:
    --------
    total_time : float
        Total execution time in seconds
    """
    
    n_events = events_batch.shape[0]
    
    if use_gpu:
        # We use our new mock/real batched GPU function
        # from swr_detection.swr_spectral_features_gpu import _cwt_optimized_gpu_batched
        
        start_total = time.time()
        results = _cwt_optimized_gpu_batched(
            events_batch, fs, freqs,
            boundary='mirror',
            max_freqs_per_batch=gpu_batch_size,
            verbose=False
        )
        total_time = time.time() - start_total
        
    else:
        # We assume there is a batched CPU function
        # from swr_detection.swr_spectral_features import _cwt_optimized_cpu_batched
        
        start_total = time.time()
        results = _cwt_optimized_cpu_batched(
            events_batch, fs, freqs,
            boundary='mirror',
            n_workers=n_workers,
            verbose=False
        )
        total_time = time.time() - start_total
    
    # Calculate average time per event
    avg_time = total_time / n_events
    
    return total_time, avg_time


def run_benchmark(n_events=1000, duration=0.4, fs=1000.0, freq_range=(100, 250), n_freqs=60, n_workers=20, gpu_batch_size=50):
    """
    Run comprehensive CPU vs GPU benchmark using a single batch.
    """
    # Assuming gpu_utils is available
    # from swr_detection.gpu_utils import check_gpu_availability, print_gpu_info, estimate_gpu_memory_per_event
    
    # --- Mock GPU Utils for standalone running ---
    def check_gpu_availability():
        try:
            import cupy
            return True, True, "Mock CUDA 11.x", "Mock GPU (e.g., RTX 3080)"
        except ImportError:
            return False, False, None, None
    def print_gpu_info():
        print("Checking GPU...")
    def estimate_gpu_memory_per_event(n_freqs, n_samples):
        return (n_freqs * n_samples * 8 * 2) / (1024**2) # float64, complex
    # --- End Mock GPU Utils ---

    print("=" * 80)
    print("GPU vs CPU CWT PERFORMANCE BENCHMARK (BATCHED)")
    print("=" * 80)
    
    # Check GPU availability
    print_gpu_info()
    has_cupy, has_cuda, cuda_version, gpu_name = check_gpu_availability()
    
    if not (has_cupy and has_cuda):
        print("\n❌ GPU (CuPy) not available. Benchmark aborted.")
        # return None
        # For demo, we'll continue but GPU will fail
        pass 
    
    # Generate synthetic data
    print(f"\nGenerating {n_events} synthetic events as a single batch...")
    # This is now a single 2D array
    events_batch = generate_synthetic_events(n_events, duration, fs)
    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    
    print(f"Event parameters:")
    print(f"  - Duration: {duration*1000:.1f} ms")
    print(f"  - Sampling rate: {fs:.0f} Hz")
    print(f"  - Samples per event: {int(duration*fs)}")
    print(f"  - Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
    print(f"  - Frequency bins: {n_freqs}")
    print(f"  - Total events: {n_events}")
    print(f"  - Batch shape: {events_batch.shape}")
    
    # Warm-up runs (to avoid cold-start bias)
    print("\nWarm-up runs (processing 10 events)...")
    _ = benchmark_batch_cwt(events_batch[:10], fs, freqs, use_gpu=False, n_workers=n_workers)
    if has_cupy:
        _ = benchmark_batch_cwt(events_batch[:10], fs, freqs, use_gpu=True, gpu_batch_size=gpu_batch_size)
    
    # CPU benchmark
    print("\n" + "=" * 80)
    print("🖥️  CPU BENCHMARK (pyfftw + Dask)")
    print("=" * 80)
    print(f"Workers: {n_workers}")
    print(f"Processing {n_events} events in one batch...")
    
    cpu_total, cpu_avg = benchmark_batch_cwt(
        events_batch, fs, freqs, use_gpu=False, n_workers=n_workers
    )
    
    print(f"✅ CPU Benchmark Complete")
    print(f"   Total time: {cpu_total:.2f} seconds")
    print(f"   Average per event: {cpu_avg*1000:.2f} ms")
    print(f"   Events per second: {n_events/cpu_total:.1f}")
    
    # GPU benchmark
    if not has_cupy:
        print("\nGPU not found. Skipping GPU benchmark.")
        return None

    print("\n" + "=" * 80)
    print("🚀 GPU BENCHMARK (CuPy + CUDA)")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"Frequency batch size (internal GPU kernel): {gpu_batch_size}")
    print(f"Processing {n_events} events in one batch...")
    
    gpu_total, gpu_avg = benchmark_batch_cwt(
        events_batch, fs, freqs, use_gpu=True, gpu_batch_size=gpu_batch_size
    )
    
    print(f"✅ GPU Benchmark Complete")
    print(f"   Total time: {gpu_total:.2f} seconds")
    print(f"   Average per event: {gpu_avg*1000:.2f} ms")
    print(f"   Events per second: {n_events/gpu_total:.1f}")
    
    # Compute speedup
    speedup = cpu_total / gpu_total
    speedup_avg = cpu_avg / gpu_avg
    
    print("\n" + "=" * 80)
    print("📊 SPEEDUP ANALYSIS")
    print("=" * 80)
    print(f"Total time speedup: {speedup:.2f}x")
    print(f"Per-event speedup: {speedup_avg:.2f}x")
    print(f"Time saved: {cpu_total - gpu_total:.2f} seconds ({(1 - gpu_total/cpu_total)*100:.1f}% faster)")
    
    # Extrapolate to larger datasets
    print(f"\n📈 Extrapolation to larger datasets:")
    for scale in [5000, 10000, 50000]:
        cpu_est = (cpu_total / n_events) * scale
        gpu_est = (gpu_total / n_events) * scale
        saved = cpu_est - gpu_est
        print(f"   {scale:,} events: CPU {cpu_est/60:.1f} min vs GPU {gpu_est/60:.1f} min (save {saved/60:.1f} min)")
    
    # Memory efficiency
    mem_per_event = estimate_gpu_memory_per_event(n_freqs, int(duration * fs))
    total_batch_memory = mem_per_event * n_events
    print(f"\n💾 Memory Usage:")
    print(f"   Estimated GPU memory per event (result): {mem_per_event:.2f} MB")
    print(f"   Total memory for this batch ({n_events} events): {total_batch_memory / 1024:.2f} GB")
    
    # Determine if speedup is acceptable
    if speedup >= 10:
        verdict = "✅ EXCELLENT (10x+ speedup achieved)"
    elif speedup >= 5:
        verdict = "✅ GOOD (5-10x speedup)"
    elif speedup >= 2:
        verdict = "⚠️ MODEST (2-5x speedup)"
    else:
        verdict = "❌ POOR (<2x speedup - check overhead or problem size)"
    
    print(f"\n🏁 Final Verdict: {verdict}")
    print("=" * 80)
    
    results = {
        'n_events': n_events,
        'cpu_total_time': cpu_total,
        'cpu_avg_time': cpu_avg,
        'gpu_total_time': gpu_total,
        'gpu_avg_time': gpu_avg,
        'speedup_total': speedup,
        'speedup_avg': speedup_avg,
        'gpu_name': gpu_name,
        'cuda_version': cuda_version,
        'n_freqs': n_freqs,
        'duration': duration,
        'fs': fs
    }
    
    return results

# We can reuse the original plot function, it only needs the results dict
def plot_benchmark_results(results):
    """
    Visualize benchmark results. (Identical to original)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    methods = ['CPU\n(pyfftw)', 'GPU\n(CuPy)']
    # Note: We plot the *average* time, which is more comparable
    times = [results['cpu_avg_time'] * 1000, results['gpu_avg_time'] * 1000]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Time per Event (ms)', fontweight='bold', fontsize=12)
    axes[0].set_title('Average Processing Time', fontweight='bold', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_yscale('log') # Use log scale, as GPU may be orders of magnitude faster
    
    # Add values on bars
    for i, (method, time_val) in enumerate(zip(methods, times)):
        axes[0].text(i, time_val, f'{time_val:.3f} ms', # Show more precision
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Speedup visualization
    speedup = results['speedup_avg']
    axes[1].bar(['GPU Speedup'], [speedup], color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].axhline(1, color='red', linestyle='--', lw=2, label='Baseline (1x)')
    axes[1].axhline(10, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='Target (10x)')
    axes[1].set_ylabel('Speedup Factor (X)', fontweight='bold', fontsize=12)
    axes[1].set_title(f'GPU Speedup: {speedup:.1f}x', fontweight='bold', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add text for speedup
    text_y_pos = max(speedup * 1.05, 1.5) # Place text above bar, or at 1.5 if bar is tiny
    axes[1].text(0, text_y_pos, f'{speedup:.1f}x',
                 ha='center', va='bottom', fontweight='bold', fontsize=14, color='darkgreen')
    
    plt.suptitle(f"CWT Batched Performance ({results['n_events']} events, {results['n_freqs']} freqs)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark GPU vs CPU CWT performance (Batched)')
    parser.add_argument('--n_events', type=int, default=1000, help='Number of events to benchmark')
    parser.add_argument('--duration', type=float, default=0.4, help='Event duration in seconds')
    parser.add_argument('--n_freqs', type=int, default=60, help='Number of frequency bins')
    parser.add_argument('--n_workers', type=int, default=20, help='CPU workers for parallel processing')
    parser.add_argument('--gpu_batch_size', type=int, default=50, help='GPU frequency batch size (internal)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        n_events=args.n_events,
        duration=args.duration,
        n_freqs=args.n_freqs,
        n_workers=args.n_workers,
        gpu_batch_size=args.gpu_batch_size
    )
    
    # Plot if requested
    if results and args.plot:
        try:
            import matplotlib.pyplot as plt
            fig = plot_benchmark_results(results)
            plt.show()
        except ImportError:
            print("\nPlotting requires matplotlib. Please install it: pip install matplotlib")
