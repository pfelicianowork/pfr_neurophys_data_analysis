"""
GPU Acceleration Quick Test
============================

Run this script to verify GPU acceleration is working correctly.
"""

import sys, os
print("="*60)
print("DEBUG: CuPy/CUDA status at script startup")
try:
    import cupy
    print("CuPy version:", cupy.__version__)
    print("CUDA runtime version:", cupy.cuda.runtime.runtimeGetVersion())
    print("CUDA driver version:", cupy.cuda.runtime.driverGetVersion())
    print("GPU name:", cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())
except Exception as e:
    print("CuPy/CUDA ERROR:", e)
print("="*60)

print("=" * 70)
print("GPU ACCELERATION QUICK TEST")
print("=" * 70)

# Step 1: Check GPU availability
print("\n📌 Step 1: Checking GPU availability...")
from swr_detection.gpu_utils import check_gpu_availability, print_gpu_info

print_gpu_info()

has_cupy, has_cuda, cuda_version, gpu_name = check_gpu_availability()

if not (has_cupy and has_cuda):
    print("\n❌ GPU not available. Install CuPy:")
    print("   pip install cupy-cuda11x  (or cupy-cuda12x)")
    exit(1)

# Step 2: Generate synthetic test data
print("\n📌 Step 2: Generating synthetic test data...")
import numpy as np

fs = 1000.0
duration = 0.4
t = np.arange(0, duration, 1/fs)
test_signal = np.sin(2 * np.pi * 150 * t) + 0.3 * np.random.randn(len(t))
freqs = np.linspace(100, 250, 60)

print(f"   Signal: {len(test_signal)} samples @ {fs} Hz")
print(f"   Frequencies: {len(freqs)} bins ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz)")

# Step 3: Test CPU version
print("\n📌 Step 3: Testing CPU CWT...")
from swr_detection.swr_spectral_features import _cwt_optimized
import time

start = time.time()
cwt_cpu = _cwt_optimized(test_signal, fs, freqs, boundary='mirror', n_workers=4, verbose=False)
cpu_time = time.time() - start

print(f"   ✅ CPU complete: {cpu_time*1000:.1f} ms")
print(f"   Output shape: {cwt_cpu.shape}")

# Step 4: Test GPU version
print("\n📌 Step 4: Testing GPU CWT...")
from swr_detection.swr_spectral_features_gpu import _cwt_optimized_gpu

start = time.time()
cwt_gpu = _cwt_optimized_gpu(test_signal, fs, freqs, boundary='mirror', max_freqs_per_batch=50, verbose=False)
gpu_time = time.time() - start

print(f"   ✅ GPU complete: {gpu_time*1000:.1f} ms")
print(f"   Output shape: {cwt_gpu.shape}")

# Step 5: Validate equivalence
print("\n📌 Step 5: Validating GPU vs CPU equivalence...")
from scipy.stats import pearsonr

correlation, _ = pearsonr(cwt_cpu.flatten(), cwt_gpu.flatten())
max_error = np.max(np.abs(cwt_gpu - cwt_cpu) / (np.abs(cwt_cpu) + 1e-10))

print(f"   Correlation: {correlation:.6f}")
print(f"   Max relative error: {max_error:.6f} ({max_error*100:.4f}%)")

# Step 6: Compute speedup
speedup = cpu_time / gpu_time
print("\n📌 Step 6: Performance analysis...")
print(f"   CPU time: {cpu_time*1000:.2f} ms")
print(f"   GPU time: {gpu_time*1000:.2f} ms")
print(f"   Speedup: {speedup:.2f}x")

# Final verdict
print("\n" + "=" * 70)
if correlation > 0.99 and max_error < 0.01:
    print("✅ SUCCESS: GPU acceleration is working correctly!")
    if speedup > 5:
        print(f"🚀 Excellent speedup: {speedup:.1f}x faster than CPU")
    elif speedup > 2:
        print(f"⚠️  Modest speedup: {speedup:.1f}x (expected 5-10x for larger datasets)")
    else:
        print(f"⚠️  Low speedup: {speedup:.1f}x (overhead dominates for small test)")
else:
    print("❌ FAILED: GPU results don't match CPU")
    print(f"   Correlation: {correlation:.6f} (expected > 0.99)")
    print(f"   Max error: {max_error:.6f} (expected < 0.01)")

print("=" * 70)

print("\n💡 Next steps:")
print("   1. Run full equivalence test: python swr_detection/test_gpu_equivalence.py")
print("   2. Run benchmark: python swr_detection/benchmark_gpu_speedup.py --n_events 1000")
print("   3. Update your notebook to use use_gpu='auto' in batch_compute_spectral_features()")
