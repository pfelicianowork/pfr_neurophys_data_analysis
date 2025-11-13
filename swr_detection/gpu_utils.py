"""
GPU Utilities for SWR Spectral Analysis

Provides GPU availability checking and memory estimation for CWT operations.
"""

import warnings
import numpy as np


def check_gpu_availability():
    """
    Check if GPU acceleration is available and working.
    
    Returns:
    --------
    has_cupy : bool
        True if CuPy is installed
    has_cuda : bool
        True if CUDA is available and functional
    cuda_version : int or None
        CUDA runtime version (e.g., 11020 for CUDA 11.2)
    gpu_name : str or None
        Name of the GPU device
    """
    try:
        import cupy as cp
        has_cupy = True
        
        # Test CUDA availability
        try:
            _ = cp.cuda.Device(0)
            has_cuda = True
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            # CuPy 13+ uses getDeviceProperties() to get device name
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        except Exception as e:
            has_cuda = False
            cuda_version = None
            gpu_name = None
            warnings.warn(f"CuPy installed but CUDA unavailable: {e}")
            
        return has_cupy, has_cuda, cuda_version, gpu_name
        
    except ImportError:
        return False, False, None, None


def estimate_gpu_memory_per_event(n_freqs, n_samples, dtype='complex128'):
    """
    Estimate GPU memory needed for one CWT event.
    
    Parameters:
    -----------
    n_freqs : int
        Number of frequencies to analyze
    n_samples : int
        Number of time samples in the signal
    dtype : str
        Data type ('complex128' or 'complex64')
    
    Returns:
    --------
    memory_mb : float
        Estimated memory usage in MB
    """
    bytes_per_element = 16 if dtype == 'complex128' else 8
    
    # Memory components:
    # 1. Input signal FFT (n_samples * complex)
    # 2. Wavelet FFT per frequency (n_samples * complex)
    # 3. Convolution result per frequency (n_samples * complex)
    # 4. Output array (n_freqs * n_samples * float or complex)
    
    # Conservative estimate: 2x for FFT buffers + result array
    memory_bytes = n_samples * bytes_per_element * (2 + n_freqs)
    
    return memory_bytes / (1024**2)


def get_optimal_gpu_batch_size(n_freqs, n_samples, safety_factor=0.5):
    """
    Estimate optimal batch size for GPU processing based on available memory.
    
    Parameters:
    -----------
    n_freqs : int
        Number of frequencies per event
    n_samples : int
        Number of time samples per event
    safety_factor : float
        Fraction of free memory to use (default 0.5 = 50%)
    
    Returns:
    --------
    batch_size : int
        Recommended number of frequencies to process per batch
    """
    try:
        import cupy as cp
        free_mem, total_mem = cp.cuda.Device(0).mem_info
        
        # Estimate memory per frequency
        mem_per_freq_mb = estimate_gpu_memory_per_event(1, n_samples)
        
        # Available memory in MB
        available_mb = (free_mem * safety_factor) / (1024**2)
        
        # Calculate batch size
        batch_size = max(10, int(available_mb / mem_per_freq_mb))
        
        return min(batch_size, n_freqs)  # Cap at total frequencies
        
    except Exception as e:
        warnings.warn(f"Could not estimate GPU batch size: {e}. Using default=50")
        return 50


def print_gpu_info():
    """
    Print detailed GPU information for diagnostics.
    """
    has_cupy, has_cuda, cuda_version, gpu_name = check_gpu_availability()
    
    print("=" * 70)
    print("GPU ACCELERATION STATUS")
    print("=" * 70)
    
    if not has_cupy:
        print("❌ CuPy: NOT INSTALLED")
        print("   Install with: pip install cupy-cuda11x  (or cupy-cuda12x)")
        print("=" * 70)
        return
    
    print("✅ CuPy: INSTALLED")
    
    if not has_cuda:
        print("❌ CUDA: NOT AVAILABLE")
        print("   Check NVIDIA driver installation")
    else:
        print("✅ CUDA: AVAILABLE")
        print(f"   Runtime Version: {cuda_version}")
        print(f"   GPU Name: {gpu_name}")
        
        try:
            import cupy as cp
            free_mem, total_mem = cp.cuda.Device(0).mem_info
            print(f"   GPU Memory: {free_mem / 1024**3:.1f} GB free / {total_mem / 1024**3:.1f} GB total")
            print(f"   Compute Capability: {cp.cuda.Device(0).compute_capability}")
        except:
            pass
    
    print("=" * 70)


if __name__ == "__main__":
    # Run diagnostics when executed directly
    print_gpu_info()
    
    # Test memory estimation
    print("\nMemory Estimates (example: 400ms event @ 1kHz, 60 frequencies):")
    n_samples = 400  # 400ms @ 1kHz
    n_freqs = 60
    mem_mb = estimate_gpu_memory_per_event(n_freqs, n_samples)
    print(f"  Estimated memory per event: {mem_mb:.1f} MB")
    
    has_cupy, has_cuda, _, _ = check_gpu_availability()
    if has_cupy and has_cuda:
        batch_size = get_optimal_gpu_batch_size(n_freqs, n_samples)
        print(f"  Recommended batch size: {batch_size} frequencies")
