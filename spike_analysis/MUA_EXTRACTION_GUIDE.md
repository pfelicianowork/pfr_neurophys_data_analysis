# MUA (Multi-Unit Activity) Extraction Guide

## Overview

The `SpikeAnalysis` class now includes built-in methods for extracting and visualizing Multi-Unit Activity (MUA) from spike data. This is particularly useful for SWR (Sharp-Wave Ripple) detection and other population-level analyses.

## What is MUA?

MUA represents the aggregate spiking activity of multiple neurons in a brain region. It's computed by:
1. Binning spike times from all units in a region
2. Converting to firing rate (Hz)
3. Smoothing with a Gaussian kernel

MUA is useful for:
- Detecting population synchrony events (e.g., SWRs)
- Comparing activity levels across brain regions
- Understanding network dynamics

## New Methods

### 1. `get_spike_times_by_region(region)`

Extract spike times for all units in a brain region.

```python
# Get spike times for all CA1 units
ca1_spike_times = spike_analysis.get_spike_times_by_region('CA1')
# Returns: list of np.ndarray, one per unit
```

**Parameters:**
- `region` (str): Brain region name (e.g., 'CA1', 'PFC', 'RTC')

**Returns:**
- List of numpy arrays containing spike times (in seconds) for each unit

---

### 2. `get_units_by_region(region)`

Get all unit IDs assigned to a brain region.

```python
# Get unit cluster IDs in PFC
pfc_units = spike_analysis.get_units_by_region('PFC')
# Returns: [0, 1, 5, 8, ...]  (cluster IDs)
```

**Parameters:**
- `region` (str): Brain region name

**Returns:**
- List of unit cluster IDs

---

### 3. `compute_mua(region, t_lfp, kernel_width=0.02)`

Compute MUA for a single brain region.

```python
# Compute MUA for CA1 aligned with LFP timeline
mua_ca1 = spike_analysis.compute_mua(
    region='CA1',
    t_lfp=t_lfp,           # LFP time vector
    kernel_width=0.02       # 20ms Gaussian smoothing
)
```

**Parameters:**
- `region` (str): Brain region name
- `t_lfp` (np.ndarray): LFP time vector (in seconds) to align MUA with
- `kernel_width` (float): Standard deviation of Gaussian smoothing kernel in seconds (default: 0.02)

**Returns:**
- numpy array with shape matching `t_lfp`, representing firing rate (Hz)

**Notes:**
- Returns zeros if no units are found in the region
- Kernel width of 0.02s (20ms) is typical for MUA smoothing
- MUA is automatically aligned with the LFP timeline

---

### 4. `compute_mua_all_regions(t_lfp, kernel_width=0.02, regions=None)`

Compute MUA for multiple brain regions at once.

```python
# Compute MUA for all regions
mua_by_region = spike_analysis.compute_mua_all_regions(
    t_lfp=t_lfp,
    kernel_width=0.02,
    regions=None  # None = all assigned regions
)

# Or specify specific regions
mua_by_region = spike_analysis.compute_mua_all_regions(
    t_lfp=t_lfp,
    regions=['CA1', 'PFC']  # Only these regions
)
```

**Parameters:**
- `t_lfp` (np.ndarray): LFP time vector (in seconds)
- `kernel_width` (float): Gaussian kernel width in seconds (default: 0.02)
- `regions` (list of str, optional): List of regions to compute. If None, uses all assigned regions.

**Returns:**
- Dictionary mapping region names to MUA vectors

---

### 5. `visualize_mua_segments(mua_by_region, t_lfp, windows=None, figsize=None, show_spikes=True)`

Visualize MUA segments with overlaid spike times for validation.

```python
# Visualize MUA in specific time windows
fig = spike_analysis.visualize_mua_segments(
    mua_by_region=mua_by_region,
    t_lfp=t_lfp,
    windows=[(10, 12), (20, 22), (50, 52)],  # 2-second windows
    show_spikes=True  # Overlay individual unit spikes
)
plt.show()
```

**Parameters:**
- `mua_by_region` (dict): Dictionary of region → MUA vector (from `compute_mua_all_regions`)
- `t_lfp` (np.ndarray): LFP time vector
- `windows` (list of tuples, optional): Time windows as `[(start, end), ...]` in seconds. Default: `[(1, 3), (5, 7)]`
- `figsize` (tuple, optional): Figure size (width, height). Auto-calculated if None.
- `show_spikes` (bool): Whether to overlay individual unit spike times (default: True)

**Returns:**
- matplotlib Figure object

---

## Complete Workflow Example

```python
import numpy as np
import matplotlib.pyplot as plt
from pfr_neurofunctions.spike_analysis import SpikeAnalysis, load_processed_spike_data

# 1. Load spike data
units_file = r'F:\Spikeinterface_practice\s4_rec\units.npy'
processed_spike_data = load_processed_spike_data(units_file)

# 2. Create SpikeAnalysis instance
spike_analysis = SpikeAnalysis(
    processed_data=processed_spike_data,
    sampling_rate=30000,
    duration=1000.0  # recording duration in seconds
)

# 3. Assign brain regions to tetrodes
region_mapping = {
    2: 'PFC',
    3: 'PFC',
    7: 'CA1',
    8: 'CA1',
    6: 'RTC'
}
spike_analysis.assign_brain_regions(region_mapping)

# 4. Get LFP timeline (from your LFP loader)
t_lfp = loader.time_vector()  # shape: (N_samples,)

# 5. Compute MUA for all regions
mua_by_region = spike_analysis.compute_mua_all_regions(
    t_lfp=t_lfp,
    kernel_width=0.02  # 20ms Gaussian kernel
)

# 6. Verify shapes
for region, mua_vec in mua_by_region.items():
    n_units = len(spike_analysis.get_spike_times_by_region(region))
    print(f"{region}: {n_units} units → MUA shape: {mua_vec.shape}")

# 7. Visualize MUA segments
fig = spike_analysis.visualize_mua_segments(
    mua_by_region=mua_by_region,
    t_lfp=t_lfp,
    windows=[(10, 12), (50, 52), (100, 102)],
    show_spikes=True
)
plt.show()
```

---

## Integration with SWR Detection

The computed MUA can be directly used with `SWRDetector`:

```python
from pfr_neurofunctions.swr_detection import SWRParams, SWRDetector
from pfr_neurofunctions.swr_detection.pipeline import detect_swr_by_region

# Compute MUA
mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp)

# Single-region detection with MUA
region = "CA1"
lfp_array = region_lfp[region]
mua_vec = mua_by_region[region]

swr = SWRDetector(
    lfp_data=lfp_array,
    fs=1000.0,
    mua_data=mua_vec,  # ✓ MUA enabled
    velocity_data=velocity,
    params=params,
)
swr.detect_events(channels="all", average_mode=True)
swr.classify_events_improved()

# Multi-region detection with MUA
events_by_region, detectors_by_region = detect_swr_by_region(
    region_lfp=region_lfp,
    fs=1000.0,
    velocity=velocity,
    params=params,
    mua_by_region=mua_by_region,  # ✓ Pass all regions' MUA
    classify=True,
    return_detectors=True
)
```

---

## Tips and Best Practices

### Choosing Kernel Width

- **Default (0.02s = 20ms)**: Good general-purpose smoothing for MUA
- **Narrow (0.01s = 10ms)**: Preserves more temporal detail, but noisier
- **Wide (0.05s = 50ms)**: Smoother, better for slow dynamics

```python
# Try different kernel widths
mua_fine = spike_analysis.compute_mua('CA1', t_lfp, kernel_width=0.01)
mua_smooth = spike_analysis.compute_mua('CA1', t_lfp, kernel_width=0.05)
```

### Validating MUA Quality

Always visualize a few segments to ensure MUA looks reasonable:

```python
# Check if MUA peaks align with actual spikes
fig = spike_analysis.visualize_mua_segments(
    mua_by_region, t_lfp,
    windows=[(5, 7), (20, 22)],
    show_spikes=True  # Essential for validation
)
```

**What to look for:**
- MUA peaks should coincide with clusters of spike times (vertical lines)
- MUA should be non-negative
- MUA should be smooth but not over-smoothed

### Handling Regions with Few Units

Regions with only 1-2 units will have sparse MUA:

```python
# Check number of units per region
for region in ['CA1', 'PFC', 'RTC']:
    n_units = len(spike_analysis.get_spike_times_by_region(region))
    print(f"{region}: {n_units} units")
    if n_units < 3:
        print(f"  ⚠️  Warning: {region} has few units, MUA may be sparse")
```

### Comparing Across Regions

Normalize MUA when comparing across regions with different unit counts:

```python
# Z-score normalization
for region, mua in mua_by_region.items():
    mua_normalized = (mua - np.mean(mua)) / np.std(mua)
    plt.plot(t_lfp, mua_normalized, label=region)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Normalized MUA (Z-score)')
```

---

## Troubleshooting

### Error: "Brain regions haven't been assigned"

**Solution:** Call `assign_brain_regions()` first:

```python
spike_analysis.assign_brain_regions(region_mapping)
```

### MUA shape doesn't match LFP

**Solution:** Ensure `t_lfp` is the actual LFP time vector:

```python
t_lfp = loader.time_vector()  # Not loader.get_selected_trace()!
print(f"LFP timeline shape: {t_lfp.shape}")
```

### MUA is all zeros

**Possible causes:**
1. No units in the specified region
2. Spike times outside the LFP timeline range

**Solution:**

```python
# Check units
spike_times = spike_analysis.get_spike_times_by_region('CA1')
if len(spike_times) == 0:
    print("No units found in CA1")
else:
    print(f"Found {len(spike_times)} units")
    print(f"Spike time range: {np.min(np.concatenate(spike_times)):.2f} - {np.max(np.concatenate(spike_times)):.2f}s")
    print(f"LFP time range: {t_lfp[0]:.2f} - {t_lfp[-1]:.2f}s")
```

---

## API Reference Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_spike_times_by_region(region)` | Extract spike times for a region | List of arrays |
| `get_units_by_region(region)` | Get unit IDs in a region | List of cluster IDs |
| `compute_mua(region, t_lfp, kernel_width)` | Compute MUA for one region | Array (N_samples,) |
| `compute_mua_all_regions(t_lfp, kernel_width, regions)` | Compute MUA for multiple regions | Dict of arrays |
| `visualize_mua_segments(mua_by_region, t_lfp, windows)` | Visualize MUA with spikes | matplotlib Figure |

---

## See Also

- `pfr_neurofunctions.swr_detection.pipeline.compute_region_mua_from_spikes()`: Alternative low-level MUA computation
- `pfr_neurofunctions.swr_detection.SWRDetector`: SWR detection with MUA support
- Example notebooks: `basic_analysis_v1.ipynb`, `tensor_preparation.ipynb`
