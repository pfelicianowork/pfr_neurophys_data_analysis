# Spike Analysis Module

## Overview

The `spike_analysis` module provides tools for analyzing neural spike data from multi-electrode recordings. It includes functionality for:

- Loading and processing spike data from Phy2 output
- Computing firing rates and spike statistics
- Organizing units by tetrodes and brain regions
- Computing Multi-Unit Activity (MUA) for population analysis
- Visualizing spike rasters and firing patterns
- Integration with SWR detection pipelines

## Installation

The module is part of the `pfr_neurofunctions` package:

```python
from pfr_neurofunctions.spike_analysis import (
    SpikeAnalysis,
    process_spike_data,
    load_processed_spike_data
)
```

## Quick Start

### 1. Process Raw Spike Data

First, process Phy2 output into a standardized format:

```python
from pfr_neurofunctions.spike_analysis import process_spike_data

# Process Phy2 output directory
npy_path = r'F:\your_session\phy_output'
save_path = r'F:\your_session'

process_spike_data(
    npy_path,
    save_path,
    samp_freq=30000  # Your recording sampling rate
)
# Creates: units.npy in save_path
```

### 2. Load and Analyze

```python
from pfr_neurofunctions.spike_analysis import (
    SpikeAnalysis,
    load_processed_spike_data
)

# Load processed data
units_file = r'F:\your_session\units.npy'
processed_data = load_processed_spike_data(units_file)

# Create analysis object
spike_analysis = SpikeAnalysis(
    processed_data=processed_data,
    sampling_rate=30000,
    duration=recording_duration  # in seconds
)
```

### 3. Assign Brain Regions

```python
# Map tetrodes to brain regions
region_mapping = {
    2: 'PFC',
    3: 'PFC',
    7: 'CA1',
    8: 'CA1',
    6: 'RTC'
}
spike_analysis.assign_brain_regions(region_mapping)

# Verify
for tetrode, region in region_mapping.items():
    units = spike_analysis.tetrode_mapping.get(tetrode, [])
    print(f"Tetrode {tetrode} ({region}): {len(units)} units")
```

### 4. Compute Multi-Unit Activity (MUA)

```python
# Get LFP timeline from your LFP loader
t_lfp = loader.time_vector()

# Compute MUA for all regions
mua_by_region = spike_analysis.compute_mua_all_regions(
    t_lfp=t_lfp,
    kernel_width=0.02  # 20ms Gaussian kernel
)

# Visualize
fig = spike_analysis.visualize_mua_segments(
    mua_by_region=mua_by_region,
    t_lfp=t_lfp,
    windows=[(10, 12), (50, 52)],
    show_spikes=True
)
```

### 5. Analyze Firing Rates

```python
# Analyze specific units or tetrodes
results = spike_analysis.analyze_data(
    identifiers=[7, 8],  # Tetrodes
    id_type='tetrode',
    kernel_width=0.05,
    sampling_rate=1000,
    time_range=(100, 200),  # Analyze 100-200s
    full_recording=True
)

# Plot
spike_analysis.plot_analysis(results, plot_type='both')
```

## Key Features

### SpikeAnalysis Class

Main class for spike data analysis with the following methods:

#### Data Organization
- `_create_tetrode_mapping()`: Maps tetrodes to units
- `assign_brain_regions(region_mapping)`: Assign brain regions to tetrodes
- `get_units_by_region(region)`: Get unit IDs in a region
- `get_spike_times_by_region(region)`: Extract spike times by region

#### Firing Rate Analysis
- `get_firing_rate(spike_times, bins, kernel_width, sampling_rate)`: Compute smoothed firing rates
- `analyze_data(identifiers, id_type, ...)`: Analyze firing rates for units/tetrodes
- `get_region_statistics(region)`: Get statistics for a brain region

#### Multi-Unit Activity (MUA)
- `compute_mua(region, t_lfp, kernel_width)`: Compute MUA for one region
- `compute_mua_all_regions(t_lfp, kernel_width, regions)`: Compute MUA for multiple regions
- `visualize_mua_segments(mua_by_region, t_lfp, windows)`: Visualize MUA with spike overlays

#### Visualization
- `plot_analysis(results, plot_type)`: Plot spike rasters and firing rates
- `plot_multiple_brain_regions(regions, ...)`: Multi-region spike raster plots

### Data Loading Functions

- `process_spike_data(npy_path, save_path, samp_freq)`: Process Phy2 output
- `load_processed_spike_data(units_file)`: Load processed spike data

## Data Format

Processed spike data is stored as a list of dictionaries, one per unit:

```python
{
    'cluster_id': int,           # Unit identifier from Phy
    'spk_samples': np.ndarray,   # Spike times in samples
    'spk_time': np.ndarray,      # Spike times in seconds
    'channel': int,              # Primary channel
    'ch': np.ndarray,            # All channels in tetrode
    'channel_group': int         # Tetrode/shank number
}
```

## Integration with Other Modules

### SWR Detection

MUA computed from `SpikeAnalysis` can be directly used in SWR detection:

```python
from pfr_neurofunctions.swr_detection import SWRDetector
from pfr_neurofunctions.swr_detection.pipeline import detect_swr_by_region

# Compute MUA
mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp)

# Use in SWR detection
events, detectors = detect_swr_by_region(
    region_lfp=region_lfp,
    fs=1000.0,
    velocity=velocity,
    params=params,
    mua_by_region=mua_by_region,  # ✓ Enable MUA-based detection
    return_detectors=True
)
```

### LFP Analysis

Combine with `open_ephys_loader` for simultaneous spike and LFP analysis:

```python
from pfr_neurofunctions.open_ephys_loader import fast_openephys_dat_lfp

# Load LFP
loader = fast_openephys_dat_lfp(...)
t_lfp = loader.time_vector()
lfp_data = loader.get_selected_trace('CA1_tet1')

# Compute MUA aligned with LFP
mua = spike_analysis.compute_mua('CA1', t_lfp)

# Now lfp_data and mua are aligned
```

## Examples

See complete examples in:
- `basic_analysis_v1.ipynb`: Full spike analysis workflow
- `tensor_preparation.ipynb`: Integration with SWR detection
- `MUA_EXTRACTION_GUIDE.md`: Detailed MUA extraction guide

## API Reference

### process_spike_data()

```python
process_spike_data(npy_path, save_path, samp_freq=30000)
```

Process Phy2 spike sorting output into standardized format.

**Parameters:**
- `npy_path` (str): Path to Phy2 output directory
- `save_path` (str): Where to save processed units.npy
- `samp_freq` (int): Recording sampling frequency (default: 30000)

**Creates:** `units.npy` file in `save_path`

---

### load_processed_spike_data()

```python
load_processed_spike_data(units_file)
```

Load previously processed spike data.

**Parameters:**
- `units_file` (str): Path to units.npy file

**Returns:** List of unit dictionaries

---

### SpikeAnalysis()

```python
SpikeAnalysis(processed_data, sampling_rate=30000, total_samples=None, duration=None)
```

Main analysis class for spike data.

**Parameters:**
- `processed_data` (list): List of unit dictionaries
- `sampling_rate` (int): Recording sampling rate (default: 30000)
- `total_samples` (int, optional): Total samples in recording
- `duration` (float, optional): Recording duration in seconds

**Attributes:**
- `processed_data`: Original spike data
- `sampling_rate`: Sampling frequency
- `duration`: Recording duration
- `unit_mapping`: Dict mapping cluster_id → unit data
- `tetrode_mapping`: Dict mapping tetrode → list of units
- `region_mapping`: Dict mapping tetrode → brain region (after assign_brain_regions)
- `region_to_tetrodes`: Dict mapping region → list of tetrodes

**See Also:**
- Full method documentation in class docstrings
- `MUA_EXTRACTION_GUIDE.md` for MUA-specific methods

## Requirements

- numpy
- scipy
- matplotlib

## Related Documentation

- [MUA Extraction Guide](MUA_EXTRACTION_GUIDE.md): Detailed guide for Multi-Unit Activity analysis
- [pfr_neurofunctions README](../../README.md): Main package documentation
- [SWR Detection](../swr_detection/README.md): Sharp-Wave Ripple detection module

## Contributing

When adding new functionality:
1. Follow existing code style
2. Add docstrings with parameters and returns
3. Update this README and relevant guides
4. Add examples to notebooks

## License

Part of the `pfr_neurofunctions` package.
