# Summary: MUA Extraction Integration into SpikeAnalysis

## What Was Added

### 1. **New Methods in `SpikeAnalysis` Class** (`pfr_neurofunctions/spike_analysis/analysis.py`)

#### Helper Methods
- **`get_spike_times_by_region(region)`**: Extract spike times for all units in a brain region
- **`get_units_by_region(region)`**: Get unit cluster IDs in a brain region

#### MUA Computation Methods
- **`compute_mua(region, t_lfp, kernel_width=0.02)`**: Compute Multi-Unit Activity for a single region
  - Bins spike times along LFP timeline
  - Converts to firing rate (Hz)
  - Smooths with Gaussian kernel
  
- **`compute_mua_all_regions(t_lfp, kernel_width=0.02, regions=None)`**: Compute MUA for multiple regions
  - Returns dictionary mapping region names to MUA vectors
  - All MUA vectors aligned with LFP timeline

#### Visualization Methods
- **`visualize_mua_segments(mua_by_region, t_lfp, windows, show_spikes=True)`**: Visualize MUA validation
  - Shows MUA waveforms in specified time windows
  - Overlays individual unit spike times for validation
  - Multi-region, multi-window grid layout

### 2. **Documentation**

#### MUA_EXTRACTION_GUIDE.md
- Comprehensive guide to MUA extraction
- Method documentation with examples
- Integration with SWR detection
- Tips and best practices
- Troubleshooting guide

#### README.md
- Module overview and quick start
- API reference
- Integration examples
- Data format documentation

### 3. **Example Notebook Cells** (`basic_analysis_v1.ipynb`)

Added 4 new cells after cell 17 (the manual MUA extraction):

1. **Markdown cell**: Introduction to new methods
2. **Method 1**: Compute MUA using built-in methods + comparison with manual approach
3. **Method 2**: Visualize MUA segments with spike overlays
4. **Method 3**: Compare MUA across regions (raw + normalized)

## How to Use

### Basic Usage

```python
# 1. Setup (as before)
from pfr_neurofunctions.spike_analysis import SpikeAnalysis, load_processed_spike_data

units_file = r'F:\your_session\units.npy'
processed_data = load_processed_spike_data(units_file)

spike_analysis = SpikeAnalysis(
    processed_data=processed_data,
    sampling_rate=30000,
    duration=recording_duration
)

region_mapping = {7: 'CA1', 8: 'CA1', 6: 'RTC', 2: 'PFC', 3: 'PFC'}
spike_analysis.assign_brain_regions(region_mapping)

# 2. NEW: Compute MUA (replaces manual extraction)
t_lfp = loader.time_vector()
mua_by_region = spike_analysis.compute_mua_all_regions(
    t_lfp=t_lfp,
    kernel_width=0.02
)

# 3. NEW: Visualize for validation
fig = spike_analysis.visualize_mua_segments(
    mua_by_region=mua_by_region,
    t_lfp=t_lfp,
    windows=[(10, 12), (50, 52), (100, 102)],
    show_spikes=True
)
plt.show()

# 4. Use with SWR detection (as before)
from pfr_neurofunctions.swr_detection.pipeline import detect_swr_by_region

events, detectors = detect_swr_by_region(
    region_lfp=region_lfp,
    fs=1000.0,
    velocity=velocity,
    params=params,
    mua_by_region=mua_by_region,  # ✓ Direct use
    return_detectors=True
)
```

### Single Region

```python
# Compute MUA for just one region
mua_ca1 = spike_analysis.compute_mua('CA1', t_lfp, kernel_width=0.02)

# Use in single-region SWRDetector
swr = SWRDetector(
    lfp_data=region_lfp['CA1'],
    fs=1000.0,
    mua_data=mua_ca1,
    velocity_data=velocity,
    params=params
)
```

## Key Improvements

### Before (Manual Extraction)
```python
# Required understanding of internal data structures
spike_times_by_region = {}
for region in ['CA1', 'RTC', 'PFC']:
    spike_times_by_region[region] = []
    for tetrode, reg in region_mapping.items():
        if reg == region:
            units_in_tetrode = spike_analysis.tetrode_mapping.get(tetrode, [])
            for unit_data in units_in_tetrode:
                spike_times = unit_data['spk_time']
                spike_times_by_region[region].append(spike_times)

# Then call external function
from pfr_neurofunctions.swr_detection.pipeline import compute_region_mua_from_spikes
mua_by_region = compute_region_mua_from_spikes(
    spike_times_by_region, t_lfp, sigma=0.02
)
```

### After (Built-in Methods)
```python
# Simple, self-contained
mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp, kernel_width=0.02)
```

**Benefits:**
- ✅ Cleaner API - one method call instead of manual loops
- ✅ Better encapsulation - no need to know internal data structure
- ✅ Integrated validation - built-in visualization with spike overlays
- ✅ Consistent with class design - follows existing `SpikeAnalysis` patterns
- ✅ Better documentation - comprehensive guides and examples

## Testing the New Features

Run the new notebook cells (cells 18-21 after the manual extraction cell):

1. **Cell 18**: Computes MUA using new methods and verifies they match manual approach
2. **Cell 19**: Visualizes MUA segments with spike overlays (3 windows × 3 regions)
3. **Cell 20**: Shows detailed view of one region across multiple windows
4. **Cell 21**: Compares raw and normalized MUA across regions

Expected output:
- MUA shapes match LFP timeline length
- MUA values match manual method (max difference ~0)
- Visualization shows MUA peaks aligned with spike times
- Clean, informative plots

## File Structure

```
pfr_neurofunctions/
└── spike_analysis/
    ├── __init__.py               (no changes)
    ├── analysis.py               ✓ Modified - added 5 new methods
    ├── loader.py                 (no changes)
    ├── README.md                 ✓ New - module documentation
    └── MUA_EXTRACTION_GUIDE.md   ✓ New - comprehensive MUA guide

basic_analysis_v1.ipynb           ✓ Modified - added 4 example cells
```

## Dependencies

No new dependencies required. Uses existing:
- `numpy`
- `scipy.signal.gaussian` (for kernel generation)
- `matplotlib.pyplot` (for visualization)

## Backward Compatibility

✅ All existing functionality preserved
✅ No breaking changes to existing code
✅ New methods are additive only

Existing code continues to work:
```python
# Old pattern still works
results = spike_analysis.analyze_data(identifiers=[7, 8], id_type='tetrode')
spike_analysis.plot_analysis(results)

# New pattern available
mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp)
```

## Next Steps

1. **Try the example cells** in your notebook to see the new functionality
2. **Validate MUA quality** using the visualization methods
3. **Use MUA in SWR detection** to improve event detection accuracy
4. **Explore different kernel widths** (0.01-0.05s) to find optimal smoothing

## Questions?

Refer to:
- `MUA_EXTRACTION_GUIDE.md` - Detailed usage and troubleshooting
- `README.md` - Module overview and API reference
- Example cells in `basic_analysis_v1.ipynb` - Working examples
