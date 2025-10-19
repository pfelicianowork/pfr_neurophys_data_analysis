# SWR Detection Module

A comprehensive Python module for detecting and analyzing Sharp Wave Ripples (SWRs) in neural data, particularly from Open Ephys recordings.

## Overview

This module provides tools for:
- **SWR Detection**: Automated detection of sharp wave ripples in neural signals
- **Event Classification**: Classification of detected events into single events and bursts
- **Statistical Analysis**: Comprehensive analysis of temporal patterns, cross-channel correlations, and burst statistics
- **Visualization**: Plotting and visualization of detected events and analysis results

## Installation

The module is part of the `pfr_neurofunctions` package. Ensure you have the required dependencies:

```bash
pip install numpy scipy pandas matplotlib
```

## Quick Start

```python
import numpy as np
from pfr_neurofunctions.swr_detection import SWRDetector, SWRVisualizer, SWRAnalyzer

# Load your neural data (channels x time)
# data = load_your_data()  # Shape: (n_channels, n_samples)

# Initialize detector with parameters
detector = SWRDetector(
    sampling_rate=30000,  # Hz
    frequency_band=(150, 250),  # Hz for ripples
    detection_threshold=3.0,  # Standard deviations
    min_duration=0.02,  # Minimum event duration in seconds
    max_duration=0.5     # Maximum event duration in seconds
)

# Detect SWR events
events = detector.detect_swrs(data)

# Visualize results
visualizer = SWRVisualizer(detector)
visualizer.plot_detection_summary()

# Analyze events
analyzer = SWRAnalyzer(detector)
analyzer.create_analysis_report()
```

## Module Structure

### Core Classes

#### `SWRDetector`
Main class for detecting SWR events in neural data.

**Key Methods:**
- `detect_swrs(data)`: Detect SWR events in multi-channel data
- `classify_events()`: Classify events into single events and bursts
- `get_basic_stats()`: Get basic statistics about detected events

#### `SWRVisualizer`
Tools for visualizing detection results and analysis.

**Key Methods:**
- `plot_detection_summary()`: Overview of detected events
- `plot_channel_activity()`: Per-channel activity visualization
- `plot_event_waveforms()`: Individual event waveforms

#### `SWRAnalyzer`
Statistical analysis tools for detected events.

**Key Methods:**
- `compute_temporal_statistics()`: Temporal analysis of event patterns
- `compute_cross_channel_correlations()`: Cross-channel correlation analysis
- `create_analysis_report()`: Comprehensive analysis report

### Parameter Classes

#### `SWRParams`
Configuration parameters for SWR detection.

**Key Parameters:**
- `sampling_rate`: Data sampling rate in Hz
- `frequency_band`: Frequency band for ripple detection (low, high) in Hz
- `detection_threshold`: Threshold in standard deviations
- `min_duration`/`max_duration`: Event duration constraints in seconds

## Detailed Usage Examples

### Basic Detection Workflow

```python
from pfr_neurofunctions.swr_detection import SWRDetector, SWRParams

# Set up parameters
params = SWRParams(
    sampling_rate=30000,
    frequency_band=(150, 250),
    detection_threshold=2.5,
    min_duration=0.015,
    max_duration=0.2
)

# Initialize detector
detector = SWRDetector(params)

# Load and preprocess data
data = load_open_ephys_data('path/to/data')  # Your data loading function
data = filter_and_preprocess(data)  # Your preprocessing

# Detect events
events = detector.detect_swrs(data)
print(f"Detected {len(events)} SWR events")
```

### Multi-Channel Analysis

```python
# Analyze events per channel
for channel in range(data.shape[0]):
    channel_events = [e for e in events if e['channel'] == channel]
    print(f"Channel {channel}: {len(channel_events)} events")

# Cross-channel correlation analysis
analyzer = SWRAnalyzer(detector)
correlations = analyzer.compute_cross_channel_correlations()
```

### Event Classification and Burst Detection

```python
# Classify events into single events and bursts
detector.classify_events()

# Analyze burst statistics
burst_stats = analyzer.compute_burst_statistics()
print(f"Found {burst_stats['n_bursts']} bursts")
print(f"Mean burst size: {burst_stats['mean_burst_size']:.2f} events")
```

### Visualization Examples

```python
visualizer = SWRVisualizer(detector)

# Overview plot
visualizer.plot_detection_summary()

# Per-channel activity
visualizer.plot_channel_activity()

# Event waveforms
visualizer.plot_event_waveforms(event_indices=[0, 1, 2])

# Raster plot
visualizer.plot_raster_plot()
```

## Parameter Tuning

### Detection Threshold
- **Lower values (1.5-2.5)**: More sensitive, may detect more false positives
- **Higher values (3.0-4.0)**: More conservative, fewer false positives
- **Recommendation**: Start with 2.5-3.0 and adjust based on data

### Frequency Band
- **Typical range**: 150-250 Hz for ripples
- **Adjust based on**: Signal characteristics and noise levels
- **Validation**: Check power spectral density of detected events

### Duration Constraints
- **min_duration**: Filter out very brief noise artifacts
- **max_duration**: Exclude prolonged oscillations that aren't SWRs
- **Typical values**: 0.015-0.02 s (min), 0.2-0.5 s (max)

## Integration with Open Ephys

```python
from pfr_neurofunctions.open_ephys_loader import OpenEphysLoader

# Load Open Ephys data
loader = OpenEphysLoader('path/to/experiment')
data = loader.load_continuous_data()

# Apply SWR detection
detector = SWRDetector(sampling_rate=loader.sampling_rate)
events = detector.detect_swrs(data)

# Save results
detector.save_events('swr_events.json')
```

## Output Formats

### Event Data Structure
Each detected event contains:
```python
{
    'channel': int,           # Channel number
    'start_time': float,      # Start time in seconds
    'peak_time': float,       # Peak time in seconds
    'end_time': float,        # End time in seconds
    'duration': float,        # Duration in seconds
    'peak_power': float,      # Peak power/amplitude
    'event_type': str,        # 'single' or 'burst'
    'waveform': array,        # Raw waveform data
    'filtered_waveform': array # Bandpass filtered waveform
}
```

### Analysis Results
Statistical analysis provides comprehensive metrics including:
- Temporal statistics (rates, intervals)
- Cross-channel correlations
- Burst characteristics
- Channel-specific properties

## Best Practices

1. **Data Quality**: Ensure clean, artifact-free data before detection
2. **Parameter Validation**: Validate parameters on a subset of data
3. **Multi-channel Consistency**: Check for consistent detection across channels
4. **Statistical Validation**: Use multiple metrics to validate detection quality

## Troubleshooting

### Common Issues

**Too many false positives:**
- Increase `detection_threshold`
- Adjust `frequency_band` to avoid noise
- Increase `min_duration`

**Missing events:**
- Decrease `detection_threshold`
- Widen `frequency_band`
- Check data quality and filtering

**Inconsistent channel detection:**
- Verify channel data quality
- Check for channel-specific artifacts
- Adjust parameters per channel if needed

## Citation

If you use this module in your research, please cite:

```
SWR Detection Module
Part of pfr_neurofunctions package
[Your Lab/Institution]
```

## Contributing

Contributions are welcome! Please submit issues and pull requests on the project repository.

## License

This module is part of the pfr_neurofunctions package. See the main package license for details.
