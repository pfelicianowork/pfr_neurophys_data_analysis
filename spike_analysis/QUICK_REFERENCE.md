# MUA Extraction Quick Reference

## One-Liner Cheat Sheet

```python
# Compute MUA for all regions
mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp, kernel_width=0.02)

# Visualize to validate
fig = spike_analysis.visualize_mua_segments(mua_by_region, t_lfp, windows=[(10, 12)])
```

## Common Patterns

### Pattern 1: Complete Workflow
```python
# Load and setup
from pfr_neurofunctions.spike_analysis import SpikeAnalysis, load_processed_spike_data
processed_data = load_processed_spike_data('units.npy')
spike_analysis = SpikeAnalysis(processed_data, sampling_rate=30000, duration=1000)
spike_analysis.assign_brain_regions({7: 'CA1', 8: 'CA1', 6: 'RTC'})

# Compute MUA
t_lfp = loader.time_vector()
mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp)

# Validate
fig = spike_analysis.visualize_mua_segments(mua_by_region, t_lfp, windows=[(10, 12)])
```

### Pattern 2: Single Region MUA
```python
mua_ca1 = spike_analysis.compute_mua('CA1', t_lfp, kernel_width=0.02)
```

### Pattern 3: Get Spike Times by Region
```python
ca1_spike_times = spike_analysis.get_spike_times_by_region('CA1')
# Returns: [array([0.1, 0.5, ...]), array([0.3, 0.8, ...]), ...]
```

### Pattern 4: SWR Detection Integration
```python
from pfr_neurofunctions.swr_detection.pipeline import detect_swr_by_region

mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp)
events, detectors = detect_swr_by_region(
    region_lfp=region_lfp,
    fs=1000.0,
    velocity=velocity,
    params=params,
    mua_by_region=mua_by_region,
    return_detectors=True
)
```

## Method Quick Reference

| Method | Purpose | Example |
|--------|---------|---------|
| `get_spike_times_by_region('CA1')` | Get spike times | `spikes = spike_analysis.get_spike_times_by_region('CA1')` |
| `get_units_by_region('CA1')` | Get unit IDs | `units = spike_analysis.get_units_by_region('CA1')` |
| `compute_mua('CA1', t_lfp)` | Single region MUA | `mua = spike_analysis.compute_mua('CA1', t_lfp)` |
| `compute_mua_all_regions(t_lfp)` | Multi-region MUA | `muas = spike_analysis.compute_mua_all_regions(t_lfp)` |
| `visualize_mua_segments(mua, t_lfp, windows)` | Visualize MUA | `fig = spike_analysis.visualize_mua_segments(...)` |

## Parameter Guide

### kernel_width (smoothing)
- `0.01` - Fine detail, noisier
- `0.02` - **Default**, good balance
- `0.05` - Smooth, slow dynamics

### windows (visualization)
```python
windows=[(10, 12)]              # Single 2-second window
windows=[(10, 12), (50, 52)]    # Two windows
windows=[(t, t+2) for t in range(0, 100, 20)]  # Many windows
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `"Brain regions haven't been assigned"` | Call `spike_analysis.assign_brain_regions(region_mapping)` first |
| MUA shape ≠ LFP shape | Use `t_lfp = loader.time_vector()`, not `loader.get_selected_trace()` |
| MUA is all zeros | Check: `len(spike_analysis.get_spike_times_by_region('CA1')) > 0` |
| Visualization shows no spikes | Set `show_spikes=True` in `visualize_mua_segments()` |

## Validation Checklist

✅ MUA shape matches LFP: `mua.shape == t_lfp.shape`  
✅ MUA is non-negative: `np.all(mua >= 0)`  
✅ MUA peaks align with spikes: Check visualization  
✅ Multiple units found: `len(spike_analysis.get_spike_times_by_region('CA1')) > 1`

## Copy-Paste Examples

### Example 1: Quick MUA Check
```python
# Quick check if MUA looks reasonable
for region in ['CA1', 'PFC', 'RTC']:
    mua = spike_analysis.compute_mua(region, t_lfp)
    print(f"{region}: mean={np.mean(mua):.2f} Hz, max={np.max(mua):.2f} Hz")
```

### Example 2: Normalized Comparison
```python
# Compare MUA across regions (normalized)
fig, ax = plt.subplots(figsize=(14, 4))
for region, mua in mua_by_region.items():
    mua_norm = (mua - np.mean(mua)) / np.std(mua)
    ax.plot(t_lfp, mua_norm, label=region, alpha=0.7)
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized MUA (Z-score)')
plt.show()
```

### Example 3: Custom Visualization
```python
# Custom window with detailed view
start, end = 50, 52
mask = (t_lfp >= start) & (t_lfp <= end)
t_seg = t_lfp[mask]

fig, ax = plt.subplots(figsize=(12, 4))
for region, mua in mua_by_region.items():
    ax.plot(t_seg, mua[mask], label=region, lw=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('MUA (Hz)')
ax.set_title(f'MUA Comparison: {start}-{end}s')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

## Links

- Full Guide: `MUA_EXTRACTION_GUIDE.md`
- Module README: `README.md`
- Examples: `basic_analysis_v1.ipynb` cells 18-21
