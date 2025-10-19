"""
Advanced visualization utilities for SWR detection results.
Following pynapple conventions for neurophysiological data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json


def _get_event_attribute(event, attr_name, default=None):
    """
    Safely get an attribute from an event, handling different event formats.
    
    Parameters
    ----------
    event : object, dict, or str
        The event object/dictionary/string
    attr_name : str
        Name of the attribute to retrieve
    default : any
        Default value if attribute not found
        
    Returns
    -------
    value : any
        The attribute value or default
    """
    # Handle string events (likely JSON)
    if isinstance(event, str):
        try:
            event_dict = json.loads(event)
            return event_dict.get(attr_name, default)
        except (json.JSONDecodeError, TypeError):
            return default
    
    # Handle dictionary events
    if isinstance(event, dict):
        return event.get(attr_name, default)
    
    # Handle object events
    if hasattr(event, attr_name):
        return getattr(event, attr_name)
    
    # Handle object events with get_ methods
    getter_name = f"get_{attr_name}"
    if hasattr(event, getter_name):
        getter = getattr(event, getter_name)
        if callable(getter):
            return getter()
    
    return default


def visualize_swr_events_overview(
    events_by_region: Dict[str, List],
    t_lfp: np.ndarray,
    duration: float,
    regions: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Create overview raster plot of all detected SWR events across regions.
    
    Parameters
    ----------
    events_by_region : dict
        Dictionary of detected events per region
    t_lfp : np.ndarray
        LFP timeline vector
    duration : float
        Total recording duration in seconds
    regions : list, optional
        Regions to plot (default: all)
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
        The created figure
    """
    if regions is None:
        regions = list(events_by_region.keys())
    
    fig, axes = plt.subplots(len(regions), 1, figsize=figsize, sharex=True)
    if len(regions) == 1:
        axes = [axes]
    
    colors = {'CA1': '#FF6B6B', 'RTC': '#4ECDC4', 'PFC': '#45B7D1'}
    
    for idx, (ax, region) in enumerate(zip(axes, regions)):
        events = events_by_region.get(region, [])
        
        # Plot each event as vertical span
        for event in events:
            start = _get_event_attribute(event, 'start_time', 
                                       _get_event_attribute(event, 'start_idx', 0) / 1000)
            end = _get_event_attribute(event, 'end_time',
                                     _get_event_attribute(event, 'end_idx', 0) / 1000)
            peak = _get_event_attribute(event, 'peak_time', (start + end) / 2)
            
            # Event span
            ax.axvspan(start, end, alpha=0.3, color=colors.get(region, 'gray'))
            # Peak marker
            ax.axvline(peak, color=colors.get(region, 'gray'), 
                      linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_ylabel(f'{region}\n({len(events)} events)', fontsize=11)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    axes[-1].set_xlim(0, duration)
    axes[0].set_title('Detected SWR Events Across Regions', 
                     fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


def visualize_single_event_detailed(
    event_idx: int,
    region: str,
    detectors_by_region: Dict,
    region_lfp: Dict[str, np.ndarray],
    t_lfp: np.ndarray,
    fs: float,
    mua_by_region: Optional[Dict[str, np.ndarray]] = None,
    velocity: Optional[np.ndarray] = None,
    window_sec: float = 0.6,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create detailed visualization of a single SWR event with multiple panels.
    
    Parameters
    ----------
    event_idx : int
        Index of event to visualize
    region : str
        Brain region name
    detectors_by_region : dict
        Dictionary of SWRDetector instances
    region_lfp : dict
        Dictionary of LFP arrays by region
    t_lfp : np.ndarray
        LFP timeline
    fs : float
        Sampling frequency
    mua_by_region : dict, optional
        MUA vectors by region
    velocity : np.ndarray, optional
        Velocity vector
    window_sec : float
        Time window around event (seconds)
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    detector = detectors_by_region[region]
    events = detector.swr_events

    if event_idx >= len(events):
        raise ValueError(f"Event {event_idx} not found in {region}")
    
    event = events[event_idx]
    peak_time = _get_event_attribute(event, 'peak_time', 0)
    start_time = _get_event_attribute(event, 'start_time', peak_time - 0.05)
    end_time = _get_event_attribute(event, 'end_time', peak_time + 0.05)
    
    # Time window
    t_start = max(0, peak_time - window_sec / 2)
    t_end = min(t_lfp[-1], peak_time + window_sec / 2)
    mask = (t_lfp >= t_start) & (t_lfp <= t_end)
    t_window = t_lfp[mask]
    
    # Prepare data
    lfp_data = region_lfp[region][:, mask]
    
    # Create figure with dynamic panels
    n_panels = 3 + (mua_by_region is not None) + (velocity is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    panel_idx = 0
    
    # Panel 1: Raw LFP (all channels)
    ax = axes[panel_idx]
    for ch_idx in range(lfp_data.shape[0]):
        offset = ch_idx * np.std(lfp_data[ch_idx, :]) * 3
        ax.plot(t_window, lfp_data[ch_idx, :] + offset, 
               'k-', lw=0.5, alpha=0.6)
    ax.axvspan(start_time, end_time, alpha=0.2, color='red', label='SWR')
    ax.axvline(peak_time, color='red', linestyle='--', lw=2, label='Peak')
    ax.set_ylabel('Raw LFP\n(all channels)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'{region} - Event {event_idx} (Peak: {peak_time:.3f}s)', 
                fontsize=13, fontweight='bold')
    panel_idx += 1
    
    # Panel 2: Average filtered LFP
    ax = axes[panel_idx]
    lfp_avg = np.mean(lfp_data, axis=0)
    ax.plot(t_window, lfp_avg, 'b-', lw=1.5, label='Ripple (150-250 Hz)')
    ax.axvspan(start_time, end_time, alpha=0.2, color='red')
    ax.axvline(peak_time, color='red', linestyle='--', lw=2)
    ax.set_ylabel('Filtered LFP\n(average)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    panel_idx += 1
    
    # Panel 3: Ripple power envelope
    ax = axes[panel_idx]
    ripple_env = getattr(detector, '_ripple_envelope', None)
    if ripple_env is not None and len(ripple_env) == len(t_lfp):
        env_window = ripple_env[mask]
        ax.plot(t_window, env_window, 'r-', lw=2, label='Ripple Power')
        threshold = detector.params.threshold_multiplier * np.std(ripple_env)
        ax.axhline(threshold, color='gray', linestyle='--', 
                  lw=1.5, label=f'Threshold ({detector.params.threshold_multiplier}×SD)')
    ax.axvspan(start_time, end_time, alpha=0.2, color='red')
    ax.axvline(peak_time, color='red', linestyle='--', lw=2)
    ax.set_ylabel('Ripple Power\n(envelope)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    panel_idx += 1
    
    # Panel 4: MUA (if available)
    if mua_by_region is not None and region in mua_by_region:
        ax = axes[panel_idx]
        mua_window = mua_by_region[region][mask]
        ax.plot(t_window, mua_window, 'g-', lw=2, label='MUA')
        ax.fill_between(t_window, 0, mua_window, alpha=0.3, color='green')
        ax.axvspan(start_time, end_time, alpha=0.2, color='red')
        ax.axvline(peak_time, color='red', linestyle='--', lw=2)
        ax.set_ylabel('MUA\n(Hz)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        panel_idx += 1
    
    # Panel 5: Velocity (if available)
    if velocity is not None:
        ax = axes[panel_idx]
        vel_window = velocity[mask]
        ax.plot(t_window, vel_window, 'k-', lw=1.5, label='Velocity')
        ax.axhline(5.0, color='orange', linestyle='--', 
                  lw=1.5, label='Immobility threshold')
        ax.axvspan(start_time, end_time, alpha=0.2, color='red')
        ax.axvline(peak_time, color='red', linestyle='--', lw=2)
        ax.set_ylabel('Velocity\n(cm/s)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        panel_idx += 1
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout()
    return fig


def visualize_events_grid(
    events_by_region: Dict,
    detectors_by_region: Dict,
    region_lfp: Dict,
    t_lfp: np.ndarray,
    fs: float,
    n_events: int = 6,
    regions: Optional[List[str]] = None,
    window_sec: float = 0.4,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create grid of multiple SWR events for quick overview.
    
    Parameters
    ----------
    events_by_region : dict
        Events per region
    detectors_by_region : dict
        Detector instances per region
    region_lfp : dict
        LFP data per region
    t_lfp : np.ndarray
        Timeline
    fs : float
        Sampling frequency
    n_events : int
        Number of events to show per region
    regions : list, optional
        Regions to include
    window_sec : float
        Time window per event
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if regions is None:
        regions = list(events_by_region.keys())
    
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, n_events, 
                            figsize=figsize, 
                            sharex=True, sharey='row')
    
    if n_regions == 1:
        axes = axes.reshape(1, -1)
    if n_events == 1:
        axes = axes.reshape(-1, 1)
    
    for row, region in enumerate(regions):
        events = events_by_region[region][:n_events]
        if hasattr(events, 'to_dict'):  # pandas DataFrame
            events = events.to_dict('records')
        detector = detectors_by_region[region]
        lfp_data = region_lfp[region]

        for col, event in enumerate(events):
            ax = axes[row, col]

            peak_time = _get_event_attribute(event, 'peak_time', 0)
            start_time = _get_event_attribute(event, 'start_time', peak_time - 0.05)
            end_time = _get_event_attribute(event, 'end_time', peak_time + 0.05)
            
            # Extract window
            t_start = peak_time - window_sec / 2
            t_end = peak_time + window_sec / 2
            mask = (t_lfp >= t_start) & (t_lfp <= t_end)
            t_window = t_lfp[mask] - peak_time  # Center at 0
            
            # Plot average LFP
            lfp_avg = np.mean(lfp_data[:, mask], axis=0)
            ax.plot(t_window, lfp_avg, 'k-', lw=1)
            
            # Highlight event
            ax.axvspan(start_time - peak_time, end_time - peak_time,
                      alpha=0.3, color='red')
            ax.axvline(0, color='red', linestyle='--', lw=1.5, alpha=0.7)
            
            # Formatting
            if row == 0:
                ax.set_title(f'Event {col}', fontsize=10)
            if col == 0:
                ax.set_ylabel(region, fontsize=11)
            if row == n_regions - 1:
                ax.set_xlabel('Time (s)', fontsize=9)
            
            ax.grid(alpha=0.3)
    
    fig.suptitle('SWR Events Grid View', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_event_statistics_comparison(
    events_by_region: Dict,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Compare event statistics across regions with 6-panel visualization.
    
    Parameters
    ----------
    events_by_region : dict
        Events dictionary
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    regions = list(events_by_region.keys())
    colors = {'CA1': '#FF6B6B', 'RTC': '#4ECDC4', 'PFC': '#45B7D1'}
    
    # 1. Event count
    ax1 = fig.add_subplot(gs[0, 0])
    counts = [len(events_by_region[r]) for r in regions]
    bars = ax1.bar(regions, counts, color=[colors.get(r, 'gray') for r in regions])
    ax1.set_ylabel('Number of Events', fontsize=11)
    ax1.set_title('Total Event Count', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    # 2. Duration distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for region in regions:
        events = events_by_region[region]
        durations = []
        for e in events:
            start = _get_event_attribute(e, 'start_time', 
                                       _get_event_attribute(e, 'start_idx', 0) / 1000)
            end = _get_event_attribute(e, 'end_time',
                                     _get_event_attribute(e, 'end_idx', 0) / 1000)
            durations.append(end - start)
        if durations:
            ax2.hist(durations, bins=30, alpha=0.5, 
                    label=region, color=colors.get(region, 'gray'))
    ax2.set_xlabel('Duration (s)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Duration Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. Inter-event intervals
    ax3 = fig.add_subplot(gs[1, 0])
    for region in regions:
        events = events_by_region[region]
        peak_times = sorted([_get_event_attribute(e, 'peak_time', 0) for e in events])
        if len(peak_times) > 1:
            iei = np.diff(peak_times)
            ax3.hist(iei, bins=30, alpha=0.5,
                    label=region, color=colors.get(region, 'gray'))
    ax3.set_xlabel('Inter-Event Interval (s)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Inter-Event Intervals', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # 4. Peak amplitude
    ax4 = fig.add_subplot(gs[1, 1])
    for region in regions:
        events = events_by_region[region]
        amplitudes = [_get_event_attribute(e, 'peak_amplitude', 0) for e in events 
                     if _get_event_attribute(e, 'peak_amplitude', None) is not None]
        if amplitudes:
            ax4.hist(amplitudes, bins=30, alpha=0.5,
                    label=region, color=colors.get(region, 'gray'))
    ax4.set_xlabel('Peak Amplitude (SD)', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Peak Amplitude Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # 5. Event rate over time
    ax5 = fig.add_subplot(gs[2, 0])
    for region in regions:
        events = events_by_region[region]
        peak_times = sorted([_get_event_attribute(e, 'peak_time', 0) for e in events])
        if peak_times:
            max_time = max(peak_times)
            bins = np.arange(0, max_time + 60, 60)
            counts, _ = np.histogram(peak_times, bins=bins)
            rates = counts
            ax5.plot(bins[:-1] / 60, rates, '-o', 
                    label=region, color=colors.get(region, 'gray'))
    ax5.set_xlabel('Time (min)', fontsize=11)
    ax5.set_ylabel('Events per minute', fontsize=11)
    ax5.set_title('Event Rate Over Time', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Duration vs amplitude scatter
    ax6 = fig.add_subplot(gs[2, 1])
    for region in regions:
        events = events_by_region[region]
        durations = []
        amplitudes = []
        for e in events:
            amp = _get_event_attribute(e, 'peak_amplitude', None)
            if amp is not None:
                start = _get_event_attribute(e, 'start_time', 
                                           _get_event_attribute(e, 'start_idx', 0) / 1000)
                end = _get_event_attribute(e, 'end_time',
                                         _get_event_attribute(e, 'end_idx', 0) / 1000)
                durations.append(end - start)
                amplitudes.append(amp)
        
        if durations:
            ax6.scatter(durations, amplitudes, alpha=0.6, s=30,
                       label=region, color=colors.get(region, 'gray'))
    ax6.set_xlabel('Duration (s)', fontsize=11)
    ax6.set_ylabel('Peak Amplitude (SD)', fontsize=11)
    ax6.set_title('Duration vs Amplitude', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    fig.suptitle('SWR Event Statistics Comparison', 
                fontsize=15, fontweight='bold', y=0.995)
    
    return fig