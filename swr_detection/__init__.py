"""
pfr_neurofunctions.swr_detection - Sharp Wave Ripple Detection Library

This module provides comprehensive tools for detecting and analyzing sharp wave ripples (SWRs)
in neural data, including advanced signal processing, HMM-based edge detection, and
sophisticated classification algorithms.

Main Classes:
- SWRDetector: Core detection and analysis class
- SWRParams: Configuration parameters for detection

Key Features:
- Multi-channel SWR detection
- HMM-based edge refinement
- MUA integration
- Advanced event classification
- Interactive visualization
- Statistical analysis
"""

from .detector import SWRDetector
from .params import SWRParams, PRESETS
from .visualization import SWRVisualizer
from .analysis import SWRAnalyzer
from .utils import *
from .pipeline import (
    find_region_channels,
    build_region_lfp,
    interpolate_velocity_to_lfp,
    interpolate_signal_to_lfp,
    compute_immobility_mask,
    detector_events_to_df,
    pick_time_column,
    detect_swr_by_region,
    quick_overlay_plot,
    build_spike_times_by_region,
)
from .mua import compute_region_mua_from_spikes
# from swr_hmm_detection import (
#     SWRHMMDetector,
#     SWRHMMParams,   
# )

__version__ = "1.0.0"
__all__ = [
    'SWRDetector',
    'SWRParams',
    'PRESETS',
    'SWRVisualizer',
    'SWRAnalyzer',
    'find_region_channels',
    'build_region_lfp',
    'interpolate_velocity_to_lfp',
    'interpolate_signal_to_lfp',
    'compute_immobility_mask',
    'detector_events_to_df',
    'pick_time_column',
    'detect_swr_by_region',
    'quick_overlay_plot',
    'build_spike_times_by_region',
    'compute_region_mua_from_spikes',
]
