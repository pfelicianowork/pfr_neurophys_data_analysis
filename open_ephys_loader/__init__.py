"""
open_ephys_loader - Module for loading Open Ephys data in multiple formats

This module provides a unified interface for loading Open Ephys data from both
legacy (.continuous files) and modern formats (structure.openephys).
"""

from .data_loader import (
    load_open_ephys_data,
    detect_format,
    detect_sources,
    get_sample_rate,
    get_continuous_data,
    DataFormat
)
from . import utils
from . import processing
from . import visualization
from .lazy_loader import LazyBinaryLoader
from .visualization import unified_viewer, dat_file_viewer
from .load_openephys_dat_lazy import load_openephys_dat_lazy, load_openephys_dat_lazy_with_tetrodes, test_channel_differences
from .fast_lfp import fast_openephys_dat_lfp, open_cached_lfp, build_cached_lfp, CachedLFPLoader

__all__ = [
    'load_open_ephys_data',
    'detect_format',
    'detect_sources',
    'get_sample_rate',
    'get_continuous_data',
    'DataFormat',
    'LazyBinaryLoader',
    'unified_viewer',
    'dat_file_viewer',
    'load_openephys_dat_lazy',
    'load_openephys_dat_lazy_with_tetrodes',
    'test_channel_differences',
    'fast_openephys_dat_lfp',
    'open_cached_lfp',
    'build_cached_lfp',
    'CachedLFPLoader',
    'utils',
    'processing',
    'visualization'
]
