# ===============================
# Open Ephys Data Loader
# ===============================
# This module provides functions to load Open Ephys data from both legacy and modern formats,
# automatically detecting the format and using the appropriate loader.
#
# Typical workflow:
#   1. format = detect_format(directory)
#   2. data = load_open_ephys_data(directory, format_type=format)
#   3. continuous = get_continuous_data(data)
#   4. downsampled = _downsample_data(continuous, target_sample_rate)
#   5. For .dat files: use load_openephys_dat(...)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

def load_openephys_dat(filepath, num_channels, dtype='int16', channel_ids=None,
                       start_time=None, end_time=None, sampling_frequency=30000.0,
                       file_offset=0, target_sampling_frequency=None, plot=False, return_dict=False,
                       verbose=True, time_axis=0):
    """
    Efficiently load Open Ephys .dat file and extract specific channels and time window using memory mapping.
    Optionally downsample and plot the data. Uses SpikeInterface-compatible binary loading approach.

    Parameters
    ----------
    filepath : str
        Path to the .dat file.
    num_channels : int
        Number of channels in the recording.
    dtype : str
        Data type of the recording (default: 'int16').
    channel_ids : list or None
        List of channel indices to extract (default: None, returns all).
        IMPORTANT: Use 0-based indexing (0, 1, 2, ...) for channel numbers.
        For example, to load channels 1, 2, 3 from a typical probe setup, use [0, 1, 2].
    start_time : float or None
        Start time in seconds (default: None, returns from beginning).
    end_time : float or None
        End time in seconds (default: None, returns to end).
    sampling_frequency : float
        Sampling frequency in Hz (default: 30000.0).
    file_offset : int
        Number of bytes to skip at the start of the file (default: 0).
    target_sampling_frequency : float or None
        If set, downsample to this frequency.
    plot : bool or str
        If True, show a quick plot of the loaded data.
        If 'separate', plot each channel in separate matplotlib figures (for debugging).
        If 'widget' or 'multi', show interactive multi-channel viewer.
    return_dict : bool
        If True, return data as dictionary with channel keys.
    verbose : bool
        If True, print information about loading process (default: True).
    time_axis : int
        The axis of the time dimension (default: 0).
        - 0: time axis is first (samples, channels)
        - 1: time axis is second (channels, samples)
        Use this to handle different .dat file formats.

    Returns
    -------
    data : np.ndarray or dict
        If return_dict=False: Array of shape (n_channels_selected, n_samples_selected).
        If return_dict=True: Dict with channel data and metadata.
    """
    import os
    dtype_np = np.dtype(dtype)
    bytes_per_sample = num_channels * dtype_np.itemsize
    file_size = os.path.getsize(filepath) - file_offset

    if verbose:
        print(f"Loading .dat file: {filepath}")
        print(f"  File size: {file_size:,} bytes")
        print(f"  Bytes per sample: {bytes_per_sample}")
        print(f"  Number of channels: {num_channels}")

    num_samples = file_size // bytes_per_sample

    if num_channels == 0:
        raise ValueError(f"Number of channels cannot be zero")

    if num_samples == 0:
        raise ValueError(f"File appears to be empty or corrupted: {filepath}")

    if verbose:
        duration = num_samples / sampling_frequency
        print(f"  Number of samples: {num_samples:,}")
        print(f"  Recording duration: {duration:.2f} seconds")

    # SPIKEINTERFACE-STYLE MEMORY MAPPING: Use mmap for proper alignment
    import mmap

    # Create memory-mapped file using SpikeInterface approach
    file_obj = open(filepath, 'rb')
    file_size_bytes = os.path.getsize(filepath) - file_offset

    # Align memory mapping offset to ALLOCATIONGRANULARITY for performance
    memmap_offset, start_offset = divmod(file_offset, mmap.ALLOCATIONGRANULARITY)
    memmap_offset *= mmap.ALLOCATIONGRANULARITY
    memmap_length = file_size_bytes + start_offset

    # Create mmap object
    memmap_obj = mmap.mmap(file_obj.fileno(), length=memmap_length,
                          access=mmap.ACCESS_READ, offset=memmap_offset)

    if verbose:
        print(f"  Memory mapping: offset={memmap_offset}, length={memmap_length:,} bytes")

    # Create numpy array from mmap buffer
    if time_axis == 0:
        shape = (num_samples, num_channels)
    else:  # time_axis == 1
        shape = (num_channels, num_samples)

    raw = np.ndarray(shape=shape, dtype=dtype_np, buffer=memmap_obj, offset=start_offset)

    # Transpose if time_axis indicates channels come first
    if time_axis == 1:
        raw = raw.T

    if verbose:
        print(f"  Memory map shape: {raw.shape} (time_axis={time_axis})")
    file_obj.close()  # Don't close mmap object until raw is destroyed

    # Time window selection
    start_idx = int(start_time * sampling_frequency) if start_time is not None else 0
    end_idx = int(end_time * sampling_frequency) if end_time is not None else num_samples

    # Validate time indices
    if start_idx >= num_samples:
        raise ValueError(f"Start time {start_time}s corresponds to sample {start_idx}, but recording only has {num_samples} samples")
    if end_idx > num_samples:
        end_idx = num_samples  # Cap at end of recording
    if end_idx <= start_idx:
        raise ValueError(f"End time {end_time}s <= start time {start_time}s ")

    if start_idx > 0 or end_idx < num_samples:
        raw = raw[start_idx:end_idx]
        actual_samples = end_idx - start_idx
        if verbose:
            print(f"  Selected time window: {start_time if start_time is not None else 0:.2f} - {end_time if end_time is not None else duration:.2f}s ({actual_samples:,} samples)")
    else:
        actual_samples = num_samples

    # Channel selection with validation
    if channel_ids is not None:
        # Validate channel IDs
        invalid_channels = [ch for ch in channel_ids if ch < 0 or ch >= num_channels]
        if invalid_channels:
            raise ValueError(f"Invalid channel IDs: {invalid_channels}. Must be between 0 and {num_channels-1}")

        ch_indices = list(channel_ids)
        if len(set(ch_indices)) != len(ch_indices):
            # Remove duplicates
            ch_indices = list(dict.fromkeys(ch_indices))
            if verbose:
                print(f"  Warning: Duplicate channel IDs provided, using unique set: {ch_indices}")
    else:
        ch_indices = list(range(num_channels))

    if verbose:
        print(f"  Loading {len(ch_indices)} channels: {ch_indices}")

    # FIXED CHANNEL EXTRACTION: Extract each channel individually to ensure correctness
    # raw[:, ch_indices].T was causing issues - let's do it explicitly
    data = np.zeros((len(ch_indices), actual_samples), dtype=dtype_np)
    for i, ch_idx in enumerate(ch_indices):
        data[i, :] = raw[:, ch_idx]

    if verbose:
        print(f"  Data shape after channel extraction: {data.shape}")
        for i, ch_idx in enumerate(ch_indices):
            ch_mean = data[i, :1000].mean()
            ch_std = data[i, :1000].std()
            print(f"    Channel {ch_idx}: mean={ch_mean:.2f}, std={ch_std:.2f}")

    # Downsampling
    fs_out = sampling_frequency
    if target_sampling_frequency is not None and target_sampling_frequency < sampling_frequency:
        factor = int(sampling_frequency / target_sampling_frequency)
        data = decimate(data, factor, axis=1, zero_phase=True)
        fs_out = target_sampling_frequency

    # Prepare output dictionary
    data_dict = {f'Ch{ch}': {'data': data[i], 'header': {'sampleRate': fs_out}} for i, ch in enumerate(ch_indices)}

    # Visualization
    if plot == 'separate':
        # Special mode for debugging: plot each channel in separate matplotlib figures
        import matplotlib.pyplot as plt

        print(f"Plotting {len(data_dict)} channels separately for debugging...")

        for channel_name, channel_data in data_dict.items():
            plt.figure(figsize=(12, 4))
            signal = channel_data['data']
            time = np.arange(len(signal)) / channel_data['header']['sampleRate']

            # Plot first 10000 samples for quick inspection
            plot_samples = min(10000, len(signal))
            plt.plot(time[:plot_samples], signal[:plot_samples], 'b-', linewidth=0.5)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (µV)')
            plt.title(f'{channel_name} - First {plot_samples} samples')
            plt.grid(True, alpha=0.3)

            # Add statistics annotation
            plt.text(0.02, 0.95, f'mean: {signal[:plot_samples].mean():.2f}\nstd: {signal[:plot_samples].std():.2f}',
                    transform=plt.gca().transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            plt.tight_layout()
            plt.show()

        return data_dict if return_dict else data

    elif plot in ('widget', 'multi'):
        from .visualization import multi_channel_viewer
        # Pass data_dict directly with channel keys
        return multi_channel_viewer(data_dict, channels=list(data_dict.keys()))
    elif plot is True:
        from .visualization import quick_view
        quick_view(data_dict, method='simple')

    # Return format
    if return_dict:
        return data_dict
    else:
        return data

"""
data_loader.py - Core module for loading Open Ephys data

This module provides functions to load Open Ephys data from both legacy and modern formats,
automatically detecting the format and using the appropriate loader.
"""

import os
import sys
import glob
from pathlib import Path

# Add paths to existing Open Ephys loaders
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent

# Add old format loader path
_old_loader_path = _project_root / "oe_old_analysis_tools" / "Python3"
if str(_old_loader_path) not in sys.path:
    sys.path.insert(0, str(_old_loader_path))

# Add new format loader path
_new_loader_path = _project_root / "oe_python_tools" / "src"
if str(_new_loader_path) not in sys.path:
    sys.path.insert(0, str(_new_loader_path))

# Import the actual loaders
try:
    import OpenEphys as OldFormatLoader
    OLD_FORMAT_AVAILABLE = True
except ImportError as e:
    OLD_FORMAT_AVAILABLE = False
    print(f"Warning: Old format loader not available: {e}")

try:
    from open_ephys.analysis.formats.OpenEphysRecording import OpenEphysRecording
    NEW_FORMAT_AVAILABLE = True
except ImportError as e:
    NEW_FORMAT_AVAILABLE = False
    print(f"Warning: New format loader not available: {e}")


class DataFormat:
    """Enum-like class for data format types"""
    OLD = "old_format"
    NEW = "new_format"
    UNKNOWN = "unknown"


def detect_sources(directory):
    """
    Detect available source numbers in an Open Ephys recording directory.
    
    Parameters
    ----------
    directory : str or Path
        Path to the directory containing Open Ephys data
        
    Returns
    -------
    sources : list
        List of detected source numbers (as strings)
        
    Examples
    --------
    >>> sources = detect_sources('/path/to/recording')
    >>> print(f"Available sources: {sources}")
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Get all .continuous files
    continuous_files = list(directory.glob("*.continuous"))
    
    # Extract unique source numbers
    sources = set()
    for file in continuous_files:
        # Format is typically: 100_CH1.continuous or 100_CH1_2.continuous (with session)
        parts = file.stem.split('_')
        if len(parts) >= 2:
            source_num = parts[0]
            if source_num.isdigit():
                sources.add(source_num)
    
    return sorted(list(sources))

# --- Format Detection Utilities ---

def detect_format(directory):
    """
    Detect the Open Ephys data format in the given directory.
    
    Parameters
    ----------
    directory : str or Path
        Path to the directory containing Open Ephys data, or path to a 
        .continuous file (in which case the parent directory will be used)
        
    Returns
    -------
    format_type : str
        One of DataFormat.OLD, DataFormat.NEW, or DataFormat.UNKNOWN
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Path does not exist: {directory}")
    
    # If a file was passed instead of a directory, use its parent directory
    if directory.is_file():
        print(f"Note: File path provided. Using parent directory: {directory.parent}")
        directory = directory.parent
    
    # Check for old format first (.continuous, .spikes, .events files)
    # Old format is characterized by these binary files
    continuous_files = list(directory.glob("*.continuous"))
    spikes_files = list(directory.glob("*.spikes"))
    events_files = list(directory.glob("*.events"))
    
    # If we find .continuous files, it's old format (even if .openephys files exist)
    if continuous_files or spikes_files or events_files:
        return DataFormat.OLD
    
    # Check for new format (structure.openephys files without .continuous files)
    structure_files = list(directory.glob("structure*.openephys")) + \
                     list(directory.glob("Continuous_Data*.openephys"))
    
    if structure_files:
        return DataFormat.NEW
    
    return DataFormat.UNKNOWN


def load_open_ephys_data(directory, format_type=None, target_sample_rate=None,
                         channel_groups=None, **kwargs):
    """
    Load Open Ephys data from the specified directory.
    
    This function automatically detects the data format and uses the appropriate
    loader. It returns a unified data structure for easy access.
    
    Parameters
    ----------
    directory : str or Path
        Path to the directory containing Open Ephys data, or path to a 
        .continuous file (in which case the parent directory will be used)
    format_type : str, optional
        Force a specific format type (DataFormat.OLD or DataFormat.NEW).
        If None, the format will be auto-detected.
    target_sample_rate : float, optional
        Target sample rate in Hz. If provided, data will be downsampled from
        the original sample rate to this rate using anti-aliasing filtering.
        For example, to downsample 30 kHz data to 1 kHz, use target_sample_rate=1000.
        This significantly reduces memory usage and speeds up processing.
    **kwargs : dict
        Additional arguments passed to the specific loader:
        
        For old format:
            - channels : list of channel numbers to load (default: 'all')
            - dtype : data type for loading (default: float)
            - chprefix : channel prefix (default: 'CH')
            
        For new format:
            - experiment_index : index of experiment to load (default: 0)
            - recording_index : index of recording to load (default: 0)
            
    Returns
    -------
    data : dict or OpenEphysRecording object
        Loaded data. The structure depends on the format:
        
        For old format:
            Dictionary with channel names as keys, each containing:
            - 'data': numpy array of samples
            - 'timestamps': timestamp information
            - 'header': metadata
            
        For new format:
            OpenEphysRecording object with methods:
            - load_continuous(): load continuous data
            - load_spikes(): load spike data
            - load_events(): load event data
            
    Examples
    --------
    >>> # Auto-detect and load data
    >>> data = load_open_ephys_data('/path/to/recording')
    >>>
    >>> # Force old format and load specific channels
    >>> data = load_open_ephys_data('/path/to/recording', 
    ...                            format_type=DataFormat.OLD,
    ...                            channels=[1, 2, 3, 4])
    >>>
    >>> # Load new format with specific recording
    >>> recording = load_open_ephys_data('/path/to/recording',
    ...                                   format_type=DataFormat.NEW,
    ...                                   recording_index=1)
    >>> recording.load_continuous()
    """
    directory = Path(directory).resolve()
    
    # If a file was passed instead of a directory, use its parent directory
    if directory.is_file():
        parent_dir = directory.parent
        print(f"Note: Individual file provided. Using parent directory: {parent_dir}")
        print(f"      Make sure all .continuous files are in this directory!")
        directory = parent_dir
        
        # Warn if parent is a drive root
        if str(directory).endswith(':\\') or str(directory) == directory.anchor:
            print(f"WARNING: Parent directory is drive root ({directory})")
            print(f"         This is unusual. Please provide the full path to the recording folder.")
            print(f"         Example: r'I:\\my_recordings\\recording_2024-01-01'")
    
    # Convert to string for compatibility with loaders
    directory = str(directory)
    
    # Detect format if not specified
    if format_type is None:
        format_type = detect_format(directory)
        print(f"Detected format: {format_type}")
    
    if format_type == DataFormat.UNKNOWN:
        raise ValueError(
            f"Could not detect Open Ephys data format in directory: {directory}\n"
            "Expected to find either:\n"
            "  - structure*.openephys files (new format)\n"
            "  - *.continuous files (old format)"
        )
    
    # Handle channel groups if specified
    if channel_groups is not None:
        data = _load_channel_groups(directory, format_type, channel_groups, **kwargs)

        # Apply downsampling if requested
        if target_sample_rate is not None:
            data = _downsample_data(data, target_sample_rate)

        # --- Loader Functions ---
        return data

    # Load using appropriate loader
    if format_type == DataFormat.OLD:
        data = _load_old_format(directory, **kwargs)

        # Apply downsampling if requested
        if target_sample_rate is not None:
            data = _downsample_data(data, target_sample_rate)

        return data

    elif format_type == DataFormat.NEW:
        data = _load_new_format(directory, **kwargs)

        # Apply downsampling if requested
        if target_sample_rate is not None:
            data = _downsample_data(data, target_sample_rate)

        return data
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def _downsample_data(data, target_sample_rate):
    """
    Downsample loaded data to target sample rate using anti-aliasing filter.
    
    Parameters
    ----------
    data : dict
        Loaded data dictionary
    target_sample_rate : float
        Target sample rate in Hz
        
    Returns
    -------
    data : dict
        Downsampled data
    """
    from scipy import signal
    
    # Get original sample rate
    original_rate = get_sample_rate(data)
    
    if target_sample_rate >= original_rate:
        print(f"Target sample rate ({target_sample_rate} Hz) >= original rate ({original_rate} Hz). No downsampling needed.")
        return data
    
    # Calculate downsampling factor
    downsample_factor = int(original_rate / target_sample_rate)
    actual_target_rate = original_rate / downsample_factor
    
    print(f"Downsampling from {original_rate} Hz to {actual_target_rate} Hz (factor: {downsample_factor})")
    
    # Downsample each channel
    for channel_name in data.keys():
        if 'data' in data[channel_name]:
            original_data = data[channel_name]['data']
            
            # Use decimate for efficient downsampling with anti-aliasing
            # decimate applies a lowpass filter before downsampling
            downsampled = signal.decimate(original_data, downsample_factor, ftype='iir', zero_phase=True)
            
            data[channel_name]['data'] = downsampled
            
            # Update sample rate in header if it exists
            if 'header' in data[channel_name]:
                data[channel_name]['header']['sampleRate'] = actual_target_rate
                
    print(f"✓ Downsampling complete. Memory reduced by ~{downsample_factor}x")
    
    return data


def _load_old_format(directory, channels='all', dtype=float, chprefix='CH', 
                     session='0', source='100', as_array=False):
    """
    Load data using the old format loader.
    
    # --- Data Transformation Utilities ---
    Parameters
    ----------
    directory : str
        Path to directory
    channels : str or list
        'all' or list of channel numbers
    dtype : type
        Data type for loading
    chprefix : str
        Channel prefix (e.g., 'CH', 'ADC')
    session : str
        Session identifier
    source : str
        Source identifier
    as_array : bool
        If True, return data as a single numpy array instead of dict
        
    Returns
    -------
    data : dict or numpy.ndarray
        Loaded data
    """
    if not OLD_FORMAT_AVAILABLE:
        raise ImportError(
            "Old format loader is not available. "
            "Check that oe_old_analysis_tools/Python3/OpenEphys.py exists."
        )
    
    print(f"Loading old format data from: {directory}")
    
    try:
        if as_array:
            # Load as array for easier processing
            data = OldFormatLoader.loadFolderToArray(
                directory, 
                channels=channels,
                chprefix=chprefix,
                dtype=dtype,
                session=session,
                source=source
            )
        else:
            # Load as dictionary
            if channels == 'all':
                data = OldFormatLoader.loadFolder(directory, dtype=dtype)
            else:
                # Workaround: Old loader has bug where source parameter is ignored
                # when loading specific channels. Load them individually instead.
                data = {}
                for ch_num in channels:
                    # Construct filename based on parameters
                    if session == '0':
                        filename = f"{source}_{chprefix}{ch_num}.continuous"
                    else:
                        filename = f"{source}_{chprefix}{ch_num}_{session}.continuous"
                    
                    filepath = os.path.join(directory, filename)
                    
                    # Check if file exists before trying to load
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(
                            f"File not found: {filename}\n"
                            f"Looking in: {directory}\n\n"
                            f"Make sure:\n"
                            f"1. The source number is correct (you specified: {source})\n"
                            f"2. The channel prefix is correct (you specified: {chprefix})\n"
                            f"3. The session is correct (you specified: {session})\n"
                            f"4. The files exist in the directory"
                        )
                    
                    # Load the file
                    channel_key = f"{source}_{chprefix}{ch_num}" if session == '0' else f"{source}_{chprefix}{ch_num}_{session}"
                    data[channel_key] = OldFormatLoader.loadContinuous(filepath, dtype=dtype)
        
        return data
        
    except Exception as e:
        error_msg = str(e)
        if "may be corrupt" in error_msg or "File size is not consistent" in error_msg:
            # List all .continuous files to help debug
            from pathlib import Path
            continuous_files = list(Path(directory).glob("*.continuous"))
            
            raise RuntimeError(
                f"Error loading Open Ephys data: {error_msg}\n\n"
                f"This usually means one or more .continuous files are corrupt or incomplete.\n"
                f"Found {len(continuous_files)} .continuous files in directory.\n\n"
                f"Troubleshooting steps:\n"
                f"1. Check if the recording was interrupted or stopped improperly\n"
                f"2. Verify all .continuous files have reasonable file sizes\n"
                f"3. Try loading individual channels to identify the problematic file:\n"
                f"   data = load_open_ephys_data(r'{directory}', channels=[1])\n"
                f"4. Check the Open Ephys GUI logs for any recording errors\n"
            ) from e
        else:
            # Re-raise other exceptions as-is
            raise


def _load_new_format(directory, experiment_index=0, recording_index=0):
    """
    Load data using the new format loader.
    
    Parameters
    ----------
    directory : str
        Path to directory
    experiment_index : int
        Index of experiment to load
    recording_index : int
        Index of recording to load
        
    Returns
    -------
    recording : OpenEphysRecording
        Recording object with methods to load different data types
    """
    if not NEW_FORMAT_AVAILABLE:
        raise ImportError(
            "New format loader is not available. "
            "Check that oe_python_tools/src/open_ephys/analysis/ exists."
        )
    
    print(f"Loading new format data from: {directory}")
    
    # Create recording object
    recording = OpenEphysRecording(
        directory, 
        experiment_index=experiment_index,
        recording_index=recording_index
    )
    
    return recording


def get_continuous_data(data, format_type=None):
    """
    Extract continuous data from loaded Open Ephys data.
    
    This helper function provides a unified way to access continuous data
    regardless of the original format.
    
    Parameters
    ----------
    data : dict or OpenEphysRecording
        Data loaded by load_open_ephys_data()
    format_type : str, optional
        Format type if known. Will be inferred if not provided.
        
    Returns
    -------
    continuous_data : numpy.ndarray or list
        Continuous data as array(s)
    """
    # If data is a dict (old format), return the data arrays
    if isinstance(data, dict):
        return {key: val['data'] for key, val in data.items() if 'data' in val}
    
    # If data is an OpenEphysRecording object (new format)
    elif NEW_FORMAT_AVAILABLE and isinstance(data, OpenEphysRecording):
        if not hasattr(data, '_continuous') or data._continuous is None:
            data.load_continuous()
        return data.continuous
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _load_channel_groups(directory, format_type, channel_groups, **kwargs):
    """
    Load data organized by brain regions/groups.

    Parameters
    ----------
    directory : str
        Path to directory
    format_type : str
        Data format type
    channel_groups : dict, optional
        Dictionary where keys are region names and values are lists of channel numbers.
        If provided, data will be organized by brain regions instead of individual channels.
        Example: {'CA1': [1, 2, 3, 4], 'CA3': [16, 17, 18, 19], 'PFC': [28, 29, 30, 31]}
        When using channel_groups, the returned data structure will be:
        {
            'CA1': {'100_CH1': {...}, '100_CH2': {...}, ...},
            'CA3': {'100_CH16': {...}, '100_CH17': {...}, ...},
            ...
        }
    **kwargs : dict
        Additional arguments passed to the loader

    Returns
    -------
    data : dict or OpenEphysRecording
        Organized data with region structure for old format, or recording object for new format
    """
    print("Loading data organized by brain regions...")

    # Flatten all channels from all groups
    all_channels = []
    for region, channels in channel_groups.items():
        all_channels.extend(channels)

    # Remove duplicates while preserving order
    all_channels = list(dict.fromkeys(all_channels))

    print(f"Loading {len(all_channels)} channels from {len(channel_groups)} regions")

    # Load all channels
    if format_type == DataFormat.OLD:
        all_data = _load_old_format(directory, channels=all_channels, **kwargs)

        # Organize data by regions
        organized_data = {}
        for region, region_channels in channel_groups.items():
            organized_data[region] = {}

            for ch_num in region_channels:
                # Find the channel key in the loaded data
                for channel_key in all_data.keys():
                    if f"CH{ch_num}" in channel_key or f"CH{ch_num}_" in channel_key:
                        organized_data[region][channel_key] = all_data[channel_key]
                        break

        print(f"✓ Organized data into {len(organized_data)} regions")
        for region, region_data in organized_data.items():
            print(f"  {region}: {len(region_data)} channels")

        return organized_data

    elif format_type == DataFormat.NEW:
        # For new format, load the recording and add region information
        recording = _load_new_format(directory, **kwargs)

        # Add channel groups as metadata to the recording object
        recording.channel_groups = channel_groups
        recording.region_channels = all_channels

        print(f"✓ New format recording loaded with {len(channel_groups)} brain regions")
        for region, region_data in channel_groups.items():
            print(f"  {region}: {len(region_data)} channels")

        return recording
    else:
        raise ValueError(f"Unsupported format for channel groups: {format_type}")


def get_sample_rate(data):
    """
    Extract sample rate from loaded data.
    
    Parameters
    ----------
    data : dict or OpenEphysRecording
        Data loaded by load_open_ephys_data()
        
    Returns
    -------
    sample_rate : float
        Sample rate in Hz
    """
    if isinstance(data, dict):
        # Old format - get from first channel's header
        first_channel = next(iter(data.values()))
        if 'header' in first_channel:
            return float(first_channel['header'].get('sampleRate', 30000.0))
        return 30000.0  # Default
    
    elif NEW_FORMAT_AVAILABLE and isinstance(data, OpenEphysRecording):
        if not hasattr(data, '_continuous') or data._continuous is None:
            data.load_continuous()
        if data.continuous:
            return data.continuous[0].metadata['sample_rate']
        return 30000.0  # Default
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
