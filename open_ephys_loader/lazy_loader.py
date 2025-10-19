"""
lazy_loader.py - Memory-efficient lazy loading for Open Ephys .dat files

This module provides a LazyBinaryLoader class that enables memory-efficient access
to Open Ephys .dat files using memory mapping similar to SpikeInterface's approach.
Supports tetrode and brain region organization for neuroscience workflows.
"""

import os
import numpy as np
import mmap
from scipy.signal import decimate


class LazyBinaryLoader:
    """
    Memory-efficient loader for Open Ephys .dat files with tetrode organization.

    This class provides lazy loading of neural data similar to SpikeInterface,
    but with brain region and tetrode organization for neuroscience workflows.

    Parameters
    ----------
    filepath : str
        Path to the .dat file
    num_channels : int
        Total number of channels in the file
    sampling_frequency : float
        Sampling frequency in Hz (default: 30000)
    dtype : str
        Data type (default: 'int16')
    file_offset : int
        Bytes to skip at start of file (default: 0)
    time_axis : int
        Time axis orientation (0: samples×channels, 1: channels×samples) (default: 0)
    tetrode_groups : dict, optional
        Organization by brain regions and tetrodes.
        Format: {'region': {'tetX': [channel_list]}}
    selected_channels : dict, optional
        Single channel per tetrode for analysis.
        Format: {'region_tetX': channel_number}
    """

    def __init__(self, filepath, num_channels, sampling_frequency=30000.0,
                 dtype='int16', file_offset=0, time_axis=0,
                 tetrode_groups=None, selected_channels=None):

        self.filepath = filepath
        self.num_channels = num_channels
        self.sampling_frequency = sampling_frequency
        self.dtype = dtype
        self.file_offset = file_offset
        self.time_axis = time_axis

        # File properties
        self.dtype_np = np.dtype(dtype)
        self.bytes_per_sample = num_channels * self.dtype_np.itemsize
        self.file_size = os.path.getsize(filepath)
        self.file_size_bytes = self.file_size - file_offset
        self.num_samples = self.file_size_bytes // self.bytes_per_sample

        # Duration
        self.duration = self.num_samples / sampling_frequency

        # Tetrode organization
        self.tetrode_groups = tetrode_groups or {}
        self.selected_channels = selected_channels or {}

        # Derive convenient access properties
        self.regions = list(self.tetrode_groups.keys())
        self.tetrodes = {}
        self.all_tetrode_channels = []
        self.group_channels = {}

        # Build access dictionaries
        for region, tetrodes in self.tetrode_groups.items():
            self.group_channels[region] = []
            for tetrode_name, channels in tetrodes.items():
                full_name = f"{region}_{tetrode_name}"
                self.tetrodes[full_name] = channels
                self.all_tetrode_channels.extend(channels)
                self.group_channels[region].extend(channels)

        # Remove duplicates while preserving order
        self.all_tetrode_channels = list(dict.fromkeys(self.all_tetrode_channels))

        # Validate selected channels
        self._validate_selected_channels()

        # Initialize memory mapping (lazy - only when needed)
        self._mmap_obj = None
        self._file_obj = None

        if __debug__:
            print(f"LazyBinaryLoader initialized:")
            print(f"  File: {filepath}")
            print(f"  Channels: {num_channels}, Samples: {self.num_samples:,}")
            print(f"  Duration: {self.duration:.2f}s")
            print(f"  Regions: {self.regions}")
            print(f"  Tetrodes: {list(self.tetrodes.keys())}")
            print(f"  Selected channels: {len(self.selected_channels)}")

    def _validate_selected_channels(self):
        """Validate that selected channels exist in tetrode groups."""
        for tetrode_name, channel in self.selected_channels.items():
            if tetrode_name not in self.tetrodes:
                raise ValueError(f"Tetrode '{tetrode_name}' not found in tetrode_groups")
            if channel not in self.tetrodes[tetrode_name]:
                raise ValueError(f"Channel {channel} not found in tetrode '{tetrode_name}'")

    def _ensure_mmapping(self):
        """Ensure memory mapping is initialized."""
        if self._mmap_obj is None:
            self._file_obj = open(self.filepath, 'rb')

            # Align memory mapping offset
            memmap_offset, start_offset = divmod(self.file_offset, mmap.ALLOCATIONGRANULARITY)
            memmap_offset *= mmap.ALLOCATIONGRANULARITY
            memmap_length = self.file_size_bytes + start_offset

            self._memmap_offset = memmap_offset
            self._memmap_start_offset = start_offset

            # Create mmap object
            self._mmap_obj = mmap.mmap(
                self._file_obj.fileno(),
                length=memmap_length,
                access=mmap.ACCESS_READ,
                offset=memmap_offset
            )

    def _get_raw_data(self, start_idx=None, end_idx=None, channels=None):
        """
        Get raw data for specified time window and channels - OPTIMIZED VERSION.

        Parameters
        ----------
        start_idx : int, optional
            Start sample index
        end_idx : int, optional
            End sample index
        channels : list, optional
            List of channel indices

        Returns
        -------
        numpy.ndarray
            Data array of shape (n_channels, n_samples)
        """
        self._ensure_mmapping()

        # Handle defaults
        start_idx = start_idx or 0
        end_idx = end_idx or self.num_samples
        channels = channels or list(range(self.num_channels))

        n_channels = len(channels)
        n_samples = end_idx - start_idx

        # Validate inputs
        for ch_idx in channels:
            if ch_idx >= self.num_channels or ch_idx < 0:
                raise ValueError(f"Channel index {ch_idx} out of range [0, {self.num_channels})")

        # For time_axis=0: data is stored as [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ...]
        # We need to read all samples for selected channels efficiently

        # Pre-allocate output array
        data = np.zeros((n_channels, n_samples), dtype=self.dtype_np)

        # Process each sample across all selected channels (vectorized approach)
        for sample_idx in range(n_samples):
            global_sample_idx = start_idx + sample_idx

            # Read all selected channels for this sample in one operation
            for ch_list_idx, ch_idx in enumerate(channels):
                # Calculate byte offset for this channel and sample
                # Format: [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ...]
                offset = self._memmap_start_offset + (global_sample_idx * self.num_channels + ch_idx) * self.dtype_np.itemsize

                # Read single sample for this channel
                try:
                    sample_value = np.ndarray(
                        shape=(1,),
                        dtype=self.dtype_np,
                        buffer=self._mmap_obj,
                        offset=offset
                    )[0]
                    data[ch_list_idx, sample_idx] = sample_value
                except (ValueError, IndexError) as e:
                    if offset + self.dtype_np.itemsize > len(self._mmap_obj or b''):
                        raise IndexError(f"File offset {offset} exceeds file size for channel {ch_idx}, sample {sample_idx}")
                    else:
                        raise e

        return data

    def get_trace(self, channel, start_time=0, end_time=None, target_fs=None):
        """
        Get single channel trace for time window.

        Parameters
        ----------
        channel : int
            Channel number (0-based)
        start_time : float
            Start time in seconds
        end_time : float, optional
            End time in seconds (default: end of recording)
        target_fs : float, optional
            Target sampling frequency for downsampling

        Returns
        -------
        numpy.ndarray
            Channel data
        """
        start_idx = int(start_time * self.sampling_frequency)
        end_idx = int(end_time * self.sampling_frequency) if end_time else self.num_samples

        data = self._get_raw_data(start_idx, end_idx, [channel])

        trace = data[0, :]

        # Downsample if requested
        if target_fs is not None and target_fs < self.sampling_frequency:
            factor = int(self.sampling_frequency / target_fs)
            trace = decimate(trace, factor, ftype='iir', zero_phase=True)

        return trace

    def get_selected_trace(self, tetrode_name, start_time=0, end_time=None, target_fs=None):
        """
        Get the selected channel from a tetrode.

        Parameters
        ----------
        tetrode_name : str
            Name of tetrode (e.g., 'CA1_tet1')
        start_time : float
            Start time in seconds
        end_time : float, optional
            End time in seconds
        target_fs : float, optional
            Target sampling frequency

        Returns
        -------
        numpy.ndarray
            Trace data
        """
        if tetrode_name not in self.selected_channels:
            available = list(self.selected_channels.keys())
            raise ValueError(f"Tetrode '{tetrode_name}' not in selected channels. Available: {available}")

        channel = self.selected_channels[tetrode_name]
        return self.get_trace(channel, start_time, end_time, target_fs)

    def get_all_selected(self, start_time=0, end_time=None, target_fs=None):
        """
        Get all selected channels for analysis.

        Returns
        -------
        dict
            Dictionary with tetrode names as keys and traces as values
        """
        traces = {}
        for tetrode_name, channel in self.selected_channels.items():
            traces[tetrode_name] = self.get_selected_trace(
                tetrode_name, start_time, end_time, target_fs
            )
        return traces

    def get_tetrode(self, tetrode_name, start_time=0, end_time=None, target_fs=None):
        """
        Get all channels from a tetrode.

        Parameters
        ----------
        tetrode_name : str
            Name of tetrode (e.g., 'CA1_tet1')
        start_time : float
            Start time in seconds
        end_time : float, optional
            End time in seconds
        target_fs : float, optional
            Target sampling frequency

        Returns
        -------
        numpy.ndarray
            Data of shape (n_channels_in_tetrode, n_samples)
        """
        if tetrode_name not in self.tetrodes:
            available = list(self.tetrodes.keys())
            raise ValueError(f"Tetrode '{tetrode_name}' not found. Available: {available}")

        channels = self.tetrodes[tetrode_name]
        start_idx = int(start_time * self.sampling_frequency)
        end_idx = int(end_time * self.sampling_frequency) if end_time else self.num_samples

        data = self._get_raw_data(start_idx, end_idx, channels)

        # Downsample if requested
        if target_fs is not None and target_fs < self.sampling_frequency:
            factor = int(self.sampling_frequency / target_fs)
            data = np.array([decimate(ch_data, factor, ftype='iir', zero_phase=True)
                           for ch_data in data])

        return data

    def get_group(self, region_name, start_time=0, end_time=None, target_fs=None, mode='selected'):
        """
        Get all channels from a brain region.

        Parameters
        ----------
        region_name : str
            Brain region name (e.g., 'CA1')
        start_time : float
            Start time in seconds
        end_time : float, optional
            End time in seconds
        target_fs : float, optional
            Target sampling frequency
        mode : str
            'selected' for one channel per tetrode, 'all' for all channels

        Returns
        -------
        dict or numpy.ndarray
            Selected traces or all channel data
        """
        if region_name not in self.group_channels:
            available = list(self.group_channels.keys())
            raise ValueError(f"Region '{region_name}' not found. Available: {available}")

        if mode == 'selected':
            # Get selected channels from region's tetrodes
            traces = {}
            for tetrode_name, channel in self.selected_channels.items():
                if tetrode_name.startswith(f"{region_name}_"):
                    traces[tetrode_name] = self.get_selected_trace(
                        tetrode_name, start_time, end_time, target_fs
                    )
            return traces

        elif mode == 'all':
            # Get all channels from region
            channels = self.group_channels[region_name]
            start_idx = int(start_time * self.sampling_frequency)
            end_idx = int(end_time * self.sampling_frequency) if end_time else self.num_samples

            data = self._get_raw_data(start_idx, end_idx, channels)

            # Downsample if requested
            if target_fs is not None and target_fs < self.sampling_frequency:
                factor = int(self.sampling_frequency / target_fs)
                data = np.array([decimate(ch_data, factor, ftype='iir', zero_phase=True)
                               for ch_data in data])

            return data

        else:
            raise ValueError(f"Mode must be 'selected' or 'all', got '{mode}'")

    def close(self):
        """Close memory mapping and file handles."""
        if self._mmap_obj:
            self._mmap_obj.close()
            self._mmap_obj = None
        if self._file_obj:
            self._file_obj.close()
            self._file_obj = None

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.close()
