"""
load_openephys_dat_lazy.py - Lazy loading version of load_openephys_dat

This function replicates the interface and behavior of the original load_openephys_dat
function but with memory-efficient lazy loading.
"""

from .lazy_loader import LazyBinaryLoader
from .visualization import dat_file_viewer
import numpy as np


# def load_openephys_dat_lazy_with_tetrodes(filepath, num_channels, tetrode_groups, selected_channels,
#                                          sampling_frequency=30000, target_sampling_frequency=1000,
#                                          plot='widget', verbose=True, time_axis=0):
#     """
#     Lazy loading version that preserves your tetrode organization with brain regions.

#     This function accepts your tetrode_groups and selected_channels directly,
#     maintaining the brain region and tetrode structure while using memory-efficient
#     lazy loading to handle large .dat files.

#     Parameters
#     ----------
#     filepath : str
#         Path to the .dat file
#     num_channels : int
#         Total number of channels in the file
#     tetrode_groups : dict
#         Tetrode organization by brain regions: {'region': {'tetX': [channels]}}
#     selected_channels : dict
#         Selected channel per tetrode: {'region_tetX': channel_number}
#     sampling_frequency : float
#         Original sampling frequency in Hz (default: 30000)
#     target_sampling_frequency : float
#         Target sampling frequency for visualization in Hz (default: 1000)
#     plot : str or None
#         Plot mode: 'widget' for interactive widget, None for loader only
#     verbose : bool
#         Print information during loading (default: True)
#     time_axis : int
#         Data orientation (default: 0)

#     Returns
#     -------
#     widget or LazyBinaryLoader
#         Interactive widget or LazyBinaryLoader instance

#     Examples
#     --------
#     # Your tetrode organization
#     tetrode_groups = {
#         'CA1': {'tet1': [17, 18, 19, 20], 'tet2': [21, 22, 23, 24]},
#         'RTC': {'tet1': [14, 15, 16]},
#         'PFC': {'tet1': [0, 1, 2, 3], 'tet2': [4, 5, 6, 7]},
#     }

#     selected_channels = {
#         'CA1_tet1': 17, 'CA1_tet2': 21, 'RTC_tet1': 14,
#         'PFC_tet1': 0, 'PFC_tet2': 5
#     }

#     widget = load_openephys_dat_lazy_with_tetrodes(
#         filepath=r"I:\Spikeinterface_practice\s4_rec\ephys.dat",
#         num_channels=43,
#         tetrode_groups=tetrode_groups,
#         selected_channels=selected_channels,
#         sampling_frequency=30000,
#         target_sampling_frequency=1000,
#         plot='widget',
#         verbose=True
#     )
#     display(widget)
#     """

#     # VALIDATE the user's tetrode configuration
#     if verbose:
#         print(f"Loading {filepath} with tetrode organization...")
#         print(f"  Channels: {num_channels} total")
#         print(f"  Tetrodes: {sum(len(tets) for tets in tetrode_groups.values())}")
#         print(f"  Selected channels: {len(selected_channels)}")
#         print(f"  Original sampling rate: {sampling_frequency} Hz")
#         print(f"  Target sampling rate: {target_sampling_frequency or sampling_frequency} Hz")

#         print("\nTetrode Groups:")
#         for region, tetrodes in tetrode_groups.items():
#             print(f"  {region}:")
#             for tet_name, channels in tetrodes.items():
#                 selected_ch = selected_channels.get(f"{region}_{tet_name}", "None")
#                 print(f"    {tet_name}: {channels} (selected: {selected_ch})")

#     # VALIDATE configuration
#     expected_keys = [f"{region}_{tet_name}" for region, tets in tetrode_groups.items()
#                      for tet_name in tets.keys()]
#     for tetrode_name in expected_keys:
#         if tetrode_name not in selected_channels:
#             raise ValueError(f"Tetrode '{tetrode_name}' not found in selected_channels")

#     for tetrode_name, selected_ch in selected_channels.items():
#         if tetrode_name not in expected_keys:
#             raise ValueError(f"Selected channel '{tetrode_name}' not found in tetrode_groups")

#         region, tet = tetrode_name.split('_', 1)
#         if selected_ch not in tetrode_groups[region][tet]:
#             raise ValueError(f"Channel {selected_ch} not in tetrode {tetrode_name}")

#     # USE user's tetrode_groups and selected_channels directly (do NOT overwrite!)
#     # Create LazyBinaryLoader
#     loader = LazyBinaryLoader(
#         filepath=filepath,
#         num_channels=num_channels,
#         sampling_frequency=sampling_frequency,
#         dtype='int16',
#         file_offset=0,
#         time_axis=time_axis,
#         tetrode_groups=tetrode_groups,    # ← User's tetrode organization
#         selected_channels=selected_channels  # ← User's selected channels
#     )

#     if verbose:
#         print("LazyBinaryLoader initialized successfully")
#         print(f"  Duration: {loader.duration:.2f}s")
#         print(f"  Samples: {loader.num_samples:,}")

#     # Return plot or loader
#     if plot == 'widget':
#         if verbose:
#             print("Creating interactive visualization widget...")

#         # Determine if we should downsample for visualization
#         downsample_for_vis = (target_sampling_frequency is not None and
#                             target_sampling_frequency < sampling_frequency)

#         widget = dat_file_viewer(
#             loader,
#             channels=list(selected_channels.keys()),  # ← User's tetrode names
#             initial_window=10,
#             initial_start=0,
#             downsample_for_vis=downsample_for_vis
#         )

#         if verbose:
#             print(f"Widget created with {len(selected_channels)} tetrodes")
#             print("Tetrode mapping:")
#             for tetrode_name, channel in selected_channels.items():
#                 region = tetrode_name.split('_')[0]
#                 print(f"  {tetrode_name} ({region}): Channel {channel}")

#         # Attach loader attributes to the widget for easy access
#         widget.loader = loader
#         widget.duration = loader.duration
#         widget.num_samples = loader.num_samples
#         widget.sampling_frequency = loader.sampling_frequency
#         widget.num_channels = loader.num_channels
#         widget.selected_channels = loader.selected_channels
#         widget.tetrode_groups = loader.tetrode_groups

#         return widget

#     else:
#         return loader

def load_openephys_dat_lazy_with_tetrodes(filepath, num_channels, tetrode_groups, selected_channels,
                                         sampling_frequency=30000, target_sampling_frequency=1000,
                                         plot='widget', verbose=True, time_axis=0):
    """
    Lazy loading version that preserves your tetrode organization with brain regions.

    This function accepts your tetrode_groups and selected_channels directly,
    maintaining the brain region and tetrode structure while using memory-efficient
    lazy loading to handle large .dat files.

    Parameters
    ----------
    filepath : str
        Path to the .dat file
    num_channels : int
        Total number of channels in the file
    tetrode_groups : dict
        Tetrode organization by brain regions: {'region': {'tetX': [channels]}}
    selected_channels : dict
        Selected channel per tetrode: {'region_tetX': channel_number}
    sampling_frequency : float
        Original sampling frequency in Hz (default: 30000)
    target_sampling_frequency : float
        Target sampling frequency for visualization in Hz (default: 1000)
    plot : str or None
        Plot mode: 'widget' for interactive widget, None for loader only
    verbose : bool
        Print information during loading (default: True)
    time_axis : int
        Data orientation (default: 0)

    Returns
    -------
    widget or LazyBinaryLoader
        Interactive widget or LazyBinaryLoader instance

    Notes
    -----
    When a loader is returned (plot != 'widget'), you may request downsampled traces
    via `loader.get_selected_trace(name, ..., downsample=True)` or
    `loader.get_selected_trace(name, ..., resample_to=<Hz>)`.

    Examples
    --------
    # Your tetrode organization
    tetrode_groups = {
        'CA1': {'tet1': [17, 18, 19, 20], 'tet2': [21, 22, 23, 24]},
        'RTC': {'tet1': [14, 15, 16]},
        'PFC': {'tet1': [0, 1, 2, 3], 'tet2': [4, 5, 6, 7]},
    }

    selected_channels = {
        'CA1_tet1': 17, 'CA1_tet2': 21, 'RTC_tet1': 14,
        'PFC_tet1': 0, 'PFC_tet2': 5
    }

    widget = load_openephys_dat_lazy_with_tetrodes(
        filepath=r"I:\Spikeinterface_practice\s4_rec\ephys.dat",
        num_channels=43,
        tetrode_groups=tetrode_groups,
        selected_channels=selected_channels,
        sampling_frequency=30000,
        target_sampling_frequency=1000,
        plot='widget',
        verbose=True
    )
    display(widget)
    """

    # VALIDATE the user's tetrode configuration
    if verbose:
        print(f"Loading {filepath} with tetrode organization...")
        print(f"  Channels: {num_channels} total")
        print(f"  Tetrodes: {sum(len(tets) for tets in tetrode_groups.values())}")
        print(f"  Selected channels: {len(selected_channels)}")
        print(f"  Original sampling rate: {sampling_frequency} Hz")
        print(f"  Target sampling rate: {target_sampling_frequency or sampling_frequency} Hz")

        print("\nTetrode Groups:")
        for region, tetrodes in tetrode_groups.items():
            print(f"  {region}:")
            for tet_name, channels in tetrodes.items():
                selected_ch = selected_channels.get(f"{region}_{tet_name}", "None")
                print(f"    {tet_name}: {channels} (selected: {selected_ch})")

    # VALIDATE configuration
    expected_keys = [f"{region}_{tet_name}" for region, tets in tetrode_groups.items()
                     for tet_name in tets.keys()]
    for tetrode_name in expected_keys:
        if tetrode_name not in selected_channels:
            raise ValueError(f"Tetrode '{tetrode_name}' not found in selected_channels")

    for tetrode_name, selected_ch in selected_channels.items():
        if tetrode_name not in expected_keys:
            raise ValueError(f"Selected channel '{tetrode_name}' not found in tetrode_groups")

        region, tet = tetrode_name.split('_', 1)
        if selected_ch not in tetrode_groups[region][tet]:
            raise ValueError(f"Channel {selected_ch} not in tetrode {tetrode_name}")

    # USE user's tetrode_groups and selected_channels directly (do NOT overwrite!)
    # Create LazyBinaryLoader
    loader = LazyBinaryLoader(
        filepath=filepath,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        dtype='int16',
        file_offset=0,
        time_axis=time_axis,
        tetrode_groups=tetrode_groups,    # ← User's tetrode organization
        selected_channels=selected_channels  # ← User's selected channels
    )

    # Attach downsampling-enabled accessor when appropriate
    if target_sampling_frequency is not None and target_sampling_frequency < sampling_frequency:
        loader.target_sampling_frequency = target_sampling_frequency
        _raw_get = loader.get_selected_trace

        from types import MethodType

        def _get_selected_trace_with_downsampling(self, tetrode_name, *args,
                                                  downsample=False, resample_to=None, **kwargs):
            trace = np.asarray(_raw_get(tetrode_name, *args, **kwargs))
            desired_fs = self.sampling_frequency

            if resample_to is not None:
                desired_fs = float(resample_to)
            elif downsample and hasattr(self, "target_sampling_frequency"):
                desired_fs = float(self.target_sampling_frequency)

            if desired_fs <= 0:
                raise ValueError("Desired sampling frequency must be positive.")

            if np.isclose(desired_fs, self.sampling_frequency):
                return trace

            if desired_fs > self.sampling_frequency:
                raise ValueError("Upsampling is not supported by this helper.")

            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise ImportError("scipy is required for downsampling; install scipy>=1.8.") from exc

            from fractions import Fraction
            src_fs = float(self.sampling_frequency)
            frac = Fraction.from_float(desired_fs / src_fs).limit_denominator(1024)

            return resample_poly(trace.astype(np.float32, copy=False),
                                 frac.numerator, frac.denominator)

        loader.get_selected_trace = MethodType(_get_selected_trace_with_downsampling, loader)

    if verbose:
        print("LazyBinaryLoader initialized successfully")
        print(f"  Duration: {loader.duration:.2f}s")
        print(f"  Samples: {loader.num_samples:,}")

    # Return plot or loader
    if plot == 'widget':
        if verbose:
            print("Creating interactive visualization widget...")

        # Determine if we should downsample for visualization
        downsample_for_vis = (target_sampling_frequency is not None and
                              target_sampling_frequency < sampling_frequency)

        widget = dat_file_viewer(
            loader,
            channels=list(selected_channels.keys()),  # ← User's tetrode names
            initial_window=10,
            initial_start=0,
            downsample_for_vis=downsample_for_vis
        )

        if verbose:
            print(f"Widget created with {len(selected_channels)} tetrodes")
            print("Tetrode mapping:")
            for tetrode_name, channel in selected_channels.items():
                region = tetrode_name.split('_')[0]
                print(f"  {tetrode_name} ({region}): Channel {channel}")

        # Attach loader attributes to the widget for easy access
        widget.loader = loader
        widget.duration = loader.duration
        widget.num_samples = loader.num_samples
        widget.sampling_frequency = loader.sampling_frequency
        widget.num_channels = loader.num_channels
        widget.selected_channels = loader.selected_channels
        widget.tetrode_groups = loader.tetrode_groups

        if hasattr(loader, "target_sampling_frequency"):
            widget.target_sampling_frequency = loader.target_sampling_frequency

        return widget

    else:
        return loader


def test_channel_differences(loader, time_window=0.1):
    """
    Test if different channels show different data.

    This function tests if the memory mapping is working correctly by comparing
    statistics from different channels. If channels show identical data, it
    indicates a bug in the memory mapping.

    Parameters
    ----------
    loader : LazyBinaryLoader
        The LazyBinaryLoader instance to test
    time_window : float, default=0.1
        Duration in seconds to sample from each channel

    Returns
    -------
    bool
        True if all channels show different data, False if any are identical

    Examples
    --------
    # Test channel access after creating loader
    channel_test_result = test_channel_differences(loader)
    if not channel_test_result:
        print("Channel access still has bugs - channels showing same data")
    """
    print("Testing channel differences...")
    print("-" * 50)

    channel_data = {}
    ch_names = list(loader.selected_channels.keys())

    for ch_name in ch_names:
        try:
            # Get a short sample from this channel
            data = loader.get_selected_trace(ch_name, 0, time_window)
            ch_num = loader.selected_channels[ch_name]

            # Calculate statistics
            stats = {
                'channel': ch_num,
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'first_10': data[:10].tolist()
            }

            channel_data[ch_name] = stats

            print(f"Channel {ch_num} ({ch_name[:15]}...):")
            print(f"  mean={stats['mean']:.1f}, std={stats['std']:.1f}")
            print(f"  First 10 samples: {stats['first_10']}")

        except Exception as e:
            print(f"ERROR with {ch_name}: {e}")
            return False

    # Check for identical channels
    print("\nChecking for identical channels:")
    identical_pairs = []

    for i in range(len(ch_names)):
        for j in range(i+1, len(ch_names)):
            name1, name2 = ch_names[i], ch_names[j]
            data1 = np.array(channel_data[name1]['first_10'])
            data2 = np.array(channel_data[name2]['first_10'])

            if np.array_equal(data1, data2):
                ch1_num = channel_data[name1]['channel']
                ch2_num = channel_data[name2]['channel']
                identical_pairs.append((ch1_num, ch2_num))
                print(f"  ⚠️  WARNING: Channels {ch1_num} and {ch2_num} are IDENTICAL!")

    if not identical_pairs:
        print("  ✅ All tested channels have different data")
        return True
    else:
        print("  ❌ Found identical channels - memory mapping bug not fixed!")
        return False


def load_openephys_dat_lazy(filepath, num_channels, channel_ids,
                           sampling_frequency=30000, target_sampling_frequency=1000,
                           plot='widget', verbose=True, time_axis=0):
    """
    Lazy loading version of load_openephys_dat that replicates the original interface.

    This function provides the same interface as the original load_openephys_dat
    that worked correctly, but uses LazyBinaryLoader for memory-efficient data access.

    Parameters are the same as the original working function.
    """

    if verbose:
        print(f"Loading {filepath} (simple channel interface)")
        print(f"  Channels: {len(channel_ids)} ({channel_ids})")
        print(f"  Total channels in file: {num_channels}")
        print(f"  Original sampling rate: {sampling_frequency} Hz")
        print(f"  Target sampling rate: {target_sampling_frequency or sampling_frequency} Hz")
        print(f"  Time axis: {time_axis}")

    # Create simple channel-based organization (no tetrode groups)
    # Each channel gets its own entry in the tetrode_groups
    tetrode_groups = {'channels': {}}
    selected_channels = {}

    for ch_idx in channel_ids:
        ch_name = f'ch{ch_idx}'
        tetrode_groups['channels'][ch_name] = [ch_idx]
        selected_channels[ch_name] = ch_idx

    if verbose:
        print(f"  Created simple channel organization")
        print(f"  Selected channels: {selected_channels}")

    # Create LazyBinaryLoader
    loader = LazyBinaryLoader(
        filepath=filepath,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        dtype='int16',
        file_offset=0,
        time_axis=time_axis,
        tetrode_groups=tetrode_groups,
        selected_channels=selected_channels
    )

    if verbose:
        print(f"LazyBinaryLoader created successfully")
        print(f"  Duration: {loader.duration:.2f}s")
        print(f"  Samples: {loader.num_samples:,}")
        print(f"  File size: {loader.file_size:,} bytes")

    # Return plot or loader
    if plot == 'widget':
        if verbose:
            print("Creating interactive widget...")

        # Use downsampling if specified
        downsample = target_sampling_frequency != sampling_frequency if target_sampling_frequency else False

        widget = dat_file_viewer(
            loader,
            channels=list(selected_channels.keys()),
            initial_window=10,
            initial_start=0,
            downsample_for_vis=downsample
        )

        if verbose:
            print(f"Widget created with {len(channel_ids)} channels")
            print("Channel mapping:")
            for ch_id, ch_name in zip(channel_ids, list(selected_channels.keys())):
                print(f"  Channel {ch_id} -> {ch_name}")

        return widget

    else:
        return loader
