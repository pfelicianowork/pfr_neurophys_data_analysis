"""
visualization.py - Interactive visualization tools for Open Ephys data

This module provides interactive widgets for exploring neural recordings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import ipywidgets as widgets
from IPython.display import display
import warnings

def interactive_neural_viewer(data, initial_channel=None, initial_window=10, initial_start=0):
    """
    Create an interactive widget to explore neural recordings.
    
    Parameters
    ----------
    data : dict
        Loaded Open Ephys data
    initial_channel : str, optional
        Initial channel to display. If None, uses first channel.
    initial_window : float
        Initial time window in seconds (default: 10)
    initial_start : float
        Initial start time in seconds (default: 0)
        
    Returns
    -------
    widget : ipywidgets.VBox
        Interactive widget for exploring data
    """
    
    # Get channel information
    channel_names = list(data.keys())
    if initial_channel is None:
        initial_channel = channel_names[0]
    
    # Get sample rate from first channel
    sample_rate = data[channel_names[0]]['header']['sampleRate']
    total_duration = len(data[channel_names[0]]['data']) / sample_rate
    
    # Create widgets
    channel_dropdown = widgets.Dropdown(
        options=channel_names,
        value=initial_channel,
        description='Channel:',
        style={'description_width': 'initial'}
    )
    
    start_time_slider = widgets.FloatSlider(
        value=initial_start,
        min=0,
        max=total_duration - initial_window,
        step=initial_window/10,
        description='Start Time (s):',
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    
    window_size_slider = widgets.FloatSlider(
        value=initial_window,
        min=1,
        max=min(60, total_duration),
        step=1,
        description='Window Size (s):',
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    
    # Create output widget for the plot
    output = widgets.Output()
    
    def update_plot(channel_name, start_time, window_size):
        """Update the plot with current parameters."""
        with output:
            output.clear_output(wait=True)
            
            # Get data for selected channel
            signal = data[channel_name]['data']
            
            # Calculate indices
            start_idx = int(start_time * sample_rate)
            end_idx = int((start_time + window_size) * sample_rate)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(signal), end_idx)
            
            if start_idx >= end_idx:
                print("Invalid time range")
                return
                
            windowed_signal = signal[start_idx:end_idx]
            time = np.arange(len(windowed_signal)) / sample_rate + start_time
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(time, windowed_signal, 'b-', linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (µV)')
            ax.set_title(f'{channel_name} - {start_time:.1f}s to {start_time + window_size:.1f}s')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([start_time, start_time + window_size])
            
            plt.tight_layout()
            plt.show()
    
    # Connect widgets to update function
    def on_value_change(change):
        update_plot(
            channel_dropdown.value,
            start_time_slider.value,
            window_size_slider.value
        )
    
    channel_dropdown.observe(on_value_change, 'value')
    start_time_slider.observe(on_value_change, 'value')
    window_size_slider.observe(on_value_change, 'value')
    
    # Initial plot
    update_plot(initial_channel, initial_start, initial_window)
    
    # Create navigation buttons
    prev_button = widgets.Button(description='← Previous')
    next_button = widgets.Button(description='Next →')
    zoom_in_button = widgets.Button(description='Zoom In')
    zoom_out_button = widgets.Button(description='Zoom Out')
    
    def on_prev_click(b):
        new_start = max(0, start_time_slider.value - window_size_slider.value)
        start_time_slider.value = new_start
    
    def on_next_click(b):
        new_start = min(total_duration - window_size_slider.value, 
                       start_time_slider.value + window_size_slider.value)
        start_time_slider.value = new_start
    
    def on_zoom_in_click(b):
        new_window = max(1, window_size_slider.value / 2)
        window_size_slider.value = new_window
    
    def on_zoom_out_click(b):
        new_window = min(60, window_size_slider.value * 2)
        window_size_slider.value = new_window
    
    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)
    zoom_in_button.on_click(on_zoom_in_click)
    zoom_out_button.on_click(on_zoom_out_click)
    
    # Layout
    nav_buttons = widgets.HBox([prev_button, next_button, zoom_in_button, zoom_out_button])
    controls = widgets.VBox([channel_dropdown, start_time_slider, window_size_slider, nav_buttons])
    widget = widgets.VBox([controls, output])
    
    return widget


def multi_channel_viewer(data, channels=None, initial_window=10, initial_start=0):
    """
    Create an interactive widget to view multiple channels simultaneously.
    
    Parameters
    ----------
    data : dict
        Loaded Open Ephys data
    channels : list, optional
        List of channel names to display. If None, displays all channels.
    initial_window : float
        Initial time window in seconds (default: 10)
    initial_start : float
        Initial start time in seconds (default: 0)
        
    Returns
    -------
    widget : ipywidgets.VBox
        Interactive widget for exploring multiple channels
    """
    
    if channels is None:
        channels = list(data.keys())
    
    # Get sample rate and total duration
    sample_rate = data[channels[0]]['header']['sampleRate']
    total_duration = len(data[channels[0]]['data']) / sample_rate
    
    # Create widgets
    channel_checkboxes = []
    for channel in channels:
        checkbox = widgets.Checkbox(
            value=True,
            description=channel,
            indent=False
        )
        channel_checkboxes.append(checkbox)
    
    start_time_slider = widgets.FloatSlider(
        value=initial_start,
        min=0,
        max=total_duration - initial_window,
        step=initial_window/10,
        description='Start Time (s):',
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    
    window_size_slider = widgets.FloatSlider(
        value=initial_window,
        min=1,
        max=min(60, total_duration),
        step=1,
        description='Window Size (s):',
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    
    # Create output widget
    output = widgets.Output()
    
    def update_plot(start_time, window_size):
        """Update the multi-channel plot."""
        with output:
            output.clear_output(wait=True)
            
            # Get selected channels
            selected_channels = [cb.description for cb in channel_checkboxes if cb.value]
            
            if not selected_channels:
                print("Please select at least one channel")
                return
            
            # Debug: Print what channels we're trying to plot
            print(f"Plotting channels: {selected_channels}")
            print(f"Available channels in data: {list(data.keys())}")
            
            # Create subplots
            fig, axes = plt.subplots(len(selected_channels), 1, 
                                   figsize=(12, 2 * len(selected_channels)),
                                   sharex=True)
            
            if len(selected_channels) == 1:
                axes = [axes]
            
            for idx, channel_name in enumerate(selected_channels):
                # Verify channel exists in data
                if channel_name not in data:
                    print(f"Warning: Channel {channel_name} not found in data")
                    continue

                signal = data[channel_name]['data']

                # Calculate indices
                start_idx = int(start_time * sample_rate)
                end_idx = int((start_time + window_size) * sample_rate)

                # Ensure indices are within bounds
                start_idx = max(0, start_idx)
                end_idx = min(len(signal), end_idx)

                if start_idx >= end_idx:
                    print(f"Warning: Invalid time range for {channel_name}")
                    continue

                windowed_signal = signal[start_idx:end_idx]
                time = np.arange(len(windowed_signal)) / sample_rate + start_time

                # DEBUG: Add statistical fingerprinting to detect if signals are identical
                signal_mean = windowed_signal.mean()
                signal_std = windowed_signal.std()
                signal_range = windowed_signal.max() - windowed_signal.min()

                # print(f"DEBUG {channel_name}: mean={signal_mean:.2f}, std={signal_std:.2f}, "
                #       f"range={signal_range:.2f}, first_10=[{', '.join(f'{x:.1f}' for x in windowed_signal[:10])}]")

                # Plot with different colors for each channel
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
                color = colors[idx % len(colors)]

                axes[idx].plot(time, windowed_signal, color=color, linewidth=0.5)
                axes[idx].set_ylabel(f'{channel_name}\n(µV)', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xlim([start_time, start_time + window_size])

                # Add debug info to plot
                axes[idx].text(0.02, 0.95, f'mean: {signal_mean:.1f}, std: {signal_std:.1f}',
                              transform=axes[idx].transAxes, fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            axes[-1].set_xlabel('Time (s)', fontsize=12)
            plt.suptitle(f'Neural Recording - {start_time:.1f}s to {start_time + window_size:.1f}s', 
                        fontsize=14)
            plt.tight_layout()
            plt.show()
    
    # Connect widgets with a flag to prevent multiple simultaneous updates
    update_in_progress = {'flag': False}
    
    def on_value_change(change):
        if not update_in_progress['flag']:
            update_in_progress['flag'] = True
            try:
                update_plot(start_time_slider.value, window_size_slider.value)
            finally:
                update_in_progress['flag'] = False
    
    start_time_slider.observe(on_value_change, 'value')
    window_size_slider.observe(on_value_change, 'value')
    
    for checkbox in channel_checkboxes:
        checkbox.observe(on_value_change, 'value')
    
    # Navigation buttons
    prev_button = widgets.Button(description='← Previous')
    next_button = widgets.Button(description='Next →')
    zoom_in_button = widgets.Button(description='Zoom In')
    zoom_out_button = widgets.Button(description='Zoom Out')
    
    def on_prev_click(b):
        new_start = max(0, start_time_slider.value - window_size_slider.value)
        start_time_slider.value = new_start
    
    def on_next_click(b):
        new_start = min(total_duration - window_size_slider.value, 
                       start_time_slider.value + window_size_slider.value)
        start_time_slider.value = new_start
    
    def on_zoom_in_click(b):
        new_window = max(1, window_size_slider.value / 2)
        window_size_slider.value = new_window
    
    def on_zoom_out_click(b):
        new_window = min(60, window_size_slider.value * 2)
        window_size_slider.value = new_window
    
    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)
    zoom_in_button.on_click(on_zoom_in_click)
    zoom_out_button.on_click(on_zoom_out_click)
    
    # Layout
    channel_selection = widgets.VBox([
        widgets.Label('Select Channels:')
    ] + channel_checkboxes)
    
    nav_buttons = widgets.HBox([prev_button, next_button, zoom_in_button, zoom_out_button])
    time_controls = widgets.VBox([start_time_slider, window_size_slider, nav_buttons])
    controls = widgets.HBox([channel_selection, time_controls])
    widget = widgets.VBox([controls, output])
    
    # Initial plot - do this BEFORE returning to avoid timing issues
    with output:
        # Clear any previous output
        output.clear_output(wait=True)
        
        # Create initial plot
        selected_channels = [cb.description for cb in channel_checkboxes if cb.value]
        
        if selected_channels:
            fig, axes = plt.subplots(len(selected_channels), 1, 
                                   figsize=(12, 2 * len(selected_channels)),
                                   sharex=True)
            
            if len(selected_channels) == 1:
                axes = [axes]
            
            for idx, channel_name in enumerate(selected_channels):
                signal = data[channel_name]['data']

                start_idx = int(initial_start * sample_rate)
                end_idx = int((initial_start + initial_window) * sample_rate)
                start_idx = max(0, start_idx)
                end_idx = min(len(signal), end_idx)

                windowed_signal = signal[start_idx:end_idx]
                time = np.arange(len(windowed_signal)) / sample_rate + initial_start

                # DEBUG: Add statistical fingerprinting to initial plot as well
                signal_mean = windowed_signal.mean()
                signal_std = windowed_signal.std()

                print(f"INITIAL PLOT {channel_name}: mean={signal_mean:.2f}, std={signal_std:.2f}, "
                      f"first_5=[{', '.join(f'{x:.1f}' for x in windowed_signal[:5])}]")

                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
                color = colors[idx % len(colors)]

                axes[idx].plot(time, windowed_signal, color=color, linewidth=0.5)
                axes[idx].set_ylabel(f'{channel_name}\n(µV)', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xlim([initial_start, initial_start + initial_window])

                axes[idx].text(0.02, 0.95, f'mean: {signal_mean:.1f}, std: {signal_std:.1f}',
                              transform=axes[idx].transAxes, fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            axes[-1].set_xlabel('Time (s)', fontsize=12)
            plt.suptitle(f'Neural Recording - {initial_start:.1f}s to {initial_start + initial_window:.1f}s', 
                        fontsize=14)
            plt.tight_layout()
            plt.show()
    
    return widget


def simple_interactive_plot(data, channel_name=None, window_duration=10):
    """
    Simple interactive plot for quick visualization.
    
    Parameters
    ----------
    data : dict
        Loaded Open Ephys data
    channel_name : str, optional
        Channel to display. If None, uses first channel.
    window_duration : float
        Time window in seconds
        
    Returns
    -------
    None
        Displays interactive plot
    """
    
    if channel_name is None:
        channel_name = list(data.keys())[0]
    
    signal = data[channel_name]['data']
    sample_rate = data[channel_name]['header']['sampleRate']
    total_duration = len(signal) / sample_rate
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial plot
    n_samples = int(window_duration * sample_rate)
    time = np.arange(n_samples) / sample_rate
    line, = ax.plot(time, signal[:n_samples], 'b-', linewidth=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'{channel_name} - Interactive View')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, window_duration])
    
    # Create slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=ax_slider,
        label='Start Time (s)',
        valmin=0,
        valmax=total_duration - window_duration,
        valinit=0,
        valstep=window_duration/10
    )
    
    def update(val):
        start_time = time_slider.val
        start_idx = int(start_time * sample_rate)
        end_idx = start_idx + n_samples
        
        if end_idx > len(signal):
            end_idx = len(signal)
            start_idx = end_idx - n_samples
        
        windowed_signal = signal[start_idx:end_idx]
        time = np.arange(len(windowed_signal)) / sample_rate
        
        line.set_data(time, windowed_signal)
        ax.set_xlim([time[0], time[-1]])
        ax.set_title(f'{channel_name} - {start_time:.1f}s to {start_time + window_duration:.1f}s')
        fig.canvas.draw_idle()
    
    time_slider.on_changed(update)
    
    # Add navigation buttons
    ax_prev = plt.axes([0.1, 0.025, 0.1, 0.04])
    ax_next = plt.axes([0.21, 0.025, 0.1, 0.04])
    ax_zoom_in = plt.axes([0.32, 0.025, 0.1, 0.04])
    ax_zoom_out = plt.axes([0.43, 0.025, 0.1, 0.04])
    
    button_prev = Button(ax_prev, '← Prev')
    button_next = Button(ax_next, 'Next →')
    button_zoom_in = Button(ax_zoom_in, 'Zoom In')
    button_zoom_out = Button(ax_zoom_out, 'Zoom Out')
    
    def prev_click(event):
        new_start = max(0, time_slider.val - window_duration)
        time_slider.set_val(new_start)
    
    def next_click(event):
        new_start = min(total_duration - window_duration, time_slider.val + window_duration)
        time_slider.set_val(new_start)
    
    def zoom_in_click(event):
        nonlocal window_duration
        window_duration = max(1, window_duration / 2)
        update(time_slider.val)
    
    def zoom_out_click(event):
        nonlocal window_duration
        window_duration = min(60, window_duration * 2)
        update(time_slider.val)
    
    button_prev.on_clicked(prev_click)
    button_next.on_clicked(next_click)
    button_zoom_in.on_clicked(zoom_in_click)
    button_zoom_out.on_clicked(zoom_out_click)
    
    plt.show()
def unified_viewer(data_source, channels=None, initial_window=10, initial_start=0, **kwargs):
    """
    Unified viewer that automatically works with both .continuous files and .dat files.

    This function detects the data format and uses the appropriate visualization method.

    Parameters
    ----------
    data_source : str, dict, or LazyBinaryLoader
        - str: Path to .continuous file or .dat file (auto-detected)
        - dict: Already loaded .continuous data
        - LazyBinaryLoader: Lazy loader for .dat files
    channels : list, optional
        List of channels to display (format-dependent)
        - For dict/.continuous: channel names/keys
        - For LazyBinaryLoader: tetrode names or channel indices
    initial_window : float
        Initial time window in seconds (default: 10)
    initial_start : float
        Initial start time in seconds (default: 0)
    **kwargs : dict
        Additional arguments passed to specific viewers

    Returns
    -------
    widget or None
        Interactive visualization widget

    Examples
    --------
    # For .continuous files (existing)
    widget = unified_viewer('/path/to/recording_100_CH1.continuous')

    # For .dat files (new capability)
    widget = unified_viewer('/path/to/ephys.dat', num_channels=43, tetrode_groups=tetrode_config)

    # With pre-loaded data
    data = load_open_ephys_data('/path/to/continuous/file')
    widget = unified_viewer(data)

    # With LazyBinaryLoader
    loader = LazyBinaryLoader('ephys.dat', num_channels=43, ...)
    widget = unified_viewer(loader, channels=['CA1_tet1', 'CA3_tet1'])
    """

    from .lazy_loader import LazyBinaryLoader
    from .data_loader import load_open_ephys_data

    # CASE 1: String path - auto-detect format
    if isinstance(data_source, str):
        if data_source.endswith('.dat'):
            # .dat file with lazy loading
            if 'num_channels' not in kwargs:
                raise ValueError("num_channels required for .dat files")

            # Extract LazyBinaryLoader parameters
            loader_kwargs = {k: v for k, v in kwargs.items()
                           if k in ['num_channels', 'sampling_frequency', 'dtype',
                                  'file_offset', 'time_axis', 'tetrode_groups', 'selected_channels']}

            loader = LazyBinaryLoader(data_source, **loader_kwargs)
            return dat_file_viewer(loader, channels=channels, initial_window=initial_window,
                                 initial_start=initial_start, **kwargs)
        else:
            # .continuous file - use existing loader
            if channels is not None:
                kwargs['channels'] = channels
            data = load_open_ephys_data(data_source, **kwargs)
            return multi_channel_viewer(data, channels=channels, initial_window=initial_window,
                                      initial_start=initial_start)

    # CASE 2: LazyBinaryLoader
    elif isinstance(data_source, LazyBinaryLoader):
        return dat_file_viewer(data_source, channels=channels, initial_window=initial_window,
                             initial_start=initial_start, **kwargs)

    # CASE 3: Already loaded continuous data (dict)
    elif isinstance(data_source, dict):
        return multi_channel_viewer(data_source, channels=channels, initial_window=initial_window,
                                  initial_start=initial_start)

    else:
        raise TypeError(f"Unsupported data_source type: {type(data_source)}. "
                       "Use string path, LazyBinaryLoader, or loaded continuous data dict.")


def dat_file_viewer(loader, channels=None, initial_window=10, initial_start=0,
                    downsample_for_vis=True, **kwargs):
    """
    Interactive viewer specifically designed for .dat files with tetrode organization.

    This viewer shows ONE selected channel per tetrode in separate subplots,
    creating clean neural recordings that look like traditional voltage traces.

    Parameters
    ----------
    loader : LazyBinaryLoader
        Lazy loader instance with tetrode organization
    channels : list, optional
        Tetrode names to show. If None, shows all selected channels.
    initial_window : float
        Initial time window in seconds (default: 10)
    initial_start : float
        Initial start time in seconds (default: 0)
    downsample_for_vis : bool
        Automatically downsample to 1kHz for visualization speed (default: True)
    **kwargs : dict
        Additional arguments

    Returns
    -------
    widget : ipywidgets.VBox
        Interactive visualization widget

    Examples
    --------
    # Basic usage with auto-configuration
    widget = dat_file_viewer(loader)

    # Custom channels and settings
    widget = dat_file_viewer(loader, channels=['CA1_tet1', 'CA3_tet1'],
                           initial_window=5, downsample_for_vis=False)
    """

    # Determine channels to plot (only selected ones - one per tetrode)
    if channels is None:
        selected_channels = list(loader.selected_channels.keys())
    else:
        # Filter to only available selected channels
        selected_channels = [ch for ch in channels if ch in loader.selected_channels]

    if not selected_channels:
        raise ValueError("No valid selected channels to display. Check tetrode configuration.")

    # Brain region colors for consistent coloring
    region_colors = {
        'CA1': '#FF6B6B',  # Red
        'CA3': '#4ECDC4',  # Teal
        'PFC': '#45B7D1',  # Blue
        'RTC': '#FFA07A',  # Light salmon
        'default': '#96CEB4'  # Light green
    }

    def get_color_for_channel(channel_name):
        """Get color based on brain region."""
        if '_' in channel_name:
            region = channel_name.split('_')[0]
        else:
            region = 'default'
        return region_colors.get(region, region_colors['default'])

    # Get total duration
    total_duration = loader.duration

    # Channel info display
    channel_info = widgets.HTML(
        f"<b>Showing {len(selected_channels)} selected channels (one per tetrode):</b><br>" +
        "".join([f"• <span style='color:{get_color_for_channel(ch)}'>●</span> {ch} "
                f"(ch{loader.selected_channels[ch]})<br>"
                for ch in selected_channels])
    )

    start_time_slider = widgets.FloatSlider(
        value=initial_start,
        min=0,
        max=max(0, total_duration - initial_window),
        step=max(0.1, initial_window/20),
        description='Start Time (s):',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )

    window_size_slider = widgets.FloatSlider(
        value=initial_window,
        min=1,
        max=min(60, total_duration),
        step=1,
        description='Window Size (s):',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )

    # Downsampling control
    downsample_toggle = widgets.Checkbox(
        value=downsample_for_vis,
        description=f'Visualize at 1kHz (fast plotting)',
        indent=False
    )

    # Create output widget with appropriate height
    plot_height = max(400, len(selected_channels) * 80)
    output = widgets.Output(layout=widgets.Layout(height=f'{plot_height}px', overflow='auto'))

    def update_plot(start_time, window_size, downsample=False):
        """Update the multi-subplot neural recording view."""
        with output:
            output.clear_output(wait=True)

            if not selected_channels:
                print("No channels to display")
                return

            # Determine target sample rate
            target_fs = 1000 if downsample else None

            # Get data for all selected channels
            traces_data = []
            traces_metadata = []

            for tetrode_name in selected_channels:
                try:
                    # Get selected trace for this tetrode
                    trace = loader.get_selected_trace(
                        tetrode_name,
                        start_time,
                        start_time + window_size,
                        target_fs=target_fs
                    )

                    if len(trace) > 0:
                        traces_data.append(trace)
                        traces_metadata.append({
                            'name': tetrode_name,
                            'channel': loader.selected_channels[tetrode_name],
                            'color': get_color_for_channel(tetrode_name),
                            'region': tetrode_name.split('_')[0] if '_' in tetrode_name else 'unknown'
                        })

                except Exception as e:
                    print(f"Warning: Could not load {tetrode_name}: {e}")
                    continue

            if not traces_data:
                print("No data loaded - check time range and file")
                return

            # Create subplots - one per channel like traditional neural recordings
            n_channels = len(traces_data)
            fig, axes = plt.subplots(n_channels, 1, figsize=(14, max(4, n_channels * 1.5)),
                                   sharex=True)

            if n_channels == 1:
                axes = [axes]

            # Effective sample rate for time axis
            effective_fs = 1000 if downsample else loader.sampling_frequency

            # Plot each neural trace in its own subplot
            for idx, (trace, metadata) in enumerate(zip(traces_data, traces_metadata)):
                time_axis = np.arange(len(trace)) / effective_fs + start_time

                # Plot the neural trace
                axes[idx].plot(time_axis, trace, color=metadata['color'],
                             linewidth=0.8, alpha=0.9)

                # Label the channel
                ylabel = f'{metadata["name"]}\nCh{metadata["channel"]}'
                axes[idx].set_ylabel(ylabel, fontsize=10, fontweight='bold',
                                   color=metadata['color'])

                # Add statistics in small text
                if len(trace) > 10:
                    mean_amp = trace.mean()
                    std_amp = trace.std()
                    stats_text = f'μ={mean_amp:.1f}, σ={std_amp:.1f}'
                    axes[idx].text(0.98, 0.95, stats_text,
                                 transform=axes[idx].transAxes, fontsize=8,
                                 verticalalignment='top', horizontalalignment='right',
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

                # Grid and limits
                axes[idx].grid(True, alpha=0.2)
                axes[idx].set_xlim([start_time, start_time + window_size])

                # Remove x-axis ticks/labels for all but bottom subplot
                if idx < n_channels - 1:
                    axes[idx].set_xticklabels([])
                else:
                    axes[idx].set_xlabel('Time (s)', fontsize=11)

            # Add title with recording info
            sampling_info = f"@{effective_fs}Hz" if downsample else f"@{int(loader.sampling_frequency)}Hz"
            fig.suptitle(f'Neural Recordings - {start_time:.1f}s to {start_time + window_size:.1f}s {sampling_info}',
                        fontsize=14, fontweight='bold', y=0.95)

            # Add legend for regions in the top-right
            region_legend = {}
            for metadata in traces_metadata:
                region = metadata['region']
                if region not in region_legend:
                    region_legend[region] = metadata['color']

            if region_legend:
                legend_elements = [plt.Line2D([0], [0], color=color, linewidth=3,
                                     label=region.upper()) for region, color in region_legend.items()]
                axes[0].legend(handles=legend_elements, loc='upper right',
                              bbox_to_anchor=(1.05, 1.15), fontsize=9, title='Regions')

            plt.tight_layout()
            plt.subplots_adjust(top=0.85, right=0.85)  # Make room for legend
            plt.show()

    # Connect widget events
    update_in_progress = {'flag': False}

    def on_value_change(change):
        if not update_in_progress['flag']:
            update_in_progress['flag'] = True
            try:
                update_plot(
                    start_time_slider.value,
                    window_size_slider.value,
                    downsample_toggle.value
                )
            finally:
                update_in_progress['flag'] = False

    # Observe changes
    start_time_slider.observe(on_value_change, 'value')
    window_size_slider.observe(on_value_change, 'value')
    downsample_toggle.observe(on_value_change, 'value')

    # Navigation buttons with time jumps
    nav_buttons = widgets.HBox([
        widgets.Button(description='← Prev 50%', layout=widgets.Layout(width='80px')),
        widgets.Button(description='Next 50% →', layout=widgets.Layout(width='80px')),
        widgets.Button(description='Zoom In', layout=widgets.Layout(width='70px')),
        widgets.Button(description='Zoom Out', layout=widgets.Layout(width='80px')),
    ])

    def on_prev_click(b):
        new_start = max(0, start_time_slider.value - window_size_slider.value * 0.5)
        start_time_slider.value = new_start

    def on_next_click(b):
        new_start = min(total_duration - window_size_slider.value,
                       start_time_slider.value + window_size_slider.value * 0.5)
        start_time_slider.value = new_start

    def on_zoom_in_click(b):
        new_window = max(1, window_size_slider.value / 2)
        window_size_slider.value = new_window

    def on_zoom_out_click(b):
        new_window = min(60, window_size_slider.value * 2)
        window_size_slider.value = new_window

    nav_buttons.children[0].on_click(on_prev_click)
    nav_buttons.children[1].on_click(on_next_click)
    nav_buttons.children[2].on_click(on_zoom_in_click)
    nav_buttons.children[3].on_click(on_zoom_out_click)

    # Layout
    time_controls = widgets.VBox([
        widgets.HTML("<b>Time Navigation:</b>"),
        start_time_slider,
        window_size_slider,
        nav_buttons,
        downsample_toggle,
    ], layout=widgets.Layout(width='500px'))

    controls = widgets.HBox([channel_info, time_controls])
    widget = widgets.VBox([controls, output])

    # Initial plot
    initial_downsample = downsample_for_vis
    update_plot(initial_start, initial_window, initial_downsample)

    return widget


# Example usage function (existing)
def quick_view(data, method='simple'):
    """
    Quick visualization of loaded data.

    Parameters
    ----------
    data : dict
        Loaded Open Ephys data
    method : str
        'simple', 'single', or 'multi'

    Returns
    -------
    widget or None
        Depending on method
    """

    if method == 'simple':
        simple_interactive_plot(data)
    elif method == 'single':
        return interactive_neural_viewer(data)
    elif method == 'multi':
        return multi_channel_viewer(data)
    else:
        raise ValueError("Method must be 'simple', 'single', or 'multi'")
