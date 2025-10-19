"""
brain_regions.py - Analysis functions for brain region-specific data

This module provides specialized functions for analyzing neural data organized by brain regions.
It includes tools for cross-region correlation, region-specific filtering, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .utils import create_time_vector, format_time
from matplotlib.gridspec import GridSpec
from .visualization import dat_file_viewer


def get_region_data(data, region_name, format_type='old'):
    """
    Extract data for a specific brain region.

    Parameters
    ----------
    data : dict or OpenEphysRecording
        Data loaded with channel_groups
    region_name : str
        Name of the brain region to extract
    format_type : str
        'old' or 'new' format

    Returns
    -------
    region_data : dict or list
        Data for the specified region
    """
    if format_type == 'old':
        if region_name not in data:
            raise ValueError(f"Region '{region_name}' not found in data")
        return data[region_name]
    else:
        # For new format, we'd need to implement region-based extraction
        # For now, return the recording object
        return data



"""
Fixed version of plot_region_lfp_with_spike_raster with firing rate bug fix.

KEY FIX: Pass raw tetrode numbers (e.g., [7, 8]) instead of formatted strings 
(e.g., ['CA1_7', 'CA1_8']) to spike_analyzer.analyze_data()
"""

def plot_region_lfp_with_spike_raster(data_lfp, selected_channels, tetrode_groups, spike_analyzer, regions,
                                       duration=None, figsize=(12, 10), lfp_color='blue', spike_color='red',
                                       spike_markersize=2, lfp_alpha=0.8, lfp_downsample_factor=100,
                                       target_fs=500, show_progress=True, region_colors=None,
                                       widget_mode=False, window_size=10.0, show_firing_rate=False,
                                       firing_rate_kernel_width=0.05, firing_rate_sampling_rate=1000,
                                       firing_rate_color='green', firing_rate_alpha=0.8):
    """
    FIXED VERSION: Plot LFP from one channel per brain region with spike raster and optional firing rate.

    The key fix: Use raw tetrode numbers (from region_mapping keys) instead of formatted strings
    when calling spike_analyzer.analyze_data().
    """
    if not hasattr(spike_analyzer, 'region_mapping'):
        raise ValueError("Assign brain regions to spike_analyzer first")

    # Build region to tetrode keys mapping
    region_to_tet_keys = {}
    for region, tets in tetrode_groups.items():
        region_to_tet_keys[region] = [f"{region}_{tet}" for tet in tets.keys()]

    if duration is None:
        duration = data_lfp.duration

    # Define default region colors if not provided
    if region_colors is None:
        region_colors = {
            'CA1': '#1f77b4',
            'RTC': '#2ca02c',
            'PFC': '#ff7f0e',
            'HP': '#d62728',
            'CTX': '#9467bd',
            'TH': '#8c564b',
        }

    if widget_mode:
        print(f"\n🖱️  Launching Custom Interactive Widget!")
        print(f"📏 Initial window: {window_size}s")
        print(f"📊 Total duration: {data_lfp.duration:.1f}s")
        print(f"🎛️  Region-based layout with navigation controls")
        print(f"💡 Tip: Close widget to continue")
        
        # Use the FIXED widget function
        widget = create_brain_region_widget_FIXED(
            data_lfp, selected_channels, tetrode_groups, spike_analyzer, regions,
            region_colors, window_size, show_firing_rate, firing_rate_kernel_width,
            firing_rate_sampling_rate, firing_rate_color, firing_rate_alpha
        )
        
        from IPython.display import display
        display(widget)
        return

    # Static plotting
    print(f"\n🖱️  Static Plot Mode")
    print(f"📏 Duration: {duration}s at {target_fs}Hz")
    print(f"💡 Tip: Set widget_mode=True for interactive exploration")

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    n_regions = len(regions)

    if show_firing_rate:
        rows_per_region = 3
        height_ratios = [1, 1, 1] * n_regions
        hspace = 0.05
    else:
        rows_per_region = 2
        height_ratios = [1, 1] * n_regions
        hspace = 0.05

    gs = GridSpec(n_regions * rows_per_region, 1, figure=fig,
                   height_ratios=height_ratios,
                   hspace=hspace)

    if show_progress:
        print(f"Loading and plotting {len(regions)} brain regions for {duration}s at {target_fs}Hz...")

    for i, region in enumerate(regions):
        region_color = region_colors.get(region, '#1f77b4')
        print(f"Processing {region} (color: {region_color})...")

        # ========== LFP TRACE (BOTTOM) ==========
        lfp_row = i * rows_per_region
        ax_lfp = fig.add_subplot(gs[lfp_row])

        ax_lfp.spines['top'].set_visible(False)
        ax_lfp.spines['right'].set_visible(False)
        ax_lfp.spines['left'].set_visible(False)
        if i == n_regions - 1:
            ax_lfp.spines['bottom'].set_visible(True)
        else:
            ax_lfp.spines['bottom'].set_visible(False)

        tet_keys = region_to_tet_keys.get(region, [])
        if tet_keys:
            tet_key = tet_keys[0]
            selected_ch = selected_channels.get(tet_key)

            if selected_ch is not None:
                print(f"  Loading LFP from {tet_key} (channel {selected_ch})...")

                data = data_lfp.get_trace(selected_ch, start_time=0, end_time=duration,
                                        target_fs=target_fs)
                time_vector = np.arange(len(data)) / target_fs

                print(f"  Loaded {len(data)} samples ({len(data)/target_fs:.1f}s at {target_fs}Hz)")

                ax_lfp.plot(time_vector, data, color=region_color, alpha=lfp_alpha,
                           linewidth=1.0, label=f'{tet_key} LFP')
                ax_lfp.set_ylabel('Voltage\n(µV)', fontsize=10)
                ax_lfp.grid(True, alpha=0.3)
                ax_lfp.tick_params(axis='both', which='major', labelsize=9)

                if i == n_regions - 1:
                    ax_lfp.set_xlabel('Time (s)', fontsize=10)
                else:
                    ax_lfp.set_xlabel('')
                    ax_lfp.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            else:
                ax_lfp.set_title(f'{region} - No LFP Data', color=region_color, fontweight='bold')
                ax_lfp.text(0.5, 0.5, 'No LFP data', transform=ax_lfp.transAxes,
                           ha='center', va='center', alpha=0.5)

        # ========== SPIKE RASTER (MIDDLE) ==========
        spike_row = lfp_row + 1
        ax_spikes = fig.add_subplot(gs[spike_row])

        ax_spikes.spines['top'].set_visible(False)
        ax_spikes.spines['right'].set_visible(False)
        ax_spikes.spines['left'].set_visible(False)
        ax_spikes.spines['bottom'].set_visible(False)

        tetrodes = spike_analyzer.region_to_tetrodes.get(region, [])
        if tetrodes:
            print(f"  Processing spike data for {len(tetrodes)} tetrodes: {tetrodes}...")

            spike_times_list = []
            unit_labels = []

            for tetrode in tetrodes:
                units = spike_analyzer.tetrode_mapping.get(tetrode, [])
                for unit_idx, unit in enumerate(units):
                    spike_times = unit['spk_time']
                    spike_times = spike_times[spike_times <= duration]
                    if len(spike_times) > 0:
                        spike_times_list.append(spike_times)
                        unit_labels.append(f'{tetrode}_u{unit_idx}')

            print(f"  Found {len(spike_times_list)} units with spikes")

            if spike_times_list:
                ax_spikes.eventplot(spike_times_list, colors=region_color,
                                   lineoffsets=list(range(len(spike_times_list))),
                                   linelengths=0.8, linewidths=0.5)
                ax_spikes.set_yticks(np.arange(len(unit_labels)))
                ax_spikes.set_yticklabels(unit_labels, fontsize=8)
                ax_spikes.set_title(f'{region}', color=region_color, fontweight='bold')
                ax_spikes.grid(True, alpha=0.3)
                ax_spikes.set_ylabel('Units', fontsize=10)
                ax_spikes.tick_params(axis='both', which='major', labelsize=9)
            else:
                ax_spikes.set_title(f'{region} - No Spikes', color=region_color, fontweight='bold')
                ax_spikes.set_yticks([])
                ax_spikes.text(0.5, 0.5, 'No spike data', transform=ax_spikes.transAxes,
                              ha='center', va='center', alpha=0.5)

        if i < n_regions - 1:
            ax_spikes.set_xlabel('')
            ax_spikes.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # ========== FIRING RATE (TOP) - FIX IS HERE ==========
        if show_firing_rate:
            firing_rate_row = spike_row + 1
            ax_firing_rate = fig.add_subplot(gs[firing_rate_row])

            ax_firing_rate.spines['top'].set_visible(False)
            ax_firing_rate.spines['right'].set_visible(False)
            ax_firing_rate.spines['left'].set_visible(False)
            if i == n_regions - 1:
                ax_firing_rate.spines['bottom'].set_visible(True)
            else:
                ax_firing_rate.spines['bottom'].set_visible(False)

            tetrodes = spike_analyzer.region_to_tetrodes.get(region, [])
            if tetrodes:
                print(f"  Calculating firing rate for {len(tetrodes)} tetrodes...")

                # ✅ FIX: Use raw tetrode numbers, NOT formatted strings
                tetrode_ids = tetrodes

                print(f"  Computing firing rate for raw tetrode IDs: {tetrode_ids}")
                
                try:
                    firing_rate_results = spike_analyzer.analyze_data(
                        identifiers=tetrode_ids,
                        id_type='tetrode',
                        kernel_width=firing_rate_kernel_width,
                        sampling_rate=firing_rate_sampling_rate,
                        time_range=(0, duration) if duration else None,
                        full_recording=False  # ✅ FIX: Use False to respect time_range
                    )

                    if firing_rate_results['window']['average'] is not None:
                        times = firing_rate_results['window']['times']
                        avg_firing_rate = firing_rate_results['window']['average']
                        print(f"  ✓ Plotting firing rate: {len(times)} time points, mean rate: {np.mean(avg_firing_rate):.2f} Hz")

                        ax_firing_rate.plot(times, avg_firing_rate, color=region_color,
                                           alpha=firing_rate_alpha, linewidth=1.5,
                                           label=f'{region} Avg Firing Rate')
                        ax_firing_rate.set_ylabel('Firing Rate\n(Hz)', fontsize=10)
                        ax_firing_rate.grid(True, alpha=0.3)
                        ax_firing_rate.tick_params(axis='both', which='major', labelsize=9)

                        if i == n_regions - 1:
                            ax_firing_rate.set_xlabel('Time (s)', fontsize=10)
                        else:
                            ax_firing_rate.set_xlabel('')
                            ax_firing_rate.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                    else:
                        print(f"  ✗ No firing rate data returned for {region}")
                        ax_firing_rate.text(0.5, 0.5, f'No spikes in {region}', 
                                           transform=ax_firing_rate.transAxes,
                                           ha='center', va='center', alpha=0.5, fontsize=10)
                        if i == n_regions - 1:
                            ax_firing_rate.set_xlabel('Time (s)', fontsize=10)
                        else:
                            ax_firing_rate.set_xlabel('')
                            ax_firing_rate.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                            
                except Exception as e:
                    print(f"  ✗ Error calculating firing rate for {region}: {e}")
                    ax_firing_rate.text(0.5, 0.5, f'Error: {region}', 
                                       transform=ax_firing_rate.transAxes,
                                       ha='center', va='center', alpha=0.5, fontsize=10, color='red')
                    if i == n_regions - 1:
                        ax_firing_rate.set_xlabel('Time (s)', fontsize=10)
                    else:
                        ax_firing_rate.set_xlabel('')
                        ax_firing_rate.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            else:
                ax_firing_rate.text(0.5, 0.5, 'No tetrode data', transform=ax_firing_rate.transAxes,
                                   ha='center', va='center', alpha=0.5)

    plt.tight_layout()
    plt.show()


def create_brain_region_widget_FIXED(data_lfp, selected_channels, tetrode_groups, spike_analyzer, regions,
                               region_colors, initial_window=10.0, show_firing_rate=False,
                               firing_rate_kernel_width=0.05, firing_rate_sampling_rate=1000,
                               firing_rate_color='green', firing_rate_alpha=0.8):
    """
    FIXED VERSION: Create interactive widget with corrected firing rate calculation.

    Key fix: Use raw tetrode numbers instead of formatted strings when calling analyze_data().
    """
    import ipywidgets as widgets
    from IPython.display import display

    total_duration = data_lfp.duration

    start_time_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=max(0.1, total_duration - initial_window),
        step=max(0.1, initial_window / 20),
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

    output = widgets.Output(layout=widgets.Layout(height='600px', overflow='auto'))
    update_in_progress = False

    def update_plot(start_time, window_size):
        nonlocal update_in_progress
        if update_in_progress:
            return
        update_in_progress = True

        try:
            with output:
                output.clear_output(wait=True)

                fig = plt.figure(figsize=(12, 10))
                n_regions = len(regions)

                if show_firing_rate:
                    rows_per_region = 3
                    height_ratios = [1, 1, 1] * n_regions
                else:
                    rows_per_region = 2
                    height_ratios = [1, 1] * n_regions

                gs = GridSpec(n_regions * rows_per_region, 1, figure=fig,
                              height_ratios=height_ratios, hspace=0.05)

                region_to_tet_keys = {
                    region: [f"{region}_{tet}" for tet in tets.keys()]
                    for region, tets in tetrode_groups.items()
                }

                for i, region in enumerate(regions):
                    region_color = region_colors.get(region, '#1f77b4')

                    # LFP TRACE
                    lfp_row = i * rows_per_region
                    ax_lfp = fig.add_subplot(gs[lfp_row])
                    ax_lfp.spines['top'].set_visible(False)
                    ax_lfp.spines['right'].set_visible(False)
                    ax_lfp.spines['left'].set_visible(False)
                    ax_lfp.spines['bottom'].set_visible(False)

                    tet_keys = region_to_tet_keys.get(region, [])
                    if tet_keys:
                        tet_key = tet_keys[0]
                        selected_ch = selected_channels.get(tet_key)

                        if selected_ch is not None:
                            data = data_lfp.get_trace(
                                selected_ch,
                                start_time,
                                start_time + window_size,
                                target_fs=500
                            )
                            time_vector = np.arange(len(data)) / 500

                            ax_lfp.plot(time_vector + start_time, data,
                                        color=region_color, alpha=0.8, linewidth=1.0)
                            ax_lfp.set_ylabel('Voltage\n(µV)', fontsize=10)
                            ax_lfp.grid(True, alpha=0.3)
                            ax_lfp.tick_params(axis='both', which='major', labelsize=9)
                            ax_lfp.set_xlim(start_time, start_time + window_size)

                            if i == n_regions - 1:
                                ax_lfp.set_xlabel('Time (s)', fontsize=10)
                            else:
                                ax_lfp.set_xlabel('')
                                ax_lfp.tick_params(axis='x', which='both',
                                                   bottom=False, labelbottom=False)
                        else:
                            ax_lfp.set_title(f'{region} - No LFP Data',
                                             color=region_color, fontweight='bold')
                            ax_lfp.text(0.5, 0.5, 'No LFP data',
                                        transform=ax_lfp.transAxes,
                                        ha='center', va='center', alpha=0.5)
                            ax_lfp.set_xlim(start_time, start_time + window_size)

                    # SPIKE RASTER
                    spike_row = lfp_row + 1
                    ax_spikes = fig.add_subplot(gs[spike_row])
                    ax_spikes.spines['top'].set_visible(False)
                    ax_spikes.spines['right'].set_visible(False)
                    ax_spikes.spines['left'].set_visible(False)
                    ax_spikes.spines['bottom'].set_visible(False)

                    tetrodes = spike_analyzer.region_to_tetrodes.get(region, [])
                    if tetrodes:
                        spike_times_list = []
                        unit_labels = []

                        for tetrode in tetrodes:
                            units = spike_analyzer.tetrode_mapping.get(tetrode, [])
                            for unit_idx, unit in enumerate(units):
                                spike_times = unit['spk_time']
                                window_spikes = spike_times[
                                    (spike_times >= start_time) &
                                    (spike_times <= start_time + window_size)
                                ]
                                if window_spikes.size > 0:
                                    spike_times_list.append(window_spikes)
                                    unit_labels.append(f'{tetrode}_u{unit_idx}')

                        if spike_times_list:
                            ax_spikes.eventplot(
                                spike_times_list,
                                colors=region_color,
                                lineoffsets=list(range(len(spike_times_list))),
                                linelengths=0.8,
                                linewidths=0.5
                            )
                            ax_spikes.set_yticks(np.arange(len(unit_labels)))
                            ax_spikes.set_yticklabels(unit_labels, fontsize=8)
                            ax_spikes.set_title(f'{region}',
                                                color=region_color, fontweight='bold')
                            ax_spikes.grid(True, alpha=0.3)
                            ax_spikes.set_ylabel('Units', fontsize=10)
                            ax_spikes.tick_params(axis='both', which='major', labelsize=9)
                            ax_spikes.set_xlim(start_time, start_time + window_size)
                        else:
                            ax_spikes.set_title(f'{region} - No Spikes',
                                                color=region_color, fontweight='bold')
                            ax_spikes.set_yticks([])
                            ax_spikes.text(0.5, 0.5, 'No spike data',
                                           transform=ax_spikes.transAxes,
                                           ha='center', va='center', alpha=0.5)
                            ax_spikes.set_xlim(start_time, start_time + window_size)

                    if i < n_regions - 1:
                        ax_spikes.set_xlabel('')
                        ax_spikes.tick_params(axis='x', which='both',
                                              bottom=False, labelbottom=False)

                    # FIRING RATE
                    if show_firing_rate:
                        firing_rate_row = spike_row + 1
                        ax_firing_rate = fig.add_subplot(gs[firing_rate_row])
                        ax_firing_rate.spines['top'].set_visible(False)
                        ax_firing_rate.spines['right'].set_visible(False)
                        ax_firing_rate.spines['left'].set_visible(False)
                        ax_firing_rate.spines['bottom'].set_visible(False)

                        if tetrodes:
                            tetrode_ids = tetrodes
                            try:
                                firing_rate_results = spike_analyzer.analyze_data(
                                    identifiers=tetrode_ids,
                                    id_type='tetrode',
                                    kernel_width=firing_rate_kernel_width,
                                    sampling_rate=firing_rate_sampling_rate,
                                    time_range=(start_time, start_time + window_size),
                                    full_recording=False
                                )

                                window_result = firing_rate_results.get('window', {})
                                avg_firing_rate = window_result.get('average')
                                times = np.asarray(window_result.get('times', []))

                                if avg_firing_rate is not None and times.size > 0:
                                    avg_firing_rate = np.asarray(avg_firing_rate)

                                    is_relative = (
                                        np.nanmin(times) >= -1e-9 and
                                        np.nanmax(times) <= window_size + 1e-9
                                    )
                                    plot_times = times + start_time if is_relative else times
                                    x_start = start_time if is_relative else np.nanmin(plot_times)
                                    x_stop = (start_time + window_size) if is_relative else np.nanmax(plot_times)

                                    ax_firing_rate.plot(
                                        plot_times,
                                        avg_firing_rate,
                                        color=region_color,
                                        alpha=firing_rate_alpha,
                                        linewidth=1.5
                                    )
                                    ax_firing_rate.set_ylabel('Firing Rate\n(Hz)', fontsize=10)
                                    ax_firing_rate.grid(True, alpha=0.3)
                                    ax_firing_rate.tick_params(axis='both', which='major', labelsize=9)
                                    if np.isfinite(x_start) and np.isfinite(x_stop):
                                        ax_firing_rate.set_xlim(x_start, x_stop)

                                    if i == n_regions - 1:
                                        ax_firing_rate.set_xlabel('Time (s)', fontsize=10)
                                    else:
                                        ax_firing_rate.set_xlabel('')
                                        ax_firing_rate.tick_params(axis='x', which='both',
                                                                   bottom=False, labelbottom=False)
                                else:
                                    ax_firing_rate.text(
                                        0.5, 0.5, f'No spikes in {region}',
                                        transform=ax_firing_rate.transAxes,
                                        ha='center', va='center',
                                        alpha=0.5, fontsize=10
                                    )
                                    ax_firing_rate.set_xlim(start_time, start_time + window_size)
                            except Exception as exc:
                                ax_firing_rate.text(
                                    0.5, 0.5, f'Error: {str(exc)[:30]}',
                                    transform=ax_firing_rate.transAxes,
                                    ha='center', va='center',
                                    alpha=0.5, fontsize=10, color='red'
                                )
                                ax_firing_rate.set_xlim(start_time, start_time + window_size)
                        else:
                            ax_firing_rate.text(
                                0.5, 0.5, f'No tetrodes in {region}',
                                transform=ax_firing_rate.transAxes,
                                ha='center', va='center',
                                alpha=0.5, fontsize=10
                            )
                            ax_firing_rate.set_xlim(start_time, start_time + window_size)

                plt.tight_layout()
                plt.show()
        finally:
            update_in_progress = False

    def on_value_change(change):
        update_plot(start_time_slider.value, window_size_slider.value)

    start_time_slider.observe(on_value_change, 'value')
    window_size_slider.observe(on_value_change, 'value')

    prev_button = widgets.Button(description='← Previous')
    next_button = widgets.Button(description='Next →')
    zoom_in_button = widgets.Button(description='Zoom In')
    zoom_out_button = widgets.Button(description='Zoom Out')

    def on_prev_click(_):
        start_time_slider.value = max(0, start_time_slider.value - window_size_slider.value)

    def on_next_click(_):
        start_time_slider.value = min(
            total_duration - window_size_slider.value,
            start_time_slider.value + window_size_slider.value
        )

    def on_zoom_in_click(_):
        window_size_slider.value = max(1, window_size_slider.value / 2)

    def on_zoom_out_click(_):
        window_size_slider.value = min(60, window_size_slider.value * 2)

    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)
    zoom_in_button.on_click(on_zoom_in_click)
    zoom_out_button.on_click(on_zoom_out_click)

    nav_buttons = widgets.HBox([prev_button, next_button, zoom_in_button, zoom_out_button])
    time_controls = widgets.VBox([start_time_slider, window_size_slider, nav_buttons])
    widget = widgets.VBox([time_controls, output])

    update_plot(0, initial_window)

    return widget


def compute_region_correlations(data, format_type='old'):
    """
    Compute correlation matrices between channels within and across brain regions.

    Parameters
    ----------
    data : dict
        Data organized by brain regions
    format_type : str
        Data format type

    Returns
    -------
    correlations : dict
        Dictionary containing:
        - 'within_region': correlations within each region
        - 'between_regions': correlations between regions
        - 'region_names': list of region names
    """
    if format_type != 'old':
        raise NotImplementedError("Cross-region correlations only implemented for old format")

    region_names = list(data.keys())
    n_regions = len(region_names)

    # Compute within-region correlations
    within_region = {}
    for region_name in region_names:
        region_data = data[region_name]
        channels = list(region_data.keys())

        if len(channels) < 2:
            within_region[region_name] = None
            continue

        # Extract data arrays
        channel_data = []
        for ch_name in channels:
            channel_data.append(region_data[ch_name]['data'])

        # Compute correlation matrix
        corr_matrix = np.corrcoef(channel_data)
        within_region[region_name] = corr_matrix

    # Compute between-region correlations
    between_regions = np.zeros((n_regions, n_regions))
    region_means = []

    for i, region1 in enumerate(region_names):
        region1_data = data[region1]
        channels1 = list(region1_data.keys())

        # Compute mean signal for region 1
        region1_signals = [region1_data[ch]['data'] for ch in channels1]
        region1_mean = np.mean(region1_signals, axis=0)
        region_means.append(region1_mean)

    # Compute correlations between region means
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                between_regions[i, j] = 1.0  # Self-correlation
            else:
                corr = np.corrcoef(region_means[i], region_means[j])[0, 1]
                between_regions[i, j] = corr

    return {
        'within_region': within_region,
        'between_regions': between_regions,
        'region_names': region_names
    }


def filter_region_data(data, region_name, low_freq=None, high_freq=None,
                      sample_rate=30000, format_type='old'):
    """
    Apply frequency filters to data from a specific brain region.

    Parameters
    ----------
    data : dict
        Data organized by brain regions
    region_name : str
        Name of the region to filter
    low_freq : float, optional
        Low cutoff frequency in Hz
    high_freq : float, optional
        High cutoff frequency in Hz
    sample_rate : float
        Sample rate in Hz
    format_type : str
        Data format type

    Returns
    -------
    filtered_data : dict
        Filtered data for the region
    """
    if format_type != 'old':
        raise NotImplementedError("Region filtering only implemented for old format")

    region_data = get_region_data(data, region_name, format_type)
    filtered_region = {}

    for ch_name, ch_data in region_data.items():
        signal_data = ch_data['data']

        if low_freq is not None and high_freq is not None:
            # Bandpass filter
            filter_type = 'bandpass'
            freqs = [low_freq, high_freq]
        elif low_freq is not None:
            # Highpass filter
            filter_type = 'highpass'
            freqs = low_freq
        elif high_freq is not None:
            # Lowpass filter
            filter_type = 'lowpass'
            freqs = high_freq
        else:
            # No filtering
            filtered_signal = signal_data.copy()
        # Apply filter
        if 'filter_type' in locals():
            b, a = signal.butter(4, freqs, filter_type, fs=sample_rate)
            filtered_signal = signal.filtfilt(b, a, signal_data)

        # Create new channel data structure
        filtered_region[ch_name] = ch_data.copy()
        filtered_region[ch_name]['data'] = filtered_signal

    return filtered_region


def plot_region_overview(data, sample_rate=30000, duration=None, format_type='old'):
    """
    Create an overview plot showing data from all brain regions.

    Parameters
    ----------
    data : dict
        Data organized by brain regions
    sample_rate : float
        Sample rate in Hz
    duration : float, optional
        Duration to plot in seconds. If None, plots entire recording
    format_type : str
        Data format type
    """
    if format_type != 'old':
        raise NotImplementedError("Region overview plotting only implemented for old format")

    region_names = list(data.keys())
    n_regions = len(region_names)

    if n_regions == 0:
        print("No regions found in data")
        return

    # Determine time range
    first_region = region_names[0]
    first_channel = list(data[first_region].keys())[0]
    total_samples = len(data[first_region][first_channel]['data'])

    if duration is not None:
        max_samples = int(duration * sample_rate)
        total_samples = min(total_samples, max_samples)

    time_vector = create_time_vector(total_samples, sample_rate)

    # Create subplot for each region
    fig, axes = plt.subplots(n_regions, 1, figsize=(15, 3*n_regions))
    if n_regions == 1:
        axes = [axes]

    for i, region_name in enumerate(region_names):
        region_data = data[region_name]
        channels = list(region_data.keys())

        # Plot each channel in the region
        for ch_name in channels:
            channel_data = region_data[ch_name]['data']
            if duration is not None:
                channel_data = channel_data[:total_samples]

            # Offset each channel for visibility
            offset = channels.index(ch_name) * 100  # 100 µV offset between channels
            axes[i].plot(time_vector, channel_data + offset, linewidth=0.5, alpha=0.7)

        axes[i].set_ylabel('Voltage (µV)')
        axes[i].set_title(f'{region_name} ({len(channels)} channels)')
        axes[i].grid(True, alpha=0.3)

        # Add channel labels on the right
        for j, ch_name in enumerate(channels):
            offset = j * 100
            axes[i].text(time_vector[-1] + 0.1, offset, ch_name,
                        verticalalignment='center', fontsize=8)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_region_correlation_matrix(correlations):
    """
    Plot correlation matrices for brain regions.

    Parameters
    ----------
    correlations : dict
        Output from compute_region_correlations()
    """
    within_region = correlations['within_region']
    between_regions = correlations['between_regions']
    region_names = correlations['region_names']

    # Plot between-region correlations
    plt.figure(figsize=(10, 8))
    plt.imshow(between_regions, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation coefficient')
    plt.xticks(range(len(region_names)), region_names, rotation=45)
    plt.yticks(range(len(region_names)), region_names)
    plt.title('Between-Region Correlations')
    plt.tight_layout()
    plt.show()

    # Plot within-region correlations
    n_regions = len(region_names)
    fig, axes = plt.subplots(1, n_regions, figsize=(5*n_regions, 4))

    if n_regions == 1:
        axes = [axes]

    for i, region_name in enumerate(region_names):
        corr_matrix = within_region[region_name]
        if corr_matrix is not None:
            im = axes[i].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[i].set_title(f'{region_name} Within-Region')
            plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.tight_layout()
    plt.show()


def detect_swr_in_region(data, region_name, sample_rate=30000,
                        swr_band=(150, 250), threshold=3, format_type='old'):
    """
    Detect sharp wave ripples in a specific brain region.

    Parameters
    ----------
    data : dict
        Data organized by brain regions
    region_name : str
        Name of the region to analyze
    sample_rate : float
        Sample rate in Hz
    swr_band : tuple
        Frequency band for ripple detection (low, high) in Hz
    threshold : float
        Threshold in standard deviations for ripple detection
    format_type : str
        Data format type

    Returns
    -------
    swr_events : dict
        Dictionary containing detected SWR events for each channel
    """
    if format_type != 'old':
        raise NotImplementedError("SWR detection only implemented for old format")

    region_data = get_region_data(data, region_name, format_type)
    swr_events = {}

    for ch_name, ch_data in region_data.items():
        signal_data = ch_data['data']

        # Filter for ripple band
        low_freq, high_freq = swr_band
        b, a = signal.butter(4, [low_freq, high_freq], 'bandpass', fs=sample_rate)
        ripple_signal = signal.filtfilt(b, a, signal_data)

        # Compute envelope
        envelope = np.abs(ripple_signal)
        envelope_smooth = signal.filtfilt(b, a, envelope)  # Additional smoothing

        # Detect events above threshold
        threshold_value = np.mean(envelope_smooth) + threshold * np.std(envelope_smooth)
        above_threshold = envelope_smooth > threshold_value

        # Find event starts and ends
        event_starts = []
        event_ends = []
        in_event = False

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_event:
                event_starts.append(i)
                in_event = True
            elif not above_threshold[i] and in_event:
                event_ends.append(i)
                in_event = False

        # Handle case where event continues to end
        if in_event:
            event_ends.append(len(above_threshold))

        # Convert to time
        event_times = []
        for start, end in zip(event_starts, event_ends):
            duration = (end - start) / sample_rate
            if duration >= 0.02:  # Minimum 20ms duration
                event_times.append({
                    'start_sample': start,
                    'end_sample': end,
                    'start_time': start / sample_rate,
                    'end_time': end / sample_rate,
                    'duration': duration,
                    'peak_amplitude': np.max(envelope_smooth[start:end])
                })

        swr_events[ch_name] = event_times

    return swr_events


def compute_region_synchrony(data, window_size=1.0, sample_rate=30000, format_type='old'):
    """
    Compute synchrony measures within and between brain regions.

    Parameters
    ----------
    data : dict
        Data organized by brain regions
    window_size : float
        Window size for synchrony computation in seconds
    sample_rate : float
        Sample rate in Hz
    format_type : str
        Data format type

    Returns
    -------
    synchrony : dict
        Synchrony measures for each region and between regions
    """
    if format_type != 'old':
        raise NotImplementedError("Synchrony computation only implemented for old format")

    region_names = list(data.keys())
    window_samples = int(window_size * sample_rate)

    synchrony = {
        'within_region': {},
        'between_regions': {}
    }

    # Compute within-region synchrony
    for region_name in region_names:
        region_data = data[region_name]
        channels = list(region_data.keys())

        if len(channels) < 2:
            synchrony['within_region'][region_name] = None
            continue

        # Extract data for all channels in region
        channel_signals = []
        for ch_name in channels:
            channel_signals.append(region_data[ch_name]['data'])

        # Compute pairwise correlations in sliding windows
        n_channels = len(channels)
        n_windows = len(channel_signals[0]) // window_samples

        window_correlations = []

        for w in range(n_windows):
            start = w * window_samples
            end = (w + 1) * window_samples

            window_data = [sig[start:end] for sig in channel_signals]
            corr_matrix = np.corrcoef(window_data)
            # Mean correlation (excluding diagonal)
            mean_corr = (np.sum(corr_matrix) - n_channels) / (n_channels * (n_channels - 1))
            window_correlations.append(mean_corr)

        synchrony['within_region'][region_name] = {
            'correlations': window_correlations,
            'time_vector': np.arange(n_windows) * window_size
        }

    # Compute between-region synchrony
    region_signals = {}
    for region_name in region_names:
        region_data = data[region_name]
        channels = list(region_data.keys())

        # Average signal across channels in region
        region_signal = np.mean([region_data[ch]['data'] for ch in channels], axis=0)
        region_signals[region_name] = region_signal

    # Compute correlations between region pairs
    for i, region1 in enumerate(region_names):
        for j, region2 in enumerate(region_names):
            if i >= j:
                continue

            signal1 = region_signals[region1]
            signal2 = region_signals[region2]

            n_windows = len(signal1) // window_samples
            window_correlations = []

            for w in range(n_windows):
                start = w * window_samples
                end = (w + 1) * window_samples

                corr = np.corrcoef(signal1[start:end], signal2[start:end])[0, 1]
                window_correlations.append(corr)

            key = f'{region1}_vs_{region2}'
            synchrony['between_regions'][key] = {
                'correlations': window_correlations,
                'time_vector': np.arange(n_windows) * window_size
            }

    return synchrony


def create_channel_map(channel_groups, spacing=50):
    """
    Create a 2D map of channel positions for visualization.

    Parameters
    ----------
    channel_groups : dict
        Dictionary of brain regions and their channels
    spacing : float
        Spacing between channels in arbitrary units

    Returns
    -------
    channel_positions : dict
        Dictionary mapping channel names to (x, y) positions
    region_positions : dict
        Dictionary mapping region names to (x, y) positions
    """
    positions = {}
    region_centers = {}

    x_offset = 0
    for region_name, channels in channel_groups.items():
        n_channels = len(channels)

        # Position channels vertically within each region
        y_positions = np.linspace(0, (n_channels - 1) * spacing, n_channels)

        region_channels = []
        for i, ch_num in enumerate(channels):
            ch_name = f'CH{ch_num}'
            positions[ch_name] = (x_offset, y_positions[i])
            region_channels.append(ch_name)

        # Region center
        region_centers[region_name] = (x_offset, np.mean(y_positions))
        x_offset += spacing * 2  # Space between regions

    return positions, region_centers


def plot_channel_map(channel_groups, title="Brain Region Channel Map"):
    """
    Plot a visual map of channels organized by brain regions.

    Parameters
    ----------
    channel_groups : dict
        Dictionary of brain regions and their channels
    title : str
        Plot title
    """
    positions, region_centers = create_channel_map(channel_groups)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot channels
    for ch_name, (x, y) in positions.items():
        ax.plot(x, y, 'bo', markersize=8)
        ax.text(x + 2, y, ch_name, fontsize=10, verticalalignment='center')

    # Plot region labels
    for region_name, (x, y) in region_centers.items():
        ax.text(x - 25, y, region_name, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
               horizontalalignment='center')

    ax.set_xlabel('Arbitrary Units')
    ax.set_ylabel('Arbitrary Units')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def create_brain_region_widget(data_lfp, selected_channels, tetrode_groups, spike_analyzer, regions,
                               region_colors, initial_window=10.0, show_firing_rate=False,
                               firing_rate_kernel_width=0.05, firing_rate_sampling_rate=1000,
                               firing_rate_color='green', firing_rate_alpha=0.8):
    """
    Create a custom interactive widget that maintains the same visual standard as the static plot.

    This widget shows spike raster on top and LFP below for each region, with the same
    color scheme and styling as the static version, but adds time navigation controls.

    Parameters
    ----------
    data_lfp : LazyBinaryLoader
        Data loader instance
    selected_channels : dict
        Selected channels configuration
    tetrode_groups : dict
        Tetrode groups by region
    spike_analyzer : SpikeAnalysis
        Spike analysis instance
    regions : list
        List of brain regions to display
    region_colors : dict
        Color mapping for each region
    initial_window : float
        Initial time window size in seconds
    show_firing_rate : bool, optional
        Whether to plot firing rate between spike raster and LFP (default: False)
    firing_rate_kernel_width : float, optional
        Gaussian kernel width for firing rate smoothing in seconds (default: 0.05)
    firing_rate_sampling_rate : int, optional
        Sampling rate for firing rate calculation in Hz (default: 1000)
    firing_rate_color : str, optional
        Color for firing rate traces (default: 'green'). Note: When using region_colors,
        firing rate will automatically use the same color as the region's LFP and spikes
    firing_rate_alpha : float, optional
        Transparency for firing rate traces (default: 0.8)

    Returns
    -------
    widget : ipywidgets.VBox
        Interactive widget with region-based layout
    """
    import ipywidgets as widgets
    from IPython.display import display

    # Get total duration
    total_duration = data_lfp.duration

    # Create time navigation controls
    start_time_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=max(0.1, total_duration - initial_window),
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

    # Create output widget for the plot
    output = widgets.Output(layout=widgets.Layout(height='600px', overflow='auto'))

    # Initialize update tracking variable
    update_in_progress = False

    def update_plot(start_time, window_size):
        """Update the plot with current time window - PREVENTS DUPLICATE PLOTTING."""
        nonlocal update_in_progress
        if update_in_progress:
            return
        update_in_progress = True

        try:
            with output:
                output.clear_output(wait=True)

                # Create the same visual layout as static plot
                fig = plt.figure(figsize=(12, 10))
                n_regions = len(regions)
        
                # Create GridSpec: 2 or 3 rows per region depending on firing rate option
                if show_firing_rate:
                    rows_per_region = 3
                    height_ratios = [1, 1, 1] * n_regions  # Equal height for LFP, spikes, and firing rate
                else:
                    rows_per_region = 2
                    height_ratios = [1, 1] * n_regions  # Equal height for LFP and spikes
        
                gs = GridSpec(n_regions * rows_per_region, 1, figure=fig,
                              height_ratios=height_ratios,
                              hspace=0.05)

                # Build region to tetrode keys mapping
                region_to_tet_keys = {}
                for region, tets in tetrode_groups.items():
                    region_to_tet_keys[region] = [f"{region}_{tet}" for tet in tets.keys()]

                for i, region in enumerate(regions):
                    region_color = region_colors.get(region, '#1f77b4')

                    # ========== LFP TRACE (BOTTOM) ==========
                    lfp_row = i * rows_per_region
                    ax_lfp = fig.add_subplot(gs[lfp_row])

                    # Remove box outline for cleaner look
                    ax_lfp.spines['top'].set_visible(False)
                    ax_lfp.spines['right'].set_visible(False)
                    ax_lfp.spines['left'].set_visible(False)
                    ax_lfp.spines['bottom'].set_visible(False)

                    tet_keys = region_to_tet_keys.get(region, [])
                    if tet_keys:
                        # Use selected channel for this region
                        tet_key = tet_keys[0]
                        selected_ch = selected_channels.get(tet_key)

                        if selected_ch is not None:
                            # Load data for current time window
                            data = data_lfp.get_trace(selected_ch, start_time, start_time + window_size, target_fs=500)
                            time_vector = np.arange(len(data)) / 500

                            # Plot LFP with region-specific color
                            ax_lfp.plot(time_vector + start_time, data, color=region_color, alpha=0.8, linewidth=1.0)
                            ax_lfp.set_ylabel('Voltage\n(µV)', fontsize=10)
                            ax_lfp.grid(True, alpha=0.3)
                            ax_lfp.tick_params(axis='both', which='major', labelsize=9)
                            ax_lfp.set_xlim(start_time, start_time + window_size)

                            # Show x-axis label only for the last region
                            if i == n_regions - 1:
                                ax_lfp.set_xlabel('Time (s)', fontsize=10)
                            else:
                                ax_lfp.set_xlabel('')
                                ax_lfp.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                        else:
                            ax_lfp.set_title(f'{region} - No LFP Data', color=region_color, fontweight='bold')
                            ax_lfp.text(0.5, 0.5, 'No LFP data', transform=ax_lfp.transAxes,
                                       ha='center', va='center', alpha=0.5)
                            ax_lfp.set_xlim(start_time, start_time + window_size)
                    else:
                        ax_lfp.set_title(f'{region} - No LFP Data', color=region_color, fontweight='bold')
                        ax_lfp.text(0.5, 0.5, 'No LFP data', transform=ax_lfp.transAxes,
                                   ha='center', va='center', alpha=0.5)
                        ax_lfp.set_xlim(start_time, start_time + window_size)

                    # ========== SPIKE RASTER (MIDDLE) ==========
                    spike_row = lfp_row + 1
                    ax_spikes = fig.add_subplot(gs[spike_row])

                    # Remove box outline for cleaner look
                    ax_spikes.spines['top'].set_visible(False)
                    ax_spikes.spines['right'].set_visible(False)
                    ax_spikes.spines['left'].set_visible(False)
                    ax_spikes.spines['bottom'].set_visible(False)

                    tetrodes = spike_analyzer.region_to_tetrodes.get(region, [])
                    if tetrodes:
                        # Pre-allocate lists for better performance
                        spike_times_list = []
                        unit_labels = []

                        # Process all units in this region more efficiently
                        for tetrode in tetrodes:
                            units = spike_analyzer.tetrode_mapping.get(tetrode, [])
                            for unit_idx, unit in enumerate(units):
                                spike_times = unit['spk_time']
                                # Filter spikes within current window
                                window_spikes = spike_times[(spike_times >= start_time) &
                                                          (spike_times <= start_time + window_size)]
                                if len(window_spikes) > 0:
                                    spike_times_list.append(window_spikes)
                                    unit_labels.append(f'{tetrode}_u{unit_idx}')

                        if spike_times_list:
                            # Use eventplot for efficient spike raster plotting
                            ax_spikes.eventplot(spike_times_list, colors=region_color,
                                               lineoffsets=list(range(len(spike_times_list))),
                                               linelengths=0.8, linewidths=0.5)
                            ax_spikes.set_yticks(np.arange(len(unit_labels)))
                            ax_spikes.set_yticklabels(unit_labels, fontsize=8)
                            ax_spikes.set_title(f'{region}', color=region_color, fontweight='bold')
                            ax_spikes.grid(True, alpha=0.3)
                            ax_spikes.set_ylabel('Units', fontsize=10)
                            ax_spikes.tick_params(axis='both', which='major', labelsize=9)
                            ax_spikes.set_xlim(start_time, start_time + window_size)
                        else:
                            ax_spikes.set_title(f'{region} - No Spikes', color=region_color, fontweight='bold')
                            ax_spikes.set_yticks([])
                            ax_spikes.text(0.5, 0.5, 'No spike data', transform=ax_spikes.transAxes,
                                          ha='center', va='center', alpha=0.5)
                            ax_spikes.set_xlim(start_time, start_time + window_size)
                    else:
                        ax_spikes.set_title(f'{region} - No Tetrodes', color=region_color, fontweight='bold')
                        ax_spikes.set_yticks([])
                        ax_spikes.text(0.5, 0.5, 'No tetrode data', transform=ax_spikes.transAxes,
                                      ha='center', va='center', alpha=0.5)
                        ax_spikes.set_xlim(start_time, start_time + window_size)

                    # Remove x-axis labels and ticks for spike plots (except for the last region)
                    if i < n_regions - 1:
                        ax_spikes.set_xlabel('')
                        ax_spikes.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

                    # ========== FIRING RATE (TOP) - Only if enabled ==========
                    if show_firing_rate:
                        firing_rate_row = spike_row + 1
                        ax_firing_rate = fig.add_subplot(gs[firing_rate_row])

                        # Remove box outline for cleaner look
                        ax_firing_rate.spines['top'].set_visible(False)
                        ax_firing_rate.spines['right'].set_visible(False)
                        ax_firing_rate.spines['left'].set_visible(False)
                        ax_firing_rate.spines['bottom'].set_visible(False)

                        # Calculate firing rate for this region and time window
                        tetrodes = spike_analyzer.region_to_tetrodes.get(region, [])
                        if tetrodes:
                            # Get all tetrode identifiers for this region
                            tetrode_ids = [f"{region}_{tet}" for tet in tetrodes]

                            # Use spike_analyzer to compute region-wide firing rate for current window
                            print(f"  Computing firing rate for tetrodes: {tetrode_ids}")
                            firing_rate_results = spike_analyzer.analyze_data(
                                identifiers=tetrode_ids,
                                id_type='tetrode',
                                kernel_width=firing_rate_kernel_width,
                                sampling_rate=firing_rate_sampling_rate,
                                time_range=(start_time, start_time + window_size),
                                full_recording=False
                            )

                            if firing_rate_results['window']['average'] is not None:
                                times = firing_rate_results['window']['times']
                                avg_firing_rate = firing_rate_results['window']['average']
                                print(f"  Plotting firing rate: {len(times)} time points, mean rate: {np.mean(avg_firing_rate):.2f} Hz")

                                # Plot firing rate with region-specific color (same as LFP and spikes)
                                ax_firing_rate.plot(times + start_time, avg_firing_rate, color=region_color,
                                                   alpha=firing_rate_alpha, linewidth=1.5)
                                ax_firing_rate.set_ylabel('Firing Rate\n(Hz)', fontsize=10)
                                ax_firing_rate.grid(True, alpha=0.3)
                                ax_firing_rate.tick_params(axis='both', which='major', labelsize=9)
                                ax_firing_rate.set_xlim(start_time, start_time + window_size)

                                # Show x-axis label only for the last region
                                if i == n_regions - 1:
                                    ax_firing_rate.set_xlabel('Time (s)', fontsize=10)
                                else:
                                    ax_firing_rate.set_xlabel('')
                                    ax_firing_rate.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                            else:
                                ax_firing_rate.text(0.5, 0.5, 'No firing rate data', transform=ax_firing_rate.transAxes,
                                                   ha='center', va='center', alpha=0.5)
                                ax_firing_rate.set_xlim(start_time, start_time + window_size)
                        else:
                            ax_firing_rate.text(0.5, 0.5, 'No tetrode data', transform=ax_firing_rate.transAxes,
                                               ha='center', va='center', alpha=0.5)
                            ax_firing_rate.set_xlim(start_time, start_time + window_size)

                    # Remove box outline for cleaner look
                    ax_lfp.spines['top'].set_visible(False)
                    ax_lfp.spines['right'].set_visible(False)
                    ax_lfp.spines['left'].set_visible(False)
                    ax_lfp.spines['bottom'].set_visible(False)

                    tet_keys = region_to_tet_keys.get(region, [])
                    if tet_keys:
                        # Use selected channel for this region
                        tet_key = tet_keys[0]
                        selected_ch = selected_channels.get(tet_key)

                        if selected_ch is not None:
                            # Load data for current time window
                            data = data_lfp.get_trace(selected_ch, start_time, start_time + window_size, target_fs=500)
                            time_vector = np.arange(len(data)) / 500

                            # Plot LFP with region-specific color
                            ax_lfp.plot(time_vector + start_time, data, color=region_color, alpha=0.8, linewidth=1.0)
                            ax_lfp.set_ylabel('Voltage\n(µV)', fontsize=10)
                            ax_lfp.grid(True, alpha=0.3)
                            ax_lfp.tick_params(axis='both', which='major', labelsize=9)
                            ax_lfp.set_xlim(start_time, start_time + window_size)

                            # Show x-axis label only for the last region
                            if i == n_regions - 1:
                                ax_lfp.set_xlabel('Time (s)', fontsize=10)
                            else:
                                ax_lfp.set_xlabel('')
                                ax_lfp.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                        else:
                            ax_lfp.set_title(f'{region} - No LFP Data', color=region_color, fontweight='bold')
                            ax_lfp.text(0.5, 0.5, 'No LFP data', transform=ax_lfp.transAxes,
                                       ha='center', va='center', alpha=0.5)
                            ax_lfp.set_xlim(start_time, start_time + window_size)
                    else:
                        ax_lfp.set_title(f'{region} - No LFP Data', color=region_color, fontweight='bold')
                        ax_lfp.text(0.5, 0.5, 'No LFP data', transform=ax_lfp.transAxes,
                                   ha='center', va='center', alpha=0.5)
                        ax_lfp.set_xlim(start_time, start_time + window_size)

                plt.tight_layout()
                plt.show()
        finally:
            update_in_progress = False

    # Connect widgets to update function
    def on_value_change(change):
        update_plot(start_time_slider.value, window_size_slider.value)

    start_time_slider.observe(on_value_change, 'value')
    window_size_slider.observe(on_value_change, 'value')

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
    nav_buttons = widgets.HBox([prev_button, next_button, zoom_in_button, zoom_out_button])
    time_controls = widgets.VBox([start_time_slider, window_size_slider, nav_buttons])
    widget = widgets.VBox([time_controls, output])

    # Initial plot
    update_plot(0, initial_window)

    return widget


def summarize_region_data(data, format_type='old'):
    """
    Print a summary of data organized by brain regions.

    Parameters
    ----------
    data : dict
        Data organized by brain regions
    format_type : str
        Data format type
    """
    if format_type != 'old':
        print("Region summary only implemented for old format")
        return

    print("Brain Region Data Summary")
    print("=" * 50)

    total_channels = 0
    for region_name, region_data in data.items():
        n_channels = len(region_data)
        total_channels += n_channels

        # Get sample rate and duration from first channel
        first_channel = next(iter(region_data.values()))
        sample_rate = first_channel['header'].get('sampleRate', 30000)
        n_samples = len(first_channel['data'])
        duration = n_samples / sample_rate

        print(f"\n{region_name}:")
        print(f"  Channels: {n_channels}")
        print(f"  Duration: {format_time(duration)}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Total samples: {n_samples:,}")

        # Show channel names
        channel_names = list(region_data.keys())
        print(f"  Channel names: {', '.join(channel_names[:5])}{'...' if len(channel_names) > 5 else ''}")

    print(f"\nTotal: {len(data)} regions, {total_channels} channels")
