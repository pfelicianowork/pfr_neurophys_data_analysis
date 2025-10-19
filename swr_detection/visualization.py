"""
visualization.py - Visualization tools for SWR detection

This module provides comprehensive visualization tools for SWR events,
including interactive plots, statistical visualizations, and event inspection tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import pandas as pd


class SWRVisualizer:
    """
    Visualization tools for SWR detection results.

    This class provides methods for creating interactive and static visualizations
    of detected SWR events, including event traces, statistical plots, and
    comparative analyses.
    """

    def __init__(self, detector):
        """
        Initialize visualizer with SWR detector.

        Parameters
        ----------
        detector : SWRDetector
            SWR detector instance with detected events
        """
        self.detector = detector

    def plot_event_traces(self, event_id, figsize=(12, 8)):
        """
        Plot detailed traces for a specific SWR event.

        Parameters
        ----------
        event_id : int
            Event ID to visualize
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        event = self.detector.get_event(event_id)
        if event is None:
            return None

        # Determine number of subplots based on available data
        n_plots = 4  # Raw, Ripple, MUA, Context
        if self.detector.velocity_data is not None:
            n_plots += 1

        # Create figure
        fig = plt.figure(figsize=figsize)

        # Plot 1: Raw trace and ripple/MUA
        ax1 = plt.subplot(n_plots, 1, 1)
        ax1.plot(event['trace_timestamps'], event['raw_trace'], 'k', label='Raw')

        if event['sharpwave_trace'] is not None:
            ax1.plot(event['trace_timestamps'], event['sharpwave_trace'], 'g',
                    label='Sharp Wave')

        # Add highlight for event duration
        ax1.axvspan(event['start_time'], event['end_time'],
                    color='yellow', alpha=0.3, label='Event')
        ax1.axvline(x=event['peak_time'], color='r',
                    linestyle='--', label='Peak')
        ax1.set_title(f"Event {event['event_id']} - Channel {event['channel']} "
                      f"(Type: {event['event_type']})")
        ax1.legend()
        ax1.set_ylabel('Amplitude')

        # Plot 2: Ripple trace and power
        ax2 = plt.subplot(n_plots, 1, 2)
        if event['ripple_trace'] is not None:
            ax2.plot(event['trace_timestamps'], event['ripple_trace'], 'b',
                    label='Ripple')

        if event['ripple_power'] is not None:
            ax2.plot(event['trace_timestamps'], event['ripple_power'], 'r',
                    label='Ripple Power')

        ax2.axvspan(event['start_time'], event['end_time'],
                    color='yellow', alpha=0.3, label='Event')
        ax2.axvline(x=event['peak_time'], color='r',
                    linestyle='--', label='Peak')
        ax2.legend()
        ax2.set_ylabel('Amplitude')

        # Plot 3: MUA trace
        ax3 = plt.subplot(n_plots, 1, 3)
        if event['mua_trace'] is not None:
            ax3.plot(event['trace_timestamps'], event['mua_trace'], 'g',
                    label='MUA')
            ax3.axvspan(event['start_time'], event['end_time'],
                        color='yellow', alpha=0.3, label='Event')
            ax3.axvline(x=event['peak_time'], color='r',
                        linestyle='--', label='Peak')
            ax3.legend()
            ax3.set_ylabel('Spike Rate')
            ax3.set_title('Multi-Unit Activity')

        # Plot 4: Context
        ax4 = plt.subplot(n_plots, 1, 4)
        context_window = 0.5  # 500ms window
        peak_idx = int(event['peak_time'] * self.detector.fs)
        context_start = max(0, int(peak_idx - self.detector.fs * context_window / 2))
        context_end = min(self.detector.n_timepoints, int(peak_idx + self.detector.fs * context_window / 2))
        time_context = np.arange(context_start, context_end) / self.detector.fs

        if isinstance(event['channel'], int):
            signal_context = self.detector.lfp_data[event['channel'], context_start:context_end]
        else:
            signal_context = np.mean(self.detector.lfp_data[:, context_start:context_end], axis=0)

        ax4.plot(time_context, signal_context, 'k', label='Signal')
        ax4.axvspan(event['start_time'], event['end_time'],
                    color='yellow', alpha=0.3, label='Event')
        ax4.axvline(x=event['peak_time'], color='r',
                    linestyle='--', label='Peak')
        ax4.set_title('Event Context (±250ms)')
        ax4.set_ylabel('Amplitude')
        ax4.legend()

        # Plot 5: Velocity data if available
        if self.detector.velocity_data is not None:
            ax5 = plt.subplot(n_plots, 1, 5)
            velocity_trace = self.detector.velocity_data[context_start:context_end]
            ax5.plot(time_context, velocity_trace, 'b', label='Velocity')

            if self.detector.params.velocity_threshold is not None:
                ax5.axhline(y=self.detector.params.velocity_threshold, color='r',
                            linestyle='--',
                            label=f'Threshold ({self.detector.params.velocity_threshold} cm/s)')

            ax5.axvspan(event['start_time'], event['end_time'],
                        color='yellow', alpha=0.3)
            ax5.axvline(x=event['peak_time'], color='r',
                        linestyle='--', label='Peak')
            ax5.set_title('Velocity')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Velocity (cm/s)')
            ax5.legend()

        plt.tight_layout()
        return fig

    def create_interactive_event_browser(self):
        """
        Create an interactive browser for exploring detected SWR events.

        This method creates a widget-based interface for browsing through
        detected events with navigation controls and detailed visualizations.
        """
        if not self.detector.swr_events:
            print("No events detected")
            return

        # Create navigation widgets
        prev_button = widgets.Button(
            description='Previous',
            tooltip='Previous event',
            icon='arrow-left'
        )

        next_button = widgets.Button(
            description='Next',
            tooltip='Next event',
            icon='arrow-right'
        )

        event_input = widgets.BoundedIntText(
            value=0,
            min=0,
            max=len(self.detector.swr_events) - 1,
            description='Event #:',
            style={'description_width': 'initial'}
        )

        # Create output widget for plots and text
        out = widgets.Output()

        def plot_current_event(event_idx):
            """Plot the current event and display details."""
            with out:
                out.clear_output(wait=True)

                # Get event
                event = self.detector.swr_events[event_idx]

                # Create the plot
                fig = self.plot_event_traces(event['event_id'])

                if fig is not None:
                    plt.show()

                # Print event details
                print(f"\nEvent {event['event_id']} Details:")
                print(f"Channel: {event['channel']}")
                print(f"Event type: {event['event_type']}")
                print(f"Start time: {event['start_time']:.3f} s")
                print(f"Peak time: {event['peak_time']:.3f} s")
                print(f"End time: {event['end_time']:.3f} s")
                print(f"Duration: {(event['end_time'] - event['start_time']) * 1000:.1f} ms")
                print(f"Peak power: {event['peak_power']:.2f}")
                print(f"Number of peaks: {len(event['peak_times'])}")

                # Add classification information if available
                if 'classification' in event:
                    print("Classification Details:")
                    print(f"Group type: {event['classification']['group_type']}")
                    print(f"Group size: {event['classification']['group_size']}")
                    print(f"Position in group: {event['classification']['position_in_group']}")
                    if event['classification']['inter_event_intervals']:
                        intervals_str = ", ".join([f"{x * 1000:.1f}ms" for x in event['classification']['inter_event_intervals']])
                        print(f"Inter-event intervals: {intervals_str}")

        def on_prev_clicked(b):
            """Handle previous button click."""
            event_input.value = max(0, event_input.value - 1)

        def on_next_clicked(b):
            """Handle next button click."""
            event_input.value = min(len(self.detector.swr_events) - 1, event_input.value + 1)

        def on_value_change(change):
            """Handle event input value change."""
            plot_current_event(change['new'])

        # Connect event handlers
        prev_button.on_click(on_prev_clicked)
        next_button.on_click(on_next_clicked)
        event_input.observe(on_value_change, names='value')

        # Create layout
        buttons = widgets.HBox([prev_button, event_input, next_button])

        # Display interface
        display(widgets.VBox([buttons, out]))

        # Show initial plot
        plot_current_event(0)

        # Print instructions
        print("\nNavigation Controls:")
        print("- Click 'Previous' and 'Next' buttons")
        print("- Type event number directly in the input box")

    def plot_events_summary(self, figsize=(15, 10)):
        """
        Create a comprehensive summary plot of all detected events.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get events summary
        summary_df = self.detector.get_events_summary()
        if summary_df is None:
            return None

        fig = plt.figure(figsize=figsize)

        # Plot 1: Event timeline
        ax1 = plt.subplot(2, 3, 1)
        for _, event in summary_df.iterrows():
            ax1.plot([event['start_time'], event['end_time']],
                    [event['channel'], event['channel']], 'b-', alpha=0.7)
            ax1.scatter(event['peak_time'], event['channel'],
                       c='r', s=20, zorder=5)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Channel')
        ax1.set_title('Event Timeline')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Duration distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(summary_df['duration'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Duration (s)')
        ax2.set_ylabel('Count')
        ax2.set_title('Event Duration Distribution')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Event type distribution
        ax3 = plt.subplot(2, 3, 3)
        event_types = summary_df['event_type'].value_counts()
        ax3.pie(event_types.values, labels=event_types.index, autopct='%1.1f%%')
        ax3.set_title('Event Type Distribution')

        # Plot 4: Peak power distribution
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(summary_df['peak_power'], bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Peak Power')
        ax4.set_ylabel('Count')
        ax4.set_title('Peak Power Distribution')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Events per channel
        ax5 = plt.subplot(2, 3, 5)
        channel_counts = summary_df['channel'].value_counts().sort_index()
        ax5.bar(channel_counts.index, channel_counts.values, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Channel')
        ax5.set_ylabel('Event Count')
        ax5.set_title('Events per Channel')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Duration vs Peak Power
        ax6 = plt.subplot(2, 3, 6)
        scatter = ax6.scatter(summary_df['duration'], summary_df['peak_power'],
                            c=summary_df['channel'], cmap='viridis', alpha=0.6, s=30)
        ax6.set_xlabel('Duration (s)')
        ax6.set_ylabel('Peak Power')
        ax6.set_title('Duration vs Peak Power')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label='Channel')

        plt.tight_layout()
        return fig

    def plot_raster_plot(self, channels=None, time_range=None, figsize=(14, 8)):
        """
        Create a raster plot of SWR events.

        Parameters
        ----------
        channels : list or None
            Channels to include. If None, includes all channels
        time_range : tuple or None
            Time range (start, end) in seconds. If None, uses full recording
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Filter events if criteria provided
        events = self.detector.swr_events
        if channels is not None:
            events = [e for e in events if e['channel'] in channels]
        if time_range is not None:
            events = [e for e in events if time_range[0] <= e['peak_time'] <= time_range[1]]

        if not events:
            print("No events match the specified criteria")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot events
        y_positions = []
        colors = []

        for event in events:
            y_pos = event['channel']
            y_positions.append(y_pos)

            # Color by event type
            if event['event_type'] == 'ripple_only':
                colors.append('blue')
            elif event['event_type'] == 'mua_only':
                colors.append('red')
            elif event['event_type'] == 'ripple_mua':
                colors.append('purple')
            else:
                colors.append('gray')

        # Create scatter plot
        scatter = ax.scatter([e['peak_time'] for e in events],
                           y_positions, c=colors, alpha=0.7, s=20)

        # Customize plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')
        ax.set_title('SWR Events Raster Plot')
        ax.grid(True, alpha=0.3)

        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                      markersize=8, label='Ripple Only'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=8, label='MUA Only'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                      markersize=8, label='Ripple + MUA')
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        return fig

    def plot_event_statistics(self, figsize=(12, 8)):
        """
        Create statistical plots for detected events.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get basic statistics
        stats = self.detector.get_basic_stats()
        if stats is None:
            return None

        fig = plt.figure(figsize=figsize)

        # Plot 1: Event type distribution
        ax1 = plt.subplot(2, 2, 1)
        event_types = list(stats['event_type_counts'].keys())
        counts = list(stats['event_type_counts'].values())
        ax1.pie(counts, labels=event_types, autopct='%1.1f%%')
        ax1.set_title('Event Type Distribution')

        # Plot 2: Duration statistics
        ax2 = plt.subplot(2, 2, 2)
        durations = [e['duration'] for e in self.detector.swr_events]
        ax2.hist(durations, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(stats['duration_stats']['mean'], color='red', linestyle='--',
                   label=f'Mean: {stats["duration_stats"]["mean"]:.3f}s')
        ax2.set_xlabel('Duration (s)')
        ax2.set_ylabel('Count')
        ax2.set_title('Event Duration Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Peak power statistics
        ax3 = plt.subplot(2, 2, 3)
        peak_powers = [e['peak_power'] for e in self.detector.swr_events]
        ax3.hist(peak_powers, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(stats['peak_power_stats']['mean'], color='red', linestyle='--',
                   label=f'Mean: {stats["peak_power_stats"]["mean"]:.2f}')
        ax3.set_xlabel('Peak Power')
        ax3.set_ylabel('Count')
        ax3.set_title('Peak Power Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Duration vs Peak Power scatter
        ax4 = plt.subplot(2, 2, 4)
        colors = []
        for event in self.detector.swr_events:
            if event['event_type'] == 'ripple_only':
                colors.append('blue')
            elif event['event_type'] == 'mua_only':
                colors.append('red')
            elif event['event_type'] == 'ripple_mua':
                colors.append('purple')
            else:
                colors.append('gray')

        ax4.scatter(durations, peak_powers, c=colors, alpha=0.6, s=30)
        ax4.set_xlabel('Duration (s)')
        ax4.set_ylabel('Peak Power')
        ax4.set_title('Duration vs Peak Power')
        ax4.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                      markersize=8, label='Ripple Only'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=8, label='MUA Only'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                      markersize=8, label='Ripple + MUA')
        ]
        ax4.legend(handles=legend_elements)

        plt.tight_layout()
        return fig

    def plot_channel_comparison(self, channels, figsize=(14, 6)):
        """
        Compare SWR events across different channels.

        Parameters
        ----------
        channels : list
            List of channel indices to compare
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Filter events for specified channels
        channel_events = []
        for ch in channels:
            events = self.detector.get_channel_events(ch)
            channel_events.extend(events)

        if not channel_events:
            print("No events found for specified channels")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Event count comparison
        channel_counts = {}
        channel_durations = {}
        channel_powers = {}

        for ch in channels:
            ch_events = [e for e in channel_events if e['channel'] == ch]
            channel_counts[ch] = len(ch_events)
            if ch_events:
                channel_durations[ch] = np.mean([e['duration'] for e in ch_events])
                channel_powers[ch] = np.mean([e['peak_power'] for e in ch_events])
            else:
                channel_durations[ch] = 0
                channel_powers[ch] = 0

        ax1.bar(channels, [channel_counts[ch] for ch in channels],
               alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Event Count')
        ax1.set_title('Event Count by Channel')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Duration and power comparison
        x_pos = np.arange(len(channels))
        width = 0.35

        ax2.bar(x_pos - width/2, [channel_durations[ch] for ch in channels],
               width, label='Mean Duration (s)', alpha=0.7)
        ax2.bar(x_pos + width/2, [channel_powers[ch] for ch in channels],
               width, label='Mean Peak Power', alpha=0.7)

        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Value')
        ax2.set_title('Duration and Power by Channel')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(channels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_classification_summary(self, figsize=(12, 6)):
        """
        Plot classification summary if events are classified.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Check if events are classified
        classified_events = [e for e in self.detector.swr_events if 'classification' in e]
        if not classified_events:
            print("Events not yet classified. Run detector.classify_events() first.")
            return None

        # Get classification data
        group_types = [e['classification']['group_type'] for e in classified_events]
        group_sizes = [e['classification']['group_size'] for e in classified_events]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Group type distribution
        group_type_counts = {}
        for gt in group_types:
            group_type_counts[gt] = group_type_counts.get(gt, 0) + 1

        ax1.pie(list(group_type_counts.values()),
               labels=list(group_type_counts.keys()),
               autopct='%1.1f%%')
        ax1.set_title('Event Group Type Distribution')

        # Plot 2: Group size distribution
        unique_sizes = sorted(set(group_sizes))
        size_counts = [group_sizes.count(size) for size in unique_sizes]

        ax2.bar([str(s) for s in unique_sizes], size_counts, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Group Size')
        ax2.set_ylabel('Count')
        ax2.set_title('Event Group Size Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_detection_summary_report(self):
        """
        Create a comprehensive summary report of the detection results.

        This method prints detailed statistics and information about
        the detected SWR events.
        """
        if not self.detector.swr_events:
            print("No events detected")
            return

        print("\n" + "="*60)
        print("SWR DETECTION SUMMARY REPORT")
        print("="*60)

        # Basic statistics
        stats = self.detector.get_basic_stats()
        if stats:
            print("BASIC STATISTICS:")
            print(f"Total events detected: {stats['total_events']}")
            print(f"Duration: {stats['duration_stats']['mean']:.3f} ± {stats['duration_stats']['std']:.3f} s")
            print(f"Peak power: {stats['peak_power_stats']['mean']:.2f} ± {stats['peak_power_stats']['std']:.2f}")

            print("EVENT TYPE DISTRIBUTION:")
            for event_type, count in stats['event_type_counts'].items():
                percentage = (count / stats['total_events']) * 100
                print(f"  {event_type}: {count} ({percentage:.1f}%)")

        # Channel information
        channels_with_events = set(e['channel'] for e in self.detector.swr_events)
        print(f"\n🎯 CHANNELS WITH EVENTS: {len(channels_with_events)}")
        print(f"Channels: {sorted(channels_with_events)}")

        # Classification information
        classified_events = [e for e in self.detector.swr_events if 'classification' in e]
        if classified_events:
            print(f"\n🏷️  CLASSIFICATION RESULTS:")
            group_types = {}
            for event in classified_events:
                gt = event['classification']['group_type']
                group_types[gt] = group_types.get(gt, 0) + 1

            for group_type, count in group_types.items():
                percentage = (count / len(classified_events)) * 100
                print(f"  {group_type}: {count} ({percentage:.1f}%)")

        # Parameter summary
        print("DETECTION PARAMETERS:")
        print(f"Ripple band: {self.detector.params.ripple_band} Hz")
        print(f"Threshold multiplier: {self.detector.params.threshold_multiplier} SD")
        print(f"Duration limits: {self.detector.params.min_duration}-{self.detector.params.max_duration} s")
        print(f"MUA detection: {'Enabled' if self.detector.params.enable_mua else 'Disabled'}")
        print(f"HMM edge detection: {'Enabled' if self.detector.params.use_hmm_edge_detection else 'Disabled'}")

        # Data information
        print("DATA INFORMATION:")
        print(f"Recording duration: {self.detector.n_timepoints / self.detector.fs:.2f} s")
        print(f"Sampling rate: {self.detector.fs} Hz")
        print(f"Number of channels: {self.detector.n_channels}")
        print(f"MUA data: {'Available' if self.detector.mua_data is not None else 'Not available'}")
        print(f"Velocity data: {'Available' if self.detector.velocity_data is not None else 'Not available'}")

        print("\n" + "="*60)

    def export_event_details(self, filename, format='csv'):
        """
        Export detailed event information to a file.

        Parameters
        ----------
        filename : str
            Output filename
        format : str
            Export format ('csv' or 'excel')
        """
        summary_df = self.detector.get_events_summary()
        if summary_df is None:
            return

        if format.lower() == 'csv':
            summary_df.to_csv(filename, index=False)
        elif format.lower() == 'excel':
            summary_df.to_excel(filename, index=False)
        else:
            raise ValueError("Format must be 'csv' or 'excel'")

        print(f"✓ Exported {len(summary_df)} events to {filename}")

    def plot_event_timeline(self, channels=None, time_range=None, figsize=(14, 6)):
        """
        Plot a timeline view of SWR events.

        Parameters
        ----------
        channels : list or None
            Channels to include
        time_range : tuple or None
            Time range (start, end) in seconds
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Filter events
        events = self.detector.swr_events
        if channels is not None:
            events = [e for e in events if e['channel'] in channels]
        if time_range is not None:
            events = [e for e in events if time_range[0] <= e['peak_time'] <= time_range[1]]

        if not events:
            print("No events match the specified criteria")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Plot event bars
        for event in events:
            color = 'blue' if event['event_type'] == 'ripple_only' else 'red'
            ax.barh(event['channel'], event['duration'],
                   left=event['start_time'], height=0.6,
                   color=color, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')
        ax.set_title('SWR Events Timeline')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def swr_events_widget_multi(
    detectors_by_region,
    region_lfp,
    fs: float,
    params,
    mua_by_region=None,
    spike_times_by_region=None,
    velocity: np.ndarray | None = None,
    default_region=None,
    default_channel="avg",
    window_sec: float | None = None,
):
    """
    Interactive widget to browse SWR events across regions and channels.

        Rows (shared x):
            1) LFP
            2) Ripple-filtered LFP
            3) Ripple envelope with threshold lines
            4) Spike raster (optional; per-unit spikes, if provided)
            5) MUA firing rate (region-level)
            6) Velocity (if provided)

    Navigation: Prev/Next buttons and a numeric event index box.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    try:
        from scipy.signal import butter, filtfilt, hilbert
    except Exception:
        butter = filtfilt = hilbert = None

    if not detectors_by_region:
        raise ValueError("detectors_by_region is empty")

    regions = list(detectors_by_region.keys())
    if default_region is None:
        default_region = regions[0]

    # Controls
    dd_region = widgets.Dropdown(options=regions, value=default_region, description="Region:")
    dd_channel = widgets.Dropdown(options=["avg"], value=(default_channel if default_channel in ["avg"] else "avg"), description="Channel:")
    wsec = float(getattr(params, "trace_window", 0.6 if window_sec is None else window_sec))
    sl_win = widgets.FloatSlider(value=wsec, min=0.2, max=1.5, step=0.05, description="Win (s)", continuous_update=False)
    btn_prev = widgets.Button(description="Prev", icon="arrow-left")
    btn_next = widgets.Button(description="Next", icon="arrow-right")
    event_box = widgets.BoundedIntText(value=0, min=0, max=0, step=1, description="Event")
    info = widgets.HTML("")
    out = widgets.Output()

    def _bandpass_and_envelope(x, band):
        if butter is None or filtfilt is None or hilbert is None:
            # Fallback: no filtering, approximate envelope
            return x, np.abs(x)
        lo, hi = float(band[0]), float(band[1])
        nyq = 0.5 * fs
        b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
        xr = filtfilt(b, a, np.asarray(x, dtype=float))
        env = np.abs(hilbert(xr))
        return xr, env

    def _robust_thr(env, mult):
        med = np.nanmedian(env)
        mad = np.nanmedian(np.abs(env - med)) + 1e-12
        sigma = mad / 0.6745
        return float(med + mult * sigma)

    def _update_channels(*_):
        reg = dd_region.value
        lfp = region_lfp.get(reg)
        if lfp is None:
            dd_channel.options = ["avg"]
            dd_channel.value = "avg"
        else:
            if lfp.ndim == 1:
                dd_channel.options = ["avg"]
                dd_channel.value = "avg"
            else:
                n = int(lfp.shape[0])
                opts = ["avg"] + [f"ch{i}" for i in range(n)]
                prev = dd_channel.value
                dd_channel.options = opts
                dd_channel.value = prev if prev in opts else "avg"

        # Update event bounds for this region
        det = detectors_by_region[reg]
        event_box.min = 0
        event_box.max = max(0, len(det.swr_events) - 1)
        event_box.value = 0

    def _get_channel_trace(reg, chan_sel):
        lfp = region_lfp.get(reg)
        if lfp is None:
            return None
        arr = np.asarray(lfp)
        if arr.ndim == 1 or chan_sel == "avg":
            return np.nanmean(np.atleast_2d(arr), axis=0)
        try:
            ci = int(str(chan_sel).replace("ch", ""))
            return arr[ci, :]
        except Exception:
            return np.nanmean(np.atleast_2d(arr), axis=0)

    def _plot(*_):
        out.clear_output(wait=True)
        with out:
            reg = dd_region.value
            det = detectors_by_region[reg]
            events = det.swr_events
            if not events:
                print(f"No events in region {reg}")
                return

            trace = _get_channel_trace(reg, dd_channel.value)
            if trace is None:
                print(f"No LFP for region {reg}")
                return

            idx = int(np.clip(event_box.value, event_box.min, event_box.max))
            ev = events[idx]

            fs_local = float(fs)
            half = max(1, int(0.5 * float(sl_win.value) * fs_local))
            peak_idx = int(round(float(ev.get("peak_time", 0.0)) * fs_local))
            a = max(0, peak_idx - half)
            b = min(trace.size, peak_idx + half)
            if b <= a:
                b = min(trace.size, a + max(1, int(0.4 * fs_local)))
            tvec = np.arange(a, b, dtype=float) / fs_local
            raw_snip = np.asarray(trace, dtype=float)[a:b]

            ripple_band = getattr(det.params, "ripple_band", (150.0, 250.0))
            xr, env = _bandpass_and_envelope(raw_snip, ripple_band)
            thr_mult = float(getattr(det.params, "threshold_multiplier", 3.0))
            thr = _robust_thr(env, thr_mult)
            thr2 = None
            # Use detector-provided thresholds if available
            if hasattr(det, "ripple_env_threshold") and det.ripple_env_threshold is not None:
                thr = float(det.ripple_env_threshold)
            if hasattr(det, "ripple_env_threshold_high") and det.ripple_env_threshold_high is not None:
                thr2 = float(det.ripple_env_threshold_high)

            # MUA segment
            mua_snip = None
            if isinstance(mua_by_region, dict) and reg in mua_by_region and mua_by_region[reg] is not None:
                mv = np.asarray(mua_by_region[reg], dtype=float).squeeze()
                if mv.ndim == 1 and mv.size >= b:
                    mua_snip = mv[a:b]

            # Velocity segment
            vel_snip = None
            if velocity is not None:
                vv = np.asarray(velocity, dtype=float).squeeze()
                if vv.ndim == 1 and vv.size >= b:
                    vel_snip = vv[a:b]

            # Figure with 6 rows (some panels may show 'No data')
            fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
            ax_lfp, ax_ripple, ax_env, ax_spk, ax_mua, ax_vel = axes

            # Row1: Raw LFP only
            ax_lfp.plot(tvec, raw_snip, color="#444", lw=0.9, label="LFP")
            ax_lfp.axvline(ev.get("start_time", np.nan), color="k", ls="--", lw=1)
            ax_lfp.axvline(ev.get("end_time", np.nan), color="k", ls="--", lw=1)
            ax_lfp.set_ylabel("LFP")
            ax_lfp.legend(fontsize=8, frameon=False)

            # Row2: Ripple-filtered LFP
            ax_ripple.plot(tvec, xr, color="#1f77b4", lw=1.0, alpha=0.95, label=f"Ripple {ripple_band[0]}-{ripple_band[1]} Hz")
            ax_ripple.axvline(ev.get("start_time", np.nan), color="k", ls="--", lw=1)
            ax_ripple.axvline(ev.get("end_time", np.nan), color="k", ls="--", lw=1)
            ax_ripple.set_ylabel("Ripple")
            ax_ripple.legend(fontsize=8, frameon=False)

            # Row2: Envelope + thresholds
            ax_env.plot(tvec, env, color="#ff7f0e", lw=1.0, label="Envelope")
            ax_env.axhline(thr, color="#d62728", ls="--", lw=1.0, label=f"Thresh x{thr_mult:g}")
            if thr2 is not None:
                ax_env.axhline(thr2, color="#9467bd", ls="--", lw=1.0, label="Thresh2")
            ax_env.axvline(ev.get("start_time", np.nan), color="k", ls="--", lw=1)
            ax_env.axvline(ev.get("end_time", np.nan), color="k", ls="--", lw=1)
            ax_env.set_ylabel("Env (a.u.)")
            ax_env.legend(fontsize=8, frameon=False)

            # Row4: Spike raster (optional)
            drew_spikes = False
            if isinstance(spike_times_by_region, dict) and reg in spike_times_by_region and spike_times_by_region[reg] is not None:
                # Normalize to list of 1D arrays (seconds). If dict, preserve insertion order for labels.
                unit_spikes_raw = spike_times_by_region[reg]
                if isinstance(unit_spikes_raw, dict):
                    unit_labels = list(unit_spikes_raw.keys())
                    unit_spikes_list = [np.asarray(unit_spikes_raw[k], dtype=float).ravel() for k in unit_labels]
                else:
                    unit_spikes_list = [np.asarray(st, dtype=float).ravel() for st in np.atleast_1d(unit_spikes_raw)]
                    unit_labels = [str(i) for i in range(len(unit_spikes_list))]

                # Slice to window [t0, t1]
                t0, t1 = (tvec[0], tvec[-1]) if tvec.size > 0 else (0.0, 0.0)
                # Optional cap to avoid heavy plots
                max_units = 120
                unit_spikes_win = []
                kept_labels = []
                for i, st in enumerate(unit_spikes_list[:max_units]):
                    if st.size == 0:
                        unit_spikes_win.append([])
                        kept_labels.append(unit_labels[i] if i < len(unit_labels) else str(i))
                        continue
                    # Efficient selection assuming unsorted okay; sort indices via mask
                    mask = (st >= t0) & (st <= t1)
                    sw = st[mask]
                    unit_spikes_win.append(sw.tolist())
                    kept_labels.append(unit_labels[i] if i < len(unit_labels) else str(i))

                # Check if any spikes
                n_spk = sum(len(s) for s in unit_spikes_win)
                if n_spk > 0:
                    # Use eventplot; one row per unit
                    line_offsets = np.arange(len(unit_spikes_win))
                    ax_spk.eventplot(unit_spikes_win, colors="#000000", lineoffsets=line_offsets, linelengths=0.8, linewidths=0.6)
                    ax_spk.set_ylim(-0.5, len(unit_spikes_win) - 0.5)
                    ax_spk.set_yticks([])  # keep clean; could add sparse labels if needed
                    ax_spk.set_ylabel("Spikes")
                    ax_spk.axvline(ev.get("start_time", np.nan), color="k", ls="--", lw=1)
                    ax_spk.axvline(ev.get("end_time", np.nan), color="k", ls="--", lw=1)
                    drew_spikes = True

            if not drew_spikes:
                ax_spk.text(0.5, 0.5, "No spikes", ha="center", va="center", transform=ax_spk.transAxes)
                ax_spk.set_ylabel("Spikes")
                ax_spk.axvline(ev.get("start_time", np.nan), color="k", ls="--", lw=1)
                ax_spk.axvline(ev.get("end_time", np.nan), color="k", ls="--", lw=1)

            # Row3: MUA firing rate
            if mua_snip is not None:
                ax_mua.plot(tvec, mua_snip, color="#2ca02c", lw=1.0)
                ax_mua.set_ylabel("MUA (Hz)")
            else:
                ax_mua.text(0.5, 0.5, "No MUA", ha="center", va="center", transform=ax_mua.transAxes)
                ax_mua.set_ylabel("MUA")
            ax_mua.axvline(ev.get("start_time", np.nan), color="k", ls="--", lw=1)
            ax_mua.axvline(ev.get("end_time", np.nan), color="k", ls="--", lw=1)

            # Row4: Velocity
            if vel_snip is not None:
                ax_vel.plot(tvec, vel_snip, color="#7f7f7f", lw=1.0)
                # Optional velocity threshold line
                if getattr(params, "velocity_threshold", None) is not None:
                    ax_vel.axhline(float(params.velocity_threshold), color="#d62728", ls="--", lw=1.0)
                ax_vel.set_ylabel("Vel")
            else:
                ax_vel.text(0.5, 0.5, "No velocity", ha="center", va="center", transform=ax_vel.transAxes)
                ax_vel.set_ylabel("Vel")
            ax_vel.axvline(ev.get("start_time", np.nan), color="k", ls="--", lw=1)
            ax_vel.axvline(ev.get("end_time", np.nan), color="k", ls="--", lw=1)
            ax_vel.set_xlabel("Time (s)")

            # Despine helper: keep only left and bottom spines
            def _despine(ax):
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_visible(True)
                ax.yaxis.set_ticks_position("left")
                ax.xaxis.set_ticks_position("bottom")
                ax.tick_params(axis="both", direction="out")

            for a in (ax_lfp, ax_ripple, ax_env, ax_spk, ax_mua, ax_vel):
                _despine(a)

            info.value = f"<b>{reg}</b> | ch={dd_channel.value} | event {idx+1}/{event_box.max+1}"
            plt.tight_layout()
            plt.show()

    def _on_prev(_):
        if event_box.max <= 0:
            return
        event_box.value = int(max(event_box.min, event_box.value - 1))

    def _on_next(_):
        if event_box.max <= 0:
            return
        event_box.value = int(min(event_box.max, event_box.value + 1))

    # Wire events
    dd_region.observe(lambda c: (_update_channels(), _plot()), names="value")
    dd_channel.observe(lambda c: _plot(), names="value")
    sl_win.observe(lambda c: _plot(), names="value")
    event_box.observe(lambda c: _plot(), names="value")
    btn_prev.on_click(_on_prev)
    btn_next.on_click(_on_next)

    # Init
    _update_channels()
    _plot()

    controls = widgets.HBox([dd_region, dd_channel, sl_win, btn_prev, btn_next, event_box, info])
    ui = widgets.VBox([controls, out])
    display(ui)
    return ui