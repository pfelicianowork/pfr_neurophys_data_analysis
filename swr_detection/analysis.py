"""
analysis.py - Statistical analysis tools for SWR detection

This module provides comprehensive statistical analysis tools for SWR events,
including temporal analysis, cross-channel correlations, and advanced metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict


class SWRAnalyzer:
    """
    Statistical analysis tools for SWR detection results.

    This class provides methods for analyzing detected SWR events, including
    temporal patterns, cross-channel relationships, and advanced statistical
    measures.
    """

    def __init__(self, detector):
        """
        Initialize analyzer with SWR detector.

        Parameters
        ----------
        detector : SWRDetector
            SWR detector instance with detected events
        """
        self.detector = detector

    def compute_temporal_statistics(self, time_bins=None, bin_width=60):
        """
        Compute temporal statistics of SWR events.

        Parameters
        ----------
        time_bins : np.ndarray or None
            Custom time bins. If None, creates bins of bin_width seconds
        bin_width : float
            Bin width in seconds for automatic binning

        Returns
        -------
        dict
            Dictionary containing temporal statistics
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get event times
        event_times = [event['peak_time'] for event in self.detector.swr_events]

        if time_bins is None:
            max_time = max(event_times)
            time_bins = np.arange(0, max_time + bin_width, bin_width)

        # Compute histogram
        counts, bin_edges = np.histogram(event_times, bins=time_bins)

        # Compute statistics
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'counts': counts,
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'bin_width': bin_width,
            'total_events': len(event_times),
            'mean_rate': np.mean(counts) / bin_width,
            'max_rate': np.max(counts) / bin_width,
            'event_times': event_times
        }

    def compute_inter_event_intervals(self, channel=None):
        """
        Compute inter-event intervals for events.

        Parameters
        ----------
        channel : int or None
            Channel to analyze. If None, analyzes all channels

        Returns
        -------
        dict
            Dictionary containing interval statistics
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Filter events by channel if specified
        if channel is not None:
            events = [e for e in self.detector.swr_events if e['channel'] == channel]
        else:
            events = self.detector.swr_events

        if len(events) < 2:
            print("Need at least 2 events to compute intervals")
            return None

        # Sort events by time
        events = sorted(events, key=lambda x: x['peak_time'])

        # Compute intervals
        intervals = []
        for i in range(1, len(events)):
            interval = events[i]['peak_time'] - events[i-1]['peak_time']
            intervals.append(interval)

        return {
            'intervals': intervals,
            'mean_interval': np.mean(intervals),
            'std_interval': np.std(intervals),
            'median_interval': np.median(intervals),
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals),
            'n_intervals': len(intervals),
            'channel': channel
        }

    def compute_cross_channel_correlations(self, max_lag=1.0):
        """
        Compute cross-channel correlations between SWR events.

        Parameters
        ----------
        max_lag : float
            Maximum lag in seconds for correlation analysis

        Returns
        -------
        dict
            Dictionary containing cross-channel correlation data
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get unique channels
        channels = sorted(set(e['channel'] for e in self.detector.swr_events))

        if len(channels) < 2:
            print("Need at least 2 channels for cross-channel analysis")
            return None

        # Create event time series for each channel
        channel_times = {}
        for ch in channels:
            ch_events = [e for e in self.detector.swr_events if e['channel'] == ch]
            channel_times[ch] = [e['peak_time'] for e in ch_events]

        # Compute correlations
        correlations = {}
        lags = {}
        max_correlations = {}

        for i, ch1 in enumerate(channels):
            for ch2 in channels[i+1:]:
                # Compute cross-correlation
                times1 = np.array(channel_times[ch1])
                times2 = np.array(channel_times[ch2])

                # Create time series for correlation
                max_time = max(np.max(times1), np.max(times2))
                min_time = min(np.min(times1), np.min(times2))

                # Use binning approach for correlation
                bin_width = 0.01  # 10ms bins
                bins = np.arange(min_time, max_time, bin_width)

                hist1, _ = np.histogram(times1, bins=bins)
                hist2, _ = np.histogram(times2, bins=bins)

                # Compute correlation
                if np.std(hist1) > 0 and np.std(hist2) > 0:
                    corr = np.corrcoef(hist1, hist2)[0, 1]
                else:
                    corr = 0

                correlations[f"{ch1}_vs_{ch2}"] = corr

                # Find optimal lag (simplified)
                lags[f"{ch1}_vs_{ch2}"] = 0  # Placeholder
                max_correlations[f"{ch1}_vs_{ch2}"] = corr

        return {
            'correlations': correlations,
            'lags': lags,
            'max_correlations': max_correlations,
            'channels': channels
        }

    def compute_event_rate_statistics(self, time_window=60):
        """
        Compute event rate statistics over time.

        Parameters
        ----------
        time_window : float
            Time window in seconds for rate calculation

        Returns
        -------
        dict
            Dictionary containing rate statistics
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get all event times
        all_times = [e['peak_time'] for e in self.detector.swr_events]
        max_time = max(all_times)

        # Create time windows
        windows = []
        rates = []

        current_time = 0
        while current_time < max_time:
            window_end = min(current_time + time_window, max_time)

            # Count events in window
            events_in_window = sum(1 for t in all_times if current_time <= t < window_end)
            rate = events_in_window / time_window

            windows.append(current_time)
            rates.append(rate)

            current_time = window_end

        return {
            'windows': windows,
            'rates': rates,
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'max_rate': np.max(rates),
            'min_rate': np.min(rates),
            'time_window': time_window
        }

    def analyze_event_clusters(self, time_threshold=0.5):
        """
        Analyze temporal clustering of SWR events.

        Parameters
        ----------
        time_threshold : float
            Time threshold in seconds for defining clusters

        Returns
        -------
        dict
            Dictionary containing cluster analysis results
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Sort events by time
        sorted_events = sorted(self.detector.swr_events, key=lambda x: x['peak_time'])

        # Find clusters
        clusters = []
        current_cluster = [sorted_events[0]]

        for i in range(1, len(sorted_events)):
            current_event = sorted_events[i]
            last_event = current_cluster[-1]

            if current_event['peak_time'] - last_event['peak_time'] <= time_threshold:
                current_cluster.append(current_event)
            else:
                clusters.append(current_cluster)
                current_cluster = [current_event]

        # Add last cluster
        if current_cluster:
            clusters.append(current_cluster)

        # Analyze clusters
        cluster_sizes = [len(cluster) for cluster in clusters]
        cluster_durations = []
        cluster_intervals = []

        for cluster in clusters:
            if len(cluster) > 1:
                duration = cluster[-1]['peak_time'] - cluster[0]['peak_time']
                cluster_durations.append(duration)

        for i in range(len(clusters) - 1):
            interval = clusters[i+1][0]['peak_time'] - clusters[i][-1]['peak_time']
            cluster_intervals.append(interval)

        return {
            'clusters': clusters,
            'n_clusters': len(clusters),
            'cluster_sizes': cluster_sizes,
            'cluster_durations': cluster_durations,
            'cluster_intervals': cluster_intervals,
            'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': np.max(cluster_sizes) if cluster_sizes else 0,
            'time_threshold': time_threshold
        }

    def compute_burst_statistics(self):
        """
        Compute statistics for SWR bursts (classified events).

        Returns
        -------
        dict
            Dictionary containing burst statistics
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Check if events are classified
        classified_events = [e for e in self.detector.swr_events if 'classification' in e]
        if not classified_events:
            print("Events not yet classified. Run detector.classify_events() first.")
            return None

        # Get burst events (groups with size > 1)
        burst_events = [e for e in classified_events if e['classification']['group_size'] > 1]

        if not burst_events:
            print("No burst events found")
            return None

        # Analyze burst properties
        burst_sizes = [e['classification']['group_size'] for e in burst_events]
        burst_durations = []

        for event in burst_events:
            if 'classification' in event and 'inter_event_intervals' in event['classification']:
                intervals = event['classification']['inter_event_intervals']
                if intervals:
                    # Estimate burst duration from intervals
                    first_event = event
                    # Find other events in the same group
                    group_id = event['classification']['group_id']
                    group_events = [e for e in self.detector.swr_events
                                   if 'classification' in e and
                                   e['classification']['group_id'] == group_id]

                    if len(group_events) > 1:
                        burst_duration = (group_events[-1]['peak_time'] -
                                       group_events[0]['peak_time'])
                        burst_durations.append(burst_duration)

        return {
            'n_bursts': len(burst_events),
            'burst_sizes': burst_sizes,
            'burst_durations': burst_durations,
            'mean_burst_size': np.mean(burst_sizes),
            'max_burst_size': np.max(burst_sizes),
            'mean_burst_duration': np.mean(burst_durations) if burst_durations else 0,
            'burst_events': burst_events
        }

    def analyze_channel_specificity(self):
        """
        Analyze channel-specific properties of SWR events.

        Returns
        -------
        dict
            Dictionary containing channel-specific analysis
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get unique channels
        channels = sorted(set(e['channel'] for e in self.detector.swr_events))

        channel_analysis = {}

        for ch in channels:
            ch_events = [e for e in self.detector.swr_events if e['channel'] == ch]

            if ch_events:
                durations = [e['duration'] for e in ch_events]
                peak_powers = [e['peak_power'] for e in ch_events]
                event_types = [e['event_type'] for e in ch_events]

                channel_analysis[ch] = {
                    'n_events': len(ch_events),
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'mean_peak_power': np.mean(peak_powers),
                    'std_peak_power': np.std(peak_powers),
                    'event_type_counts': {
                        et: event_types.count(et) for et in set(event_types)
                    },
                    'events': ch_events
                }

        return {
            'channel_analysis': channel_analysis,
            'channels': channels,
            'total_channels': len(channels)
        }

    def compute_event_synchrony(self, channel_pairs=None, max_lag=0.1):
        """
        Compute synchrony between channel pairs.

        Parameters
        ----------
        channel_pairs : list of tuples or None
            List of channel pairs to analyze. If None, analyzes all pairs
        max_lag : float
            Maximum lag in seconds for synchrony analysis

        Returns
        -------
        dict
            Dictionary containing synchrony analysis
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get unique channels
        channels = sorted(set(e['channel'] for e in self.detector.swr_events))

        if len(channels) < 2:
            print("Need at least 2 channels for synchrony analysis")
            return None

        # Generate channel pairs if not provided
        if channel_pairs is None:
            channel_pairs = []
            for i in range(len(channels)):
                for j in range(i+1, len(channels)):
                    channel_pairs.append((channels[i], channels[j]))

        synchrony_results = {}

        for ch1, ch2 in channel_pairs:
            # Get events for each channel
            events1 = [e for e in self.detector.swr_events if e['channel'] == ch1]
            events2 = [e for e in self.detector.swr_events if e['channel'] == ch2]

            if not events1 or not events2:
                continue

            times1 = np.array([e['peak_time'] for e in events1])
            times2 = np.array([e['peak_time'] for e in events2])

            # Compute synchrony metrics
            min_time = max(np.min(times1), np.min(times2))
            max_time = min(np.max(times1), np.max(times2))

            if max_time <= min_time:
                continue

            # Simple synchrony measure: correlation of binned event times
            bin_width = 0.01  # 10ms bins
            bins = np.arange(min_time, max_time, bin_width)

            hist1, _ = np.histogram(times1, bins=bins)
            hist2, _ = np.histogram(times2, bins=bins)

            if np.std(hist1) > 0 and np.std(hist2) > 0:
                correlation = np.corrcoef(hist1, hist2)[0, 1]
            else:
                correlation = 0

            synchrony_results[f"{ch1}_vs_{ch2}"] = {
                'correlation': correlation,
                'n_events_ch1': len(events1),
                'n_events_ch2': len(events2),
                'time_range': (min_time, max_time)
            }

        return {
            'synchrony_results': synchrony_results,
            'channel_pairs': channel_pairs,
            'max_lag': max_lag
        }

    def create_analysis_report(self):
        """
        Create a comprehensive analysis report.

        This method performs multiple analyses and prints a summary report.
        """
        if not self.detector.swr_events:
            print("No events detected")
            return

        print("\n" + "="*70)
        print("SWR ANALYSIS REPORT")
        print("="*70)

        # Basic statistics
        stats = self.detector.get_basic_stats()
        if stats:
            print("BASIC STATISTICS:")
            print(f"Total events: {stats['total_events']}")
            print(f"Duration: {stats['duration_stats']['mean']:.3f} ± {stats['duration_stats']['std']:.3f} s")
            print(f"Peak power: {stats['peak_power_stats']['mean']:.2f} ± {stats['peak_power_stats']['std']:.2f}")

            print("EVENT TYPES:")
            for event_type, count in stats['event_type_counts'].items():
                print(f"  {event_type}: {count}")

        # Temporal analysis
        print("TEMPORAL ANALYSIS:")
        temp_stats = self.compute_temporal_statistics()
        if temp_stats:
            print(f"Mean event rate: {temp_stats['mean_rate']:.2f} events/s")
            print(f"Peak event rate: {temp_stats['max_rate']:.2f} events/s")

        # Inter-event intervals
        interval_stats = self.compute_inter_event_intervals()
        if interval_stats:
            print("INTER-EVENT INTERVALS:")
            print(f"Mean interval: {interval_stats['mean_interval']:.3f} s")
            print(f"Interval range: {interval_stats['min_interval']:.3f} - {interval_stats['max_interval']:.3f} s")

        # Channel analysis
        channel_analysis = self.analyze_channel_specificity()
        if channel_analysis:
            print(f"\n🎯 CHANNEL ANALYSIS:")
            print(f"Active channels: {len(channel_analysis['channels'])}")
            for ch, analysis in channel_analysis['channel_analysis'].items():
                print(f"  Channel {ch}: {analysis['n_events']} events, "
                      f"mean duration {analysis['mean_duration']:.3f}s")

        # Cross-channel correlations
        corr_analysis = self.compute_cross_channel_correlations()
        if corr_analysis and corr_analysis['correlations']:
            print("CROSS-CHANNEL CORRELATIONS:")
            correlations = corr_analysis['correlations']
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            for pair, corr in sorted_correlations[:5]:  # Top 5 correlations
                print(f"  {pair}: {corr:.3f}")

        # Burst analysis
        burst_stats = self.compute_burst_statistics()
        if burst_stats:
            print("BURST ANALYSIS:")
            print(f"Number of bursts: {burst_stats['n_bursts']}")
            print(f"Mean burst size: {burst_stats['mean_burst_size']:.1f}")
            if burst_stats['burst_durations']:
                print(f"Mean burst duration: {np.mean(burst_stats['burst_durations']):.3f} s")

            print("\n" + "="*70)

    def export_analysis_results(self, filename, format='csv'):
        """
        Export analysis results to a file.

        Parameters
        ----------
        filename : str
            Output filename
        format : str
            Export format ('csv' or 'json')
        """
        if not self.detector.swr_events:
            print("No events detected")
            return

        # Collect all analysis results
        results = {}

        # Basic statistics
        basic_stats = self.detector.get_basic_stats()
        if basic_stats:
            results['basic_stats'] = basic_stats

        # Temporal statistics
        temp_stats = self.compute_temporal_statistics()
        if temp_stats:
            results['temporal_stats'] = temp_stats

        # Channel analysis
        channel_analysis = self.analyze_channel_specificity()
        if channel_analysis:
            results['channel_analysis'] = channel_analysis

        # Save results
        if format.lower() == 'csv':
            # Convert to DataFrame for CSV export
            summary_data = []
            for analysis_type, data in results.items():
                if analysis_type == 'basic_stats':
                    for stat_type, values in data.items():
                        if isinstance(values, dict):
                            for sub_stat, value in values.items():
                                summary_data.append({
                                    'analysis_type': analysis_type,
                                    'statistic': f"{stat_type}_{sub_stat}",
                                    'value': value
                                })
                        else:
                            summary_data.append({
                                'analysis_type': analysis_type,
                                'statistic': stat_type,
                                'value': values
                            })

            df = pd.DataFrame(summary_data)
            df.to_csv(filename, index=False)

        elif format.lower() == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            raise ValueError("Format must be 'csv' or 'json'")

        print(f"✓ Exported analysis results to {filename}")

    def detect_periodic_patterns(self, min_period=0.1, max_period=10, significance_threshold=0.05):
        """
        Detect periodic patterns in SWR occurrence.

        Parameters
        ----------
        min_period : float
            Minimum period to test in seconds
        max_period : float
            Maximum period to test in seconds
        significance_threshold : float
            Significance threshold for periodicity detection

        Returns
        -------
        dict
            Dictionary containing periodicity analysis results
        """
        if not self.detector.swr_events:
            print("No events detected")
            return None

        # Get event times
        event_times = np.array([e['peak_time'] for e in self.detector.swr_events])

        if len(event_times) < 10:
            print("Need at least 10 events for periodicity analysis")
            return None

        # Test different periods
        periods = np.linspace(min_period, max_period, 100)
        period_scores = []

        for period in periods:
            # Create phase-folded signal
            phases = (event_times % period) / period

            # Compute histogram
            hist, bin_edges = np.histogram(phases, bins=20)

            # Compute uniformity statistic (chi-square test)
            expected = len(event_times) / len(hist)
            chi_square = np.sum((hist - expected) ** 2 / expected)

            # Convert to p-value approximation
            p_value = 1 - stats.chi2.cdf(chi_square, len(hist) - 1)

            period_scores.append({
                'period': period,
                'chi_square': chi_square,
                'p_value': p_value,
                'significance': p_value < significance_threshold
            })

        # Find best periods
        significant_periods = [p for p in period_scores if p['significance']]
        best_period = min(period_scores, key=lambda x: x['p_value'])

        return {
            'period_scores': period_scores,
            'significant_periods': significant_periods,
            'best_period': best_period,
            'n_significant': len(significant_periods),
            'significance_threshold': significance_threshold
        }
