import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from sklearn.cluster import DBSCAN  # For improved clustering (if used)
from hmmlearn.hmm import GaussianHMM  # NEW: For HMM-based edge detection

class SWRHMMParams:
    def __init__(self,
                 # Required parameters
                 ripple_band=(150, 250),
                 threshold_multiplier=3,
                 min_duration=0.03,
                 max_duration=0.4,
                 # Optional parameters
                 notch_freq=60,
                 sharpwave_band=None,  # Set to None to disable sharp wave detection
                 velocity_threshold=None,  # Set to None to disable velocity filtering
                 trace_window=0.2,
                 # Event detection parameters
                 duration_std_threshold=0.1,
                 min_event_separation=0.05,
                 merge_threshold=0.8,
                 # Classification parameters
                 single_separation=0.2,      # 200ms minimum separation for singles
                 burst_min_interval=0.07,    # 70ms minimum between burst events
                 burst_max_interval=0.2,     # 200ms maximum between burst events
                 merge_interval=0.07,        # Merge events closer than 70ms
                 # MUA parameters
                 mua_threshold_multiplier=2.5,  # Threshold for MUA detection in SD
                 mua_min_duration=0.02,         # Minimum duration for MUA events
                 enable_mua=True,
                 # Adaptive classification parameters (from previous update)
                 adaptive_classification=False,  # If True, adjust clustering thresholds adaptively
                 dbscan_eps=0.2,                 # Default maximum time (in sec) between events for DBSCAN
                 dbscan_min_samples=1,           # Minimum samples for a cluster in DBSCAN
                 # NEW: HMM edge detection parameters
                 use_hmm_edge_detection=True,    # Enable HMM-based state transition detection
                 hmm_margin=0.1                  # Margin (in seconds) around candidate event for HMM analysis
                 ):
        """
        Parameters for SWR and MUA detection and classification.
        Also includes parameters for adaptive clustering and HMM edge detection.
        """
        # Required parameters
        self.ripple_band = ripple_band
        self.threshold_multiplier = threshold_multiplier
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Optional parameters
        self.notch_freq = notch_freq
        self.sharpwave_band = sharpwave_band
        self.velocity_threshold = velocity_threshold
        self.trace_window = trace_window

        # Event detection parameters
        self.duration_std_threshold = duration_std_threshold
        self.min_event_separation = min_event_separation
        self.merge_threshold = merge_threshold

        # Classification parameters
        self.single_separation = single_separation
        self.burst_min_interval = burst_min_interval
        self.burst_max_interval = burst_max_interval
        self.merge_interval = merge_interval

        # MUA parameters
        self.mua_threshold_multiplier = mua_threshold_multiplier
        self.mua_min_duration = mua_min_duration
        self.enable_mua = enable_mua

        # Adaptive classification parameters
        self.adaptive_classification = adaptive_classification
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # HMM-based edge detection parameters
        self.use_hmm_edge_detection = use_hmm_edge_detection
        self.hmm_margin = hmm_margin

    def update(self, **kwargs):
        """Update parameters with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

# class SWRHMMDetector:
#     def __init__(self, lfp_data, fs, mua_data=None, velocity_data=None, params=None):
#         """
#         Initialize SWR detector.
#         Parameters:
#           lfp_data : np.ndarray
#               LFP data array of shape (n_channels, n_timepoints)
#           fs : float
#               Sampling frequency in Hz
#           mua_data : np.ndarray, optional
#               Multi-unit activity data array of shape (n_timepoints,)
#           velocity_data : np.ndarray, optional
#               Velocity data array of shape (n_timepoints,)
#           params : SWRParams, optional
#               Detection parameters. If None, uses default parameters
#         """
#         # Validate input data format
#         if not isinstance(lfp_data, np.ndarray):
#             raise TypeError("lfp_data must be a numpy array")
#         if len(lfp_data.shape) != 2:
#             raise ValueError("lfp_data must be 2D array (channels x time)")

#         self.n_channels, self.n_timepoints = lfp_data.shape

#         # Validate MUA data if provided
#         if mua_data is not None:
#             if not isinstance(mua_data, np.ndarray):
#                 raise TypeError("mua_data must be a numpy array")
#             if len(mua_data) != self.n_timepoints:
#                 raise ValueError(f"mua_data length ({len(mua_data)}) must match "
#                                  f"lfp_data timepoints ({self.n_timepoints})")

#         # Validate velocity data if provided
#         if velocity_data is not None:
#             if not isinstance(velocity_data, np.ndarray):
#                 raise TypeError("velocity_data must be a numpy array")
#             if len(velocity_data) != self.n_timepoints:
#                 raise ValueError(f"velocity_data length ({len(velocity_data)}) must match "
#                                  f"lfp_data timepoints ({self.n_timepoints})")

#         self.lfp_data = lfp_data
#         self.fs = fs
#         self.mua_data = mua_data
#         self.velocity_data = velocity_data
#         self.params = params if params is not None else SWRHMMParams()
#         self.swr_events = []
#         self.event_counter = 0

#         print(f"Initialized SWR detector with {self.n_channels} channels and "
#               f"{self.n_timepoints / self.fs:.2f} seconds of data")
#         self._print_params()
class SWRHMMDetector:
    def __init__(self, lfp_data, fs, mua_data=None, velocity_data=None, params=None):
        self.multi_region = isinstance(lfp_data, dict)
        self.fs = fs
        self.params = params if params is not None else SWRHMMParams()
        self.event_counter = 0

        if self.multi_region:
            self.region_names = list(lfp_data.keys())
            self.lfp_data = lfp_data
            self.mua_data = mua_data
            self.velocity_data = velocity_data
            self.n_channels = {r: arr.shape[0] for r, arr in lfp_data.items()}
            self.n_timepoints = {r: arr.shape[1] for r, arr in lfp_data.items()}
            self.swr_events = {r: [] for r in self.region_names}
        else:
            self.lfp_data = lfp_data
            self.mua_data = mua_data
            self.velocity_data = velocity_data
            self.n_channels, self.n_timepoints = lfp_data.shape
            self.swr_events = []

        self._print_params()

    def _print_params(self):
        print()
        if self.multi_region:
            for region in self.region_names:
                print(f"Initialized SWR detector for region '{region}' with {self.n_channels[region]} channels and {self.n_timepoints[region] / self.fs:.2f} seconds of data")
        else:
            print(f"Initialized SWR detector with {self.n_channels} channels and {self.n_timepoints / self.fs:.2f} seconds of data")

        print("\nCurrent Detection Parameters:")
        print("=" * 30)
        print("\nCore Parameters:")
        print(f"Ripple band: {self.params.ripple_band} Hz")
        print(f"Threshold multiplier: {self.params.threshold_multiplier} SD")
        print(f"Duration limits: {self.params.min_duration}-{self.params.max_duration} s")

        print("\nSignal Processing:")
        print(f"Notch frequency: {getattr(self.params, 'notch_freq', 'Disabled')} Hz")
        print(f"Sharp wave band: {getattr(self.params, 'sharpwave_band', 'Disabled') or 'Disabled'} Hz")
        print(f"Trace window: {getattr(self.params, 'trace_window', 0.2)} s")

        print("\nEvent Detection:")
        print(f"Duration std threshold: {getattr(self.params, 'duration_std_threshold', 'N/A')} SD")
        print(f"Minimum event separation: {getattr(self.params, 'min_event_separation', 'N/A')} s")
        print(f"Merge threshold: {getattr(self.params, 'merge_threshold', 'N/A')}")

        print("\nClassification Parameters:")
        print(f"Single separation: {getattr(self.params, 'single_separation', 0.2) * 1000:.0f} ms")
        print(f"Burst intervals: {getattr(self.params, 'burst_min_interval', 0.07) * 1000:.0f}-{getattr(self.params, 'burst_max_interval', 0.2) * 1000:.0f} ms")
        print(f"Merge interval: {getattr(self.params, 'merge_interval', 0.07) * 1000:.0f} ms")

        print("\nMUA Parameters:")
        print(f"MUA detection: {'Enabled' if getattr(self.params, 'enable_mua', False) else 'Disabled'}")
        if getattr(self.params, 'enable_mua', False):
            print(f"MUA threshold: {getattr(self.params, 'mua_threshold_multiplier', 'N/A')} SD")
            print(f"MUA min duration: {getattr(self.params, 'mua_min_duration', 0.02) * 1000:.0f} ms")

        print("\nHMM Edge Detection:")
        print(f"Use HMM edge detection: {getattr(self.params, 'use_hmm_edge_detection', False)}")
        if getattr(self.params, 'use_hmm_edge_detection', False):
            print(f"HMM margin: {getattr(self.params, 'hmm_margin', 0.05)} s")

        print("\nMovement Filtering:")
        print(f"Velocity threshold: {getattr(self.params, 'velocity_threshold', 'Disabled')} cm/s")

        print("\nData Information:")
        if self.multi_region:
            for region in self.region_names:
                print(f"Region '{region}':")
                print(f"  Number of channels: {self.n_channels[region]}")
                print(f"  Recording duration: {self.n_timepoints[region] / self.fs:.2f} s")
            print(f"Sampling rate: {self.fs} Hz")
            print(f"MUA data: {'Provided' if self.mua_data is not None else 'Not provided'}")
            print(f"Velocity data: {'Provided' if self.velocity_data is not None else 'Not provided'}")
        else:
            print(f"Number of channels: {self.n_channels}")
            print(f"Recording duration: {self.n_timepoints / self.fs:.2f} s")
            print(f"Sampling rate: {self.fs} Hz")
            print(f"MUA data: {'Provided' if self.mua_data is not None else 'Not provided'}")
            print(f"Velocity data: {'Provided' if self.velocity_data is not None else 'Not provided'}")
        print("=" * 30)

    def _merge_close_events(self):
        """
        Merge events that are closer than merge_interval (e.g., 70ms).
        Ensures that start and end times are properly updated.
        """
        if not self.swr_events:
            return []

        sorted_events = sorted(self.swr_events, key=lambda x: x['start_time'])
        merged = []
        current_merge = None

        for event in sorted_events:
            if current_merge is None:
                current_merge = event.copy()
                continue

            interval = event['start_time'] - current_merge['end_time']

            if interval < self.params.merge_interval:
                # Update the merged event to include the full duration of all combined events
                current_merge['start_time'] = min(current_merge['start_time'], event['start_time'])
                current_merge['end_time'] = max(current_merge['end_time'], event['end_time'])
                current_merge['start_idx'] = min(current_merge['start_idx'], event['start_idx'])
                current_merge['end_idx'] = max(current_merge['end_idx'], event['end_idx'])
                current_merge['duration'] = current_merge['end_time'] - current_merge['start_time']

                # Merge peak times and keep the strongest peak power
                current_merge['peak_times'] += event['peak_times']
                current_merge['peak_power'] = max(current_merge['peak_power'], event['peak_power'])
                continue

            # Append the last merged event before starting a new one
            merged.append(current_merge)
            current_merge = event.copy()

        if current_merge is not None:
            merged.append(current_merge)

        return merged

    def _process_signal(self, signal_data, channel_id):
        """Process a single channel or averaged signal for event detection."""
        # Initial signal processing
        if self.params.notch_freq:
            notched = self._notch_filter(signal_data, self.params.notch_freq)
        else:
            notched = signal_data

        # Ripple processing
        ripple_filtered = self._bandpass_filter(notched, *self.params.ripple_band)
        ripple_power = np.abs(signal.hilbert(ripple_filtered))
        # Smooth the ripple power; here we square the moving average for a robust envelope estimate.
        smooth_ripple_power = np.convolve(ripple_power, np.ones(50) / 50, mode='same') ** 2
        ripple_threshold = (np.mean(smooth_ripple_power) +
                            self.params.threshold_multiplier * np.std(smooth_ripple_power))
        above_ripple = smooth_ripple_power > ripple_threshold

        # MUA processing (if enabled and provided)
        if self.params.enable_mua and self.mua_data is not None:
            """
            lfp_data: np.ndarray (single region) or dict[str, np.ndarray] (multi-region)
            mua_data: np.ndarray (single region) or dict[str, np.ndarray] (multi-region)
            """

            # ...existing code...
            smooth_mua = np.convolve(self.mua_data, np.ones(window_size) / window_size, mode='same')
            mua_threshold = (np.mean(smooth_mua) +
                             self.params.mua_threshold_multiplier * np.std(smooth_mua))
            above_mua = smooth_mua > mua_threshold
        else:
            smooth_mua = None
            above_mua = None

        # Sharp wave processing (if enabled)
        if self.params.sharpwave_band:
            sharpwave_filtered = self._bandpass_filter(notched, *self.params.sharpwave_band)
            sharpwave_power = np.abs(signal.hilbert(sharpwave_filtered))
            smooth_sw_power = np.convolve(sharpwave_power, np.ones(50) / 50, mode='same') ** 2
            sw_threshold = np.mean(smooth_sw_power) - 2 * np.std(smooth_sw_power)
        else:
            sharpwave_filtered = None
            smooth_sw_power = None
            sw_threshold = None

        # Combine thresholds
        if self.params.enable_mua and self.mua_data is not None:
            event_markers = above_ripple | above_mua
        else:
            event_markers = above_ripple

        crossings = np.diff(event_markers.astype(int))
        starts = np.where(crossings == 1)[0]
        ends = np.where(crossings == -1)[0]

        if len(ends) == 0 or len(starts) == 0:
            return
        if ends[0] < starts[0]:
            ends = ends[1:]
        if starts[-1] > ends[-1]:
            starts = starts[:-1]

        # Process each candidate event
        for start, end in zip(starts, ends):
            duration = (end - start) / self.fs
            valid_duration = (duration >= self.params.mua_min_duration and 
                              duration <= self.params.max_duration)
            if not valid_duration:
                continue

            # Determine event type (ripple and/or MUA)
            ripple_event = False
            mua_event = False

            if above_ripple[start:end].any():
                ripple_idx = start + np.argmax(smooth_ripple_power[start:end])
                ripple_peak = smooth_ripple_power[ripple_idx]
                if ripple_peak > ripple_threshold:
                    ripple_event = True

            if self.params.enable_mua and self.mua_data is not None and above_mua[start:end].any():
                mua_idx = start + np.argmax(smooth_mua[start:end])
                mua_peak = smooth_mua[mua_idx]
                if mua_peak > mua_threshold:
                    mua_event = True

            if ripple_event and mua_event:
                event_type = 'ripple_mua'
                peak_idx = ripple_idx
                peak_power = smooth_ripple_power[ripple_idx]
            elif ripple_event:
                event_type = 'ripple_only'
                peak_idx = ripple_idx
                peak_power = smooth_ripple_power[ripple_idx]
            elif mua_event:
                event_type = 'mua_only'
                peak_idx = mua_idx
                peak_power = smooth_mua[mua_idx]
            else:
                continue

            peak_time = peak_idx / self.fs

            # Additional validation checks
            valid_event = True
            if self.params.velocity_threshold is not None and self.velocity_data is not None:
                vel_window = self.velocity_data[start:end]
                if np.mean(vel_window) > self.params.velocity_threshold:
                    valid_event = False
            if valid_event and self.params.sharpwave_band and event_type in ['ripple_only', 'ripple_mua']:
                sw_window = smooth_sw_power[start:end]
                if not np.any(sw_window < sw_threshold):
                    valid_event = False
            if not valid_event:
                continue

            # === NEW: HMM-based Edge Refinement ===
            if self.params.use_hmm_edge_detection:
                margin_samples = int(self.params.hmm_margin * self.fs)
                win_start = max(0, start - margin_samples)
                win_end = min(len(smooth_ripple_power), end + margin_samples)
                refined_start, refined_end = self._refine_event_edges_hmm(smooth_ripple_power, win_start, win_end)
                # Update boundaries if valid
                if refined_end > refined_start:
                    start = refined_start
                    end = refined_end
                    duration = (end - start) / self.fs
            # =======================================
            
            # Extract traces based on a trace window
            win_samples = int(self.params.trace_window * self.fs)
            half_win = win_samples // 2
            trace_start = max(0, peak_idx - half_win)
            trace_end = min(len(notched), peak_idx + half_win)

            # Find peaks for additional classification
            if event_type in ['ripple_only', 'ripple_mua']:
                peak_times = signal.find_peaks(
                    smooth_ripple_power[start:end],
                    height=ripple_threshold,
                    distance=int(0.02 * self.fs)
                )[0]
            else:
                peak_times = signal.find_peaks(
                    smooth_mua[start:end],
                    height=mua_threshold,
                    distance=int(0.02 * self.fs)
                )[0]
            peak_times = [start + pk_idx for pk_idx in peak_times]

            self.event_counter += 1
            new_event = {
                'event_id': self.event_counter,
                'channel': channel_id,
                'start_idx': start,
                'end_idx': end,
                'start_time': start / self.fs,
                'end_time': end / self.fs,
                'duration': duration,
                'peak_time': peak_time,
                'peak_power': peak_power,
                'peak_times': peak_times,
                'event_type': event_type,
                'raw_trace': notched[trace_start:trace_end],
                'ripple_trace': ripple_filtered[trace_start:trace_end],
                'mua_trace': smooth_mua[trace_start:trace_end] if smooth_mua is not None else None,
                'ripple_power': ripple_power[trace_start:trace_end],
                'sharpwave_trace': sharpwave_filtered[trace_start:trace_end] if sharpwave_filtered is not None else None,
                'trace_timestamps': np.linspace(
                    peak_time - (peak_idx - trace_start) / self.fs,
                    peak_time + (trace_end - peak_idx) / self.fs,
                    trace_end - trace_start
                )
            }
            self.swr_events.append(new_event)

    def get_classification_summary(self):
        """
        Return a summary of event classifications including both temporal patterns
        and event types (ripple/MUA).
        """
        if not self.swr_events:
            return None

        # Initialize counters for each event type
        classification_stats = {
            event_type: {
                'singles': 0,
                'doubles': 0,
                'triples': 0,
                'more': 0,
                'unclassified': 0,
                'intervals': {
                    'doubles': [],
                    'triples': [],
                    'more': []
                }
            }
            for event_type in ['ripple_mua', 'ripple_only', 'mua_only']
        }

        # Add parameters section
        classification_stats['parameters'] = {
            'single_separation': self.params.single_separation,
            'burst_min_interval': self.params.burst_min_interval,
            'burst_max_interval': self.params.burst_max_interval,
            'merge_interval': self.params.merge_interval
        }

        # Count events by classification and type
        processed_groups = set()
        for event in self.swr_events:
            if 'classification' not in event:
                continue

            group_id = event['classification']['group_id']
            if group_id is None or group_id in processed_groups:
                continue

            processed_groups.add(group_id)
            event_type = event['event_type']
            group_type = event['classification']['group_type']
            intervals = event['classification']['inter_event_intervals']

            if group_type == 'single':
                classification_stats[event_type]['singles'] += 1
            elif group_type == 'double':
                classification_stats[event_type]['doubles'] += 1
                classification_stats[event_type]['intervals']['doubles'].extend(intervals)
            elif group_type == 'triple':
                classification_stats[event_type]['triples'] += 1
                classification_stats[event_type]['intervals']['triples'].extend(intervals)
            elif group_type == 'multiple':
                classification_stats[event_type]['more'] += 1
                classification_stats[event_type]['intervals']['more'].extend(intervals)
            else:
                classification_stats[event_type]['unclassified'] += 1

        return classification_stats

    def _assign_group_classification(self, group, group_id):
        """Assign classification details to a group of events."""
        group_size = len(group)
    
        # Determine group type
        if group_size == 1:
            # Check if it's truly isolated
            is_single = True
            if len(self.swr_events) > 1:
                event = group[0]
                event_idx = self.swr_events.index(event)
            
                # Check previous event if exists
                if event_idx > 0:
                    prev_event = self.swr_events[event_idx - 1]
                    if (prev_event['channel'] == event['channel'] and
                        event['start_time'] - prev_event['end_time'] < self.params.single_separation):
                        is_single = False
            
                # Check next event if exists
                if event_idx < len(self.swr_events) - 1:
                    next_event = self.swr_events[event_idx + 1]
                    if (next_event['channel'] == event['channel'] and
                        next_event['start_time'] - event['end_time'] < self.params.single_separation):
                        is_single = False
        
            group_type = 'single' if is_single else 'unclassified'
        elif group_size == 2:
            group_type = 'double'
        elif group_size == 3:
            group_type = 'triple'
        else:
            group_type = 'multiple'
    
        # Calculate inter-event intervals
        intervals = []
        for i in range(1, len(group)):
            interval = group[i]['start_time'] - group[i-1]['end_time']
            intervals.append(interval)
    
        # Update classification for each event in group
        for i, event in enumerate(group):
            event['classification'] = {
                'group_size': group_size,
                'group_id': group_id,
                'position_in_group': i + 1,
                'group_type': group_type,
                'inter_event_intervals': intervals,
                'event_type': event['event_type'],
                'peak_count': len(event.get('peak_times', [1]))
            }

    def _classify_channel_events(self, events):
        """Helper method to classify events within a single channel."""
        if not events:
            return []
        
        groups = []
        current_group = [events[0]]
    
        for i in range(1, len(events)):
            current_event = events[i]
            prev_event = current_group[-1]
        
            # Calculate interval to previous event
            interval = current_event['start_time'] - prev_event['end_time']
        
            # Check if event belongs to current group
            if (interval >= self.params.burst_min_interval and 
                interval <= self.params.burst_max_interval and
                current_event['event_type'] == prev_event['event_type']):
                current_group.append(current_event)
            else:
                # Classify previous group
                self._assign_group_classification(current_group, len(groups))
                groups.append(current_group)
                current_group = [current_event]
    
        # Classify last group
        if current_group:
            self._assign_group_classification(current_group, len(groups))
            groups.append(current_group)
    
        return groups

    def inspect_all_group_types(self, n_examples=2):
        """
        Show examples of all group types (singles, doubles, triples, multiples).
    
        Parameters:
        -----------
        n_examples : int
            Number of examples to show for each type
        """
        for group_type in ['single', 'double', 'triple', 'multiple']:
            print(f"\nInspecting {group_type} events:")
            plt.figure(figsize=(15, 1))
            plt.text(0.5, 0.5, f"{group_type.upper()} EVENTS", 
                    horizontalalignment='center', fontsize=14)
            plt.axis('off')
            plt.show()
            self.inspect_event_groups(group_type, n_examples=n_examples)

    def inspect_event_groups(self, group_type='single', n_examples=3, window_size=1.0):
        """
        Visualize examples of events from a specific group type to verify classification.
    
        Parameters:
        -----------
        group_type : str
            Type of group to inspect ('single', 'double', 'triple', 'multiple')
        n_examples : int
            Number of examples to show
        window_size : float
            Size of the window around events in seconds
        """
        # Find events of the requested type
        examples = []
        group_ids_seen = set()
    
        for event in self.swr_events:
            if 'classification' not in event:
                continue
            
            if (event['classification']['group_type'] == group_type and 
                event['classification']['group_id'] not in group_ids_seen):
            
                # Get all events in this group
                group_id = event['classification']['group_id']
                group_events = [e for e in self.swr_events 
                            if 'classification' in e and 
                            e['classification']['group_id'] == group_id]
            
                examples.append(group_events)
                group_ids_seen.add(group_id)
            
                if len(examples) >= n_examples:
                    break
    
        if not examples:
            print(f"No {group_type} events found")
            return
    
        # Create figure
        n_rows = len(examples)
        fig = plt.figure(figsize=(15, 4*n_rows))
    
        for i, group in enumerate(examples):
            # Sort events by time
            group = sorted(group, key=lambda x: x['start_time'])
        
            # Find time window
            start_time = group[0]['start_time'] - window_size/4
            end_time = group[-1]['end_time'] + window_size/4
        
            # Get data indices
            start_idx = max(0, int(start_time * self.fs))
            end_idx = min(self.n_timepoints, int(end_time * self.fs))
            t = np.arange(start_idx, end_idx) / self.fs
        
            # Plot raw signal
            ax1 = plt.subplot(n_rows, 3, i*3 + 1)
            if isinstance(group[0]['channel'], int):
                signal = self.lfp_data[group[0]['channel'], start_idx:end_idx]
            else:
                signal = np.mean(self.lfp_data[:, start_idx:end_idx], axis=0)
            
            ax1.plot(t, signal, 'k')
        
            # Highlight events
            for event in group:
                ax1.axvspan(event['start_time'], event['end_time'], 
                        color='yellow', alpha=0.3)
                ax1.axvline(x=event['peak_time'], color='r', linestyle='--')
        
            ax1.set_title(f"Example {i+1}: Raw Signal (Ch {group[0]['channel']})")
            ax1.set_ylabel('Amplitude')
        
            # Plot ripple power
            ax2 = plt.subplot(n_rows, 3, i*3 + 2)
            for event in group:
                # Get ripple power for full window
                if event['ripple_power'] is not None:
                    # Plot event ripple power
                    ax2.plot(event['trace_timestamps'], event['ripple_power'], 'r')
                ax2.axvspan(event['start_time'], event['end_time'],
                        color='yellow', alpha=0.3)
                ax2.axvline(x=event['peak_time'], color='r', linestyle='--')
        
            ax2.set_title("Ripple Power")
            ax2.set_ylabel('Power')
        
            # Plot MUA if available
            ax3 = plt.subplot(n_rows, 3, i*3 + 3)
            for event in group:
                if event['mua_trace'] is not None:
                    ax3.plot(event['trace_timestamps'], event['mua_trace'], 'g')
                ax3.axvspan(event['start_time'], event['end_time'],
                        color='yellow', alpha=0.3)
                ax3.axvline(x=event['peak_time'], color='r', linestyle='--')
        
            ax3.set_title("MUA")
            ax3.set_ylabel('Rate')
        
            # Add event information
            durations = [f"{event['duration']*1000:.1f}ms" for event in group]
            info_text = [
                f"Events: {len(group)}",
                f"Type: {group[0]['event_type']}",
                f"Duration: {', '.join(durations)}"
            ]
        
            if len(group) > 1:
                intervals = group[0]['classification']['inter_event_intervals']
                interval_text = [f"{interval*1000:.1f}ms" for interval in intervals]
                info_text.append(f"Intervals: {', '.join(interval_text)}")
        
            ax1.text(0.02, 0.98, '\n'.join(info_text),
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
        plt.tight_layout()
        plt.show()

    def plot_basic_stats(self, stats):
        """
        Create visualizations for the basic statistical analysis.
    
        Parameters:
        stats : dict
            Output from analyze_basic_stats method
        """
        if not stats:
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
    
        # 1. Event type distribution pie chart
        ax1 = plt.subplot(231)
        labels = []
        sizes = []
        for event_type, count in stats['event_counts']['by_type'].items():
            if count > 0:
                labels.append(event_type)
                sizes.append(count)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax1.set_title('Event Type Distribution')
    
        # 2. Detection type distribution pie chart
        ax2 = plt.subplot(232)
        det_labels = []
        det_sizes = []
        for det_type, count in stats['event_counts']['by_detection'].items():
            if count > 0:
                det_labels.append(det_type)
                det_sizes.append(count)
        ax2.pie(det_sizes, labels=det_labels, autopct='%1.1f%%')
        ax2.set_title('Detection Type Distribution')
    
        # 3. Duration boxplot
        ax3 = plt.subplot(233)
        duration_data = []
        duration_labels = []
        for event_type in ['single', 'double', 'triple', 'multiple']:
            if stats['durations'][event_type]:
                duration_data.append(stats['durations'][event_type])
                duration_labels.append(event_type)
        if duration_data:
            ax3.boxplot(duration_data, labels=duration_labels)
            ax3.set_ylabel('Duration (s)')
            ax3.set_title('Event Durations')
    
        # 4. MUA amplitude boxplot
        ax4 = plt.subplot(234)
        mua_data = []
        mua_labels = []
        for event_type in ['single', 'double', 'triple', 'multiple']:
            if stats['mua_amplitudes'][event_type]:
                mua_data.append(stats['mua_amplitudes'][event_type])
                mua_labels.append(event_type)
        if mua_data:
            ax4.boxplot(mua_data, labels=mua_labels)
            ax4.set_ylabel('MUA Amplitude')
            ax4.set_title('MUA Amplitudes')
    
        # 5. Inter-event interval boxplot
        ax5 = plt.subplot(235)
        iei_data = []
        iei_labels = []
        for event_type in ['double', 'triple', 'multiple']:
            if stats['inter_event_intervals'][event_type]:
                iei_data.append(stats['inter_event_intervals'][event_type])
                iei_labels.append(event_type)
        if iei_data:
            ax5.boxplot(iei_data, labels=iei_labels)
            ax5.set_ylabel('Interval (s)')
            ax5.set_title('Inter-Event Intervals')
    
        # 6. Peak power boxplot
        ax6 = plt.subplot(236)
        power_data = []
        power_labels = []
        for event_type in ['single', 'double', 'triple', 'multiple']:
            if stats['peak_powers'][event_type]:
                power_data.append(stats['peak_powers'][event_type])
                power_labels.append(event_type)
        if power_data:
            ax6.boxplot(power_data, labels=power_labels)
            ax6.set_ylabel('Peak Power')
            ax6.set_title('Peak Powers')
    
        plt.tight_layout()
        plt.show()

    def analyze_basic_stats(self):
        """
        Perform basic statistical analysis on detected events, focusing on:
        - Event type distributions (singles, doubles, triples)
        - Duration and amplitude statistics for each type
        - MUA patterns for different event types
    
        Returns:
        dict: Dictionary containing analysis results
        """
        if not self.swr_events:
            print("No events detected")
            return None
        
        # Initialize result dictionary
        stats = {
            'event_counts': {
                'total': len(self.swr_events),
                'by_type': {'single': 0, 'double': 0, 'triple': 0, 'multiple': 0},
                'by_detection': {'ripple_only': 0, 'mua_only': 0, 'ripple_mua': 0}
            },
            'durations': {
                'single': [], 'double': [], 'triple': [], 'multiple': []
            },
            'peak_powers': {
                'single': [], 'double': [], 'triple': [], 'multiple': []
            },
            'mua_amplitudes': {
                'single': [], 'double': [], 'triple': [], 'multiple': []
            },
            'inter_event_intervals': {
                'double': [], 'triple': [], 'multiple': []
            }
        }
    
        # Collect data for each event
        processed_groups = set()
        for event in self.swr_events:
            if 'classification' not in event:
                continue
            
            group_id = event['classification'].get('group_id')
            if group_id is None:
                continue
            
            group_type = event['classification']['group_type']
            if group_type not in ['single', 'double', 'triple', 'multiple']:
                continue
            
            # Count event types
            if group_id not in processed_groups:
                stats['event_counts']['by_type'][group_type] += 1
                processed_groups.add(group_id)
        
            # Count detection types
            stats['event_counts']['by_detection'][event['event_type']] += 1
        
            # Collect duration and power data
            stats['durations'][group_type].append(event['duration'])
            stats['peak_powers'][group_type].append(event['peak_power'])
        
            # Collect MUA data if available
            if event['mua_trace'] is not None:
                mua_amplitude = np.max(event['mua_trace'])
                stats['mua_amplitudes'][group_type].append(mua_amplitude)
            
            # Collect inter-event intervals
            if event['classification']['inter_event_intervals']:
                stats['inter_event_intervals'][group_type].extend(
                    event['classification']['inter_event_intervals']
                )
    
        # Calculate summary statistics
        stats['summary'] = {
            'durations': {},
            'peak_powers': {},
            'mua_amplitudes': {},
            'inter_event_intervals': {}
        }
    
        for event_type in ['single', 'double', 'triple', 'multiple']:
            for measure in ['durations', 'peak_powers', 'mua_amplitudes']:
                data = stats[measure][event_type]
                if data:
                    stats['summary'][measure][event_type] = {
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'median': np.median(data),
                        'q25': np.percentile(data, 25),
                        'q75': np.percentile(data, 75),
                        'n': len(data)
                    }
                
            # Calculate IEI statistics for non-singles
            if event_type != 'single' and stats['inter_event_intervals'][event_type]:
                iei_data = stats['inter_event_intervals'][event_type]
                stats['summary']['inter_event_intervals'][event_type] = {
                    'mean': np.mean(iei_data),
                    'std': np.std(iei_data),
                    'median': np.median(iei_data),
                    'q25': np.percentile(iei_data, 25),
                    'q75': np.percentile(iei_data, 75),
                    'n': len(iei_data)
                }
    
        # Print summary report
        print("\nEvent Type Distribution:")
        print("-" * 50)
        for event_type, count in stats['event_counts']['by_type'].items():
            print(f"{event_type.capitalize()}: {count} ({count/stats['event_counts']['total']*100:.1f}%)")
        
        print("\nDetection Type Distribution:")
        print("-" * 50)
        for det_type, count in stats['event_counts']['by_detection'].items():
            print(f"{det_type}: {count} ({count/stats['event_counts']['total']*100:.1f}%)")
        
        print("\nDuration Statistics (seconds):")
        print("-" * 50)
        for event_type in ['single', 'double', 'triple', 'multiple']:
            if event_type in stats['summary']['durations']:
                d = stats['summary']['durations'][event_type]
                print(f"{event_type.capitalize()}: {d['mean']:.3f} ± {d['std']:.3f} (n={d['n']})")
            
        if stats['summary']['mua_amplitudes']:
            print("\nMUA Amplitude Statistics:")
            print("-" * 50)
            for event_type in ['single', 'double', 'triple', 'multiple']:
                if event_type in stats['summary']['mua_amplitudes']:
                    m = stats['summary']['mua_amplitudes'][event_type]
                    print(f"{event_type.capitalize()}: {m['mean']:.2f} ± {m['std']:.2f} (n={m['n']})")
    
        print("\nInter-Event Interval Statistics (seconds):")
        print("-" * 50)
        for event_type in ['double', 'triple', 'multiple']:
            if event_type in stats['summary']['inter_event_intervals']:
                i = stats['summary']['inter_event_intervals'][event_type]
                print(f"{event_type.capitalize()}: {i['mean']:.3f} ± {i['std']:.3f} (n={i['n']})")
    
        return stats

    def visualize_events(self):
        """Create interactive visualization of detected events including MUA data in a separate subplot."""
        if not self.swr_events:
            print("No events detected")
            return

        # Create widgets
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
            max=len(self.swr_events) - 1,
            description='Event #:',
            style={'description_width': 'initial'}
        )

        # Create output widget for plots and text
        out = widgets.Output()

        def plot_event(event_idx):
            event = self.swr_events[event_idx]
            with out:
                # Clear previous output
                plt.close('all')
                out.clear_output(wait=True)

                # Determine number of subplots based on available data
                n_plots = 4  # Raw, Ripple, MUA, Context
                if self.velocity_data is not None:
                    n_plots += 1

                # Create figure
                fig = plt.figure(figsize=(12, 3 * n_plots))

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
                peak_idx = int(event['peak_time'] * self.fs)
                context_start = max(0, int(peak_idx - self.fs * context_window / 2))
                context_end = min(self.n_timepoints, int(peak_idx + self.fs * context_window / 2))
                time_context = np.arange(context_start, context_end) / self.fs

                if isinstance(event['channel'], int):
                    signal_context = self.lfp_data[event['channel'], context_start:context_end]
                else:
                    signal_context = np.mean(self.lfp_data[:, context_start:context_end], axis=0)

                ax4.plot(time_context, signal_context, 'k', label='Signal')
                ax4.axvspan(event['start_time'], event['end_time'],
                            color='yellow', alpha=0.3, label='Event')
                ax4.axvline(x=event['peak_time'], color='r',
                            linestyle='--', label='Peak')
                ax4.set_title('Event Context (±250ms)')
                ax4.set_ylabel('Amplitude')
                ax4.legend()

                # Plot 5: Velocity data if available
                if self.velocity_data is not None:
                    ax5 = plt.subplot(n_plots, 1, 5)
                    velocity_trace = self.velocity_data[context_start:context_end]
                    ax5.plot(time_context, velocity_trace, 'b', label='Velocity')

                    if self.params.velocity_threshold is not None:
                        ax5.axhline(y=self.params.velocity_threshold, color='r',
                                    linestyle='--',
                                    label=f'Threshold ({self.params.velocity_threshold} cm/s)')

                    ax5.axvspan(event['start_time'], event['end_time'],
                                color='yellow', alpha=0.3)
                    ax5.axvline(x=event['peak_time'], color='r',
                                linestyle='--', label='Peak')
                    ax5.set_title('Velocity')
                    ax5.set_xlabel('Time (s)')
                    ax5.set_ylabel('Velocity (cm/s)')
                    ax5.legend()

                    # Calculate mean velocity during event
                    event_start_idx = int(event['start_time'] * self.fs)
                    event_end_idx = int(event['end_time'] * self.fs)
                    mean_velocity = np.mean(self.velocity_data[event_start_idx:event_end_idx])

                plt.tight_layout()

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

                if self.velocity_data is not None:
                    print(f"Mean velocity during event: {mean_velocity:.2f} cm/s")

                if self.params.velocity_threshold is not None:
                    print(f"Velocity threshold: {self.params.velocity_threshold} cm/s")

                # Add classification information if available
                if 'classification' in event:
                    print("\nClassification Details:")
                    print(f"Group type: {event['classification']['group_type']}")
                    print(f"Group size: {event['classification']['group_size']}")
                    print(f"Position in group: {event['classification']['position_in_group']}")
                    if event['classification']['inter_event_intervals']:
                        print("Inter-event intervals: " +
                              ", ".join([f"{x * 1000:.1f}ms" for x in event['classification']['inter_event_intervals']]))

                plt.show()

        # Button click handlers
        def on_prev_clicked(b):
            event_input.value = max(0, event_input.value - 1)

        def on_next_clicked(b):
            event_input.value = min(len(self.swr_events) - 1, event_input.value + 1)

        def on_value_change(change):
            plot_event(change['new'])

        # Connect handlers
        prev_button.on_click(on_prev_clicked)
        next_button.on_click(on_next_clicked)
        event_input.observe(on_value_change, names='value')

        # Create layout
        buttons = widgets.HBox([prev_button, event_input, next_button])

        # Display everything
        display(widgets.VBox([buttons, out]))

        # Show initial plot
        plot_event(0)

        # Print instructions
        print("\nNavigation Controls:")
        print("- Click 'Previous' and 'Next' buttons")
        print("- Type event number directly in the input box")

    def visualize_duration_estimation(self, event_id=None, group_type=None, group_event_index=0, show_classification_info=True):
        """
        Visualize how duration is estimated for a specific event.
    
        Parameters:
        event_id : int, optional
            ID of the event to visualize (if group_type is not provided).
        group_type : str, optional
          If provided, only events with classification['group_type'] equal to this value
          are considered (e.g., 'single', 'double', 'triple', 'multiple').
        group_event_index : int, optional
          The index of the event within the filtered list (default is 0).
        show_classification_info : bool, default True
          If True, include classification details in the visualization and printed output.
        """
        # Select event based on group_type if provided
        if group_type is not None:
            filtered_events = [e for e in self.swr_events if 'classification' in e and e['classification'].get('group_type') == group_type]
            if not filtered_events:
                print(f"No events found for group type '{group_type}'.")
                return
            if group_event_index < 0 or group_event_index >= len(filtered_events):
                print(f"Invalid group_event_index: {group_event_index}. Must be between 0 and {len(filtered_events)-1}.")
                return
            event = filtered_events[group_event_index]
        elif event_id is not None:
            event = self.get_event(event_id)
            if event is None:
                return
        else:
            print("Please provide either an event_id or a group_type with group_event_index.")
            return

        # Get the channel data
        if isinstance(event['channel'], int):
            signal_data = self.lfp_data[event['channel']]
        else:  # averaged signal
            signal_data = np.mean(self.lfp_data, axis=0)

        # Process signal
        if self.params.notch_freq:
            notched = self._notch_filter(signal_data, self.params.notch_freq)
        else:
            notched = signal_data

        # Get ripple power and its smoothed envelope
        ripple_filtered = self._bandpass_filter(notched, *self.params.ripple_band)
        ripple_power = np.abs(signal.hilbert(ripple_filtered))
        smooth_ripple_power = np.convolve(ripple_power, np.ones(50) / 50, mode='same') ** 2

        # Get MUA if available and applicable
        if self.mua_data is not None and event['event_type'] in ['mua_only', 'ripple_mua']:
            window_size = int(0.02 * self.fs)  # 20ms window
            smooth_mua = np.convolve(self.mua_data, np.ones(window_size) / window_size, mode='same')
        else:
            smooth_mua = None

        # Define a window around the event for plotting
        window_samples = int(0.5 * self.fs)  # 500ms window
        start_idx = max(0, event['start_idx'] - window_samples)
        end_idx = min(len(smooth_ripple_power), event['end_idx'] + window_samples)

        # Calculate thresholds
        ripple_threshold = (np.mean(smooth_ripple_power) +
                            self.params.threshold_multiplier * np.std(smooth_ripple_power))
        if smooth_mua is not None:
            mua_threshold = (np.mean(smooth_mua) +
                            self.params.mua_threshold_multiplier * np.std(smooth_mua))

        # Create time axis for plotting
        time = np.arange(start_idx, end_idx) / self.fs

        # Plotting the results
        fig = plt.figure(figsize=(12, 8))

        # Plot 1: Ripple power and thresholds
        ax1 = plt.subplot(311)
        ax1.plot(time, smooth_ripple_power[start_idx:end_idx], 'b', label='Ripple Power')
        ax1.axhline(y=ripple_threshold, color='b', linestyle='--', label='Ripple Threshold')
        if smooth_mua is not None:
            ax1.plot(time, smooth_mua[start_idx:end_idx], 'r', label='MUA')
            ax1.axhline(y=mua_threshold, color='r', linestyle='--', label='MUA Threshold')
        ax1.axvspan(event['start_time'], event['end_time'], color='yellow', alpha=0.3, label='Event')
        ax1.axvline(x=event['peak_time'], color='k', linestyle='--', label='Peak')

        title_str = f'Event {event["event_id"]} Duration Estimation ({event["event_type"]})'
        if show_classification_info and 'classification' in event:
            title_str += f'\nClassification: {event["classification"]["group_type"]} ' \
                        f'({event["classification"]["position_in_group"]} of {event["classification"]["group_size"]})'
        ax1.set_title(title_str)
        ax1.legend()

        # Plot 2: Filtered ripple trace and (if available) sharp wave
        ax2 = plt.subplot(312)
        ax2.plot(time, ripple_filtered[start_idx:end_idx], 'b', label='Ripple')
        if event.get('sharpwave_trace') is not None:
            ax2.plot(time, event['sharpwave_trace'], 'g', label='Sharp Wave')
        ax2.axvspan(event['start_time'], event['end_time'], color='yellow', alpha=0.3)
        ax2.axvline(x=event['peak_time'], color='k', linestyle='--')
        ax2.set_ylabel('Amplitude')
        ax2.legend()

     # Plot 3: Raw signal
        ax3 = plt.subplot(313)
        ax3.plot(time, notched[start_idx:end_idx], 'k', label='Raw Signal')
        ax3.axvspan(event['start_time'], event['end_time'], color='yellow', alpha=0.3)
        ax3.axvline(x=event['peak_time'], color='k', linestyle='--', label='Peak')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.legend()

        plt.tight_layout()
        plt.show()

        # Print event details
        print(f"\nEvent {event['event_id']} Details:")
        print(f"Type: {event['event_type']}")
        print(f"Duration: {event['duration'] * 1000:.1f} ms")
        print(f"Start time: {event['start_time']:.3f} s")
        print(f"Peak time: {event['peak_time']:.3f} s")
        print(f"End time: {event['end_time']:.3f} s")
        print(f"Number of peaks: {len(event['peak_times'])}")

        if show_classification_info and 'classification' in event:
            print(f"\nClassification:")
            print(f"Group type: {event['classification']['group_type']}")
            print(f"Position in group: {event['classification']['position_in_group']} of {event['classification']['group_size']}")
            if event['classification']['inter_event_intervals']:
                intervals_str = ", ".join([f"{x * 1000:.1f}ms" for x in event['classification']['inter_event_intervals']])
                print(f"Inter-event intervals: {intervals_str}")
    
    def get_channel_events(self, channel):
        """Get all events from a specific channel."""
        channel_events = [event for event in self.swr_events if event['channel'] == channel]
        if not channel_events:
            print(f"No events found for channel {channel}")
        return channel_events

    def get_event(self, event_id):
        """Get a specific event by its ID."""
        for event in self.swr_events:
            if event['event_id'] == event_id:
                return event
        print(f"Event ID {event_id} not found")
        return None

    def save_events(self, filename):
        """Save detected events to a file."""
        events_df = self.get_events_summary()
        if events_df is not None:
            events_df.to_csv(filename, index=False)
            print(f"Saved {len(events_df)} events to {filename}")

    def get_events_summary(self):
        """Return a pandas DataFrame with key event parameters."""
        if not self.swr_events:
            print("No events detected")
            return None

        summary = []
        for event in self.swr_events:
            event_summary = {
                'event_id': event['event_id'],
                'channel': event['channel'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'duration': event['duration'],
                'peak_time': event['peak_time'],
                'peak_power': event['peak_power'],
                'event_type': event['event_type']
            }

            # Add classification information if available
            if 'classification' in event:
                event_summary.update({
                    'group_type': event['classification']['group_type'],
                    'group_size': event['classification']['group_size'],
                    'position_in_group': event['classification']['position_in_group'],
                    'peak_count': event['classification']['peak_count']
                })

            summary.append(event_summary)

        return pd.DataFrame(summary)

    def classify_events_improved(self):
        """
        Improved classification method that first merges very close events
        and then uses DBSCAN clustering (optionally with adaptive thresholds)
        to group events by their occurrence time within each channel.
        """
        # Merge events that occur within the merge_interval
        self.swr_events = self._merge_close_events()
        if not self.swr_events:
            print("No events detected for classification.")
            return

        # Group events by channel
        channels = np.unique([event['channel'] for event in self.swr_events])
        all_groups = []
    
        for ch in channels:
            channel_events = [e for e in self.swr_events if e['channel'] == ch]
            channel_events.sort(key=lambda x: x['start_time'])
            # Extract event start times as the feature for clustering
            times = np.array([e['start_time'] for e in channel_events]).reshape(-1, 1)
    
            # Determine eps: if adaptive, adjust eps based on median inter-event interval
            eps = self.params.dbscan_eps
            if self.params.adaptive_classification and len(times) > 1:
                intervals = np.diff(times.flatten())
                adaptive_eps = np.median(intervals)
                eps = (eps + adaptive_eps) / 2.0
                print(f"Channel {ch}: adaptive eps set to {eps:.3f} s based on median IEI {adaptive_eps:.3f} s")
    
            # Perform DBSCAN clustering on event start times
            db = DBSCAN(eps=eps, min_samples=self.params.dbscan_min_samples)
            labels = db.fit_predict(times)
            unique_labels = set(labels)
    
            for label in unique_labels:
                if label == -1:
                    # Noise events (not assigned to a cluster); treat each as isolated singles.
                    noise_events = [channel_events[i] for i, lab in enumerate(labels) if lab == -1]
                    for event in noise_events:
                        event['classification'] = {
                            'group_size': 1,
                            'group_id': f"{ch}_noise",
                            'position_in_group': 1,
                            'group_type': 'single',
                            'inter_event_intervals': [],
                            'event_type': event['event_type'],
                            'peak_count': len(event.get('peak_times', [1]))
                        }
                        all_groups.append([event])
                else:
                    # Group events assigned to the same cluster
                    cluster_indices = [i for i, lab in enumerate(labels) if lab == label]
                    group_events = [channel_events[i] for i in cluster_indices]
                    group_events.sort(key=lambda x: x['start_time'])
                    group_size = len(group_events)
    
                    if group_size == 1:
                        group_type = 'single'
                    elif group_size == 2:
                        group_type = 'double'
                    elif group_size == 3:
                        group_type = 'triple'
                    else:
                        group_type = 'multiple'
    
                    # Compute inter-event intervals within the group
                    intervals = []
                    for i in range(1, len(group_events)):
                        intervals.append(group_events[i]['start_time'] - group_events[i-1]['end_time'])
    
                    # Optionally, one might check for similarity in amplitudes here
    
                    # Assign classification to each event in the group
                    for i, event in enumerate(group_events):
                        event['classification'] = {
                            'group_size': group_size,
                            'group_id': f"{ch}_{label}",
                            'position_in_group': i + 1,
                            'group_type': group_type,
                            'inter_event_intervals': intervals,
                            'event_type': event['event_type'],
                            'peak_count': len(event.get('peak_times', [1]))
                        }
                    all_groups.append(group_events)
    
        # Print a summary of the classification
        singles = sum(1 for g in all_groups if len(g) == 1)
        doubles = sum(1 for g in all_groups if len(g) == 2)
        triples = sum(1 for g in all_groups if len(g) == 3)
        multiples = sum(1 for g in all_groups if len(g) > 3)
    
        print(f"\nImproved Classification Summary:")
        print(f"Singles: {singles}")
        print(f"Doubles: {doubles}")
        print(f"Triples: {triples}")
        print(f"Multiples: {multiples}")

    def detect_events(self, channels='all', average_mode=False):
        """
        Detect SWR events either on individual channels or averaged signal.
        """
        if channels == 'all':
            channels = np.arange(self.n_channels)
        else:
            channels = np.array(channels)
            if np.any(channels >= self.n_channels) or np.any(channels < 0):
                raise ValueError(f"Channel indices must be between 0 and {self.n_channels - 1}")

        if average_mode:
            print(f"Detecting events on averaged signal from {len(channels)} channels")
            avg_lfp = np.mean(self.lfp_data[channels], axis=0)
            self._process_signal(avg_lfp, 'avg')
        else:
            print(f"Detecting events on {len(channels)} individual channels")
            for ch in channels:
                self._process_signal(self.lfp_data[ch], ch)

        print(f"Detected {len(self.swr_events)} events")
    
    