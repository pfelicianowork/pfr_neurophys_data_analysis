"""
detector.py - Core SWR detection functionality

This module contains the main SWRDetector class that implements sophisticated
sharp wave ripple detection with HMM-based edge refinement and advanced
classification algorithms.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import DBSCAN
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns
import os

from .params import SWRParams


class SWRDetector:
    """
    Advanced Sharp Wave Ripple (SWR) detection and analysis system.

    This class provides comprehensive tools for detecting SWRs in neural data,
    including HMM-based edge detection, multi-channel processing, MUA integration,
    and sophisticated event classification.

    Key Features:
    - Multi-channel SWR detection
    - HMM-based edge refinement for precise boundaries
    - MUA (multi-unit activity) integration
    - Advanced event classification (singles, doubles, triples, bursts)
    - Interactive visualization tools
    - Statistical analysis capabilities
    """

    def __init__(self, lfp_data, fs, mua_data=None, velocity_data=None, params=None):
        """
        Initialize SWR detector.

        Parameters
        ----------
        lfp_data : np.ndarray
            LFP data array of shape (n_channels, n_timepoints)
        fs : float
            Sampling frequency in Hz
        mua_data : np.ndarray, optional
            Multi-unit activity data array of shape (n_timepoints,)
        velocity_data : np.ndarray, optional
            Velocity data array of shape (n_timepoints,)
        params : SWRParams, optional
            Detection parameters. If None, uses default parameters

        Raises
        ------
        TypeError
            If input data types are incorrect
        ValueError
            If input data dimensions are incorrect
        """
        # Validate input data format
        if not isinstance(lfp_data, np.ndarray):
            raise TypeError("lfp_data must be a numpy array")
        if len(lfp_data.shape) != 2:
            raise ValueError("lfp_data must be 2D array (channels x time)")

        self.n_channels, self.n_timepoints = lfp_data.shape

        # Validate MUA data if provided
        if mua_data is not None:
            if not isinstance(mua_data, np.ndarray):
                raise TypeError("mua_data must be a numpy array")
            if len(mua_data) != self.n_timepoints:
                raise ValueError(f"mua_data length ({len(mua_data)}) must match "
                                 f"lfp_data timepoints ({self.n_timepoints})")

        # Validate velocity data if provided
        if velocity_data is not None:
            if not isinstance(velocity_data, np.ndarray):
                raise TypeError("velocity_data must be a numpy array")
            if len(velocity_data) != self.n_timepoints:
                raise ValueError(f"velocity_data length ({len(velocity_data)}) must match "
                                 f"lfp_data timepoints ({self.n_timepoints})")

        self.lfp_data = lfp_data
        self.fs = fs
        self.mua_data = mua_data
        self.velocity_data = velocity_data
        self.params = params if params is not None else SWRParams()
        self.swr_events = []
        self.event_counter = 0

        print(f"Initialized SWR detector with {self.n_channels} channels and "
              f"{self.n_timepoints / self.fs:.2f} seconds of data")

        # Validate parameters
        self.params.validate_params()

    # -------------------------------------------------
    # PRx: Region-level wrapper to mirror pipeline.detect_swr_by_region
    # -------------------------------------------------
    @staticmethod
    def detect_by_region(region_lfp: dict,
                         fs: float,
                         velocity: np.ndarray | None = None,
                         params: SWRParams | None = None,
                         immobility_mask: np.ndarray | None = None,
                         mua_by_region: dict | None = None,
                         classify: bool = True,
                         channels: str | list[int] = "all",
                         average_mode: bool = False,
                         return_detectors: bool = False):
        """Run SWR detection per region using SWRDetector internally.

        Parameters
        ----------
        region_lfp : dict[str, np.ndarray]
            Mapping of region name to LFP array [n_chan, n_samples].
        fs : float
            Sampling frequency in Hz.
        velocity : np.ndarray, optional
            1D array aligned to LFP timeline used for velocity gating (inside detector).
        params : SWRParams, optional
            Detection parameters; defaults to hippocampal preset if available.
        immobility_mask : np.ndarray, optional
            1D boolean mask (len == n_samples) for post-hoc filtering.
        mua_by_region : dict[str, np.ndarray] | None
            Optional per-region MUA or firing-rate arrays (1D or 2D with matching n_samples).
        classify : bool
            If True, run classification after detection.
        channels : 'all' or list[int]
            Channels to process within each region array.
        average_mode : bool
            If True, average selected channels before detection per region.
        return_detectors : bool
            If True, also return a dict of per-region SWRDetector instances.

        Returns
        -------
        (events_by_region: dict[str, pd.DataFrame], events_all: pd.DataFrame[, detectors_by_region])
        """
        from .params import PRESETS  # local import avoids circulars

        params = params or PRESETS.get("hippocampal", SWRParams())

        events_by_region: dict[str, pd.DataFrame] = {}
        detectors_by_region: dict[str, SWRDetector] = {}

        # Basic validations for optional vectors
        n_samples_ref = None
        for r, arr in region_lfp.items():
            arr = np.asarray(arr)
            if arr.ndim != 2:
                raise ValueError(f"{r}: LFP must be 2D [n_chan, n_samples], got shape {arr.shape}")
            n_samples_ref = arr.shape[1] if n_samples_ref is None else n_samples_ref
            if arr.shape[1] != n_samples_ref:
                raise ValueError(f"All region LFP arrays must share n_samples={n_samples_ref}, got {arr.shape[1]} for {r}")

        if velocity is not None:
            v = np.asarray(velocity).squeeze()
            if n_samples_ref is not None and v.size != n_samples_ref:
                raise ValueError(f"velocity length {v.size} must match LFP n_samples {n_samples_ref}")

        if immobility_mask is not None:
            m = np.asarray(immobility_mask).astype(bool).squeeze()
            if n_samples_ref is not None and m.size != n_samples_ref:
                raise ValueError(f"immobility_mask length {m.size} must match LFP n_samples {n_samples_ref}")

        # Iterate regions
        for region, lfp in region_lfp.items():
            lfp = np.asarray(lfp, float)

            # MUA handling (optional)
            mua = None
            if mua_by_region is not None and region in mua_by_region and mua_by_region[region] is not None:
                mua = np.asarray(mua_by_region[region], float)
                if mua.ndim == 2 and mua.shape[1] != lfp.shape[1]:
                    raise ValueError(f"{region}: MUA 2D must share n_samples={lfp.shape[1]}, got {mua.shape}")
                if mua.ndim == 1 and mua.shape[0] != lfp.shape[1]:
                    raise ValueError(f"{region}: MUA 1D must have n_samples={lfp.shape[1]}, got {mua.shape[0]}")

            det = SWRDetector(lfp_data=lfp,
                               fs=float(fs),
                               mua_data=mua,
                               velocity_data=velocity,
                               params=params)
            det.detect_events(channels=channels, average_mode=average_mode)
            if classify and hasattr(det, "classify_events"):
                det.classify_events()

            # Robust DataFrame extraction
            df = None
            if hasattr(det, "get_events_summary"):
                df = det.get_events_summary()
            if df is None or df is False:
                try:
                    import pandas as _pd
                    df = _pd.DataFrame(det.swr_events)
                except Exception:
                    import pandas as _pd
                    df = _pd.DataFrame()

            if df is not None and not df.empty:
                df = df.copy()
                df["region"] = region

                # Optional immobility filtering: use center time or peak time
                if immobility_mask is not None:
                    t_center = None
                    for c in ("t_peak", "peak_time", "center_time", "t_center"):
                        if c in df.columns:
                            t_center = df[c].to_numpy()
                            break
                    if t_center is None and {"start_time", "end_time"}.issubset(df.columns):
                        t_center = 0.5 * (df["start_time"].to_numpy() + df["end_time"].to_numpy())
                    if t_center is not None:
                        idx = np.clip((np.asarray(t_center) * fs).astype(int), 0, len(immobility_mask) - 1)
                        keep = np.asarray(immobility_mask, bool)[idx]
                        import pandas as _pd
                        df = _pd.DataFrame(df.loc[keep]).reset_index(drop=True)

            events_by_region[region] = df
            if return_detectors:
                detectors_by_region[region] = det

        # Concatenate
        import pandas as _pd
        events_all = _pd.concat([df for df in events_by_region.values() if df is not None],
                                ignore_index=True) if events_by_region else _pd.DataFrame()

        if return_detectors:
            return events_by_region, events_all, detectors_by_region
        return events_by_region, events_all

    # -------------------------------------------------
    # PR4: Basic stats and plotting for notebook parity
    # -------------------------------------------------
    def analyze_basic_stats(self):
        """Compute distributions and summary stats similar to tensor_preparation.

        Returns
        -------
        dict
            counts_by_type, durations, peak_powers, iei, summary
        """
        if not self.swr_events:
            return {
                'counts_by_type': {},
                'durations': np.array([]),
                'peak_powers': np.array([]),
                'iei': np.array([]),
                'summary': {}
            }

        ev = self.swr_events

        # Group type from classification when available
        def _gtype(e):
            if isinstance(e, dict) and 'classification' in e:
                return e['classification'].get('group_type', 'single')
            return 'single'

        types = np.array([_gtype(e) for e in ev])
        unique, counts = np.unique(types, return_counts=True)
        counts_by_type = {str(u): int(c) for u, c in zip(unique, counts)}

        durations = np.array([e['end_time'] - e['start_time'] for e in ev], dtype=float)
        peak_powers = np.array([e.get('peak_power', np.nan) for e in ev], dtype=float)
        starts = np.sort(np.array([e['start_time'] for e in ev], dtype=float))
        iei = np.diff(starts) if starts.size > 1 else np.array([])

        def _summ(a: np.ndarray):
            a = a[np.isfinite(a)]
            if a.size == 0:
                return {'n': 0}
            return {
                'n': int(a.size),
                'mean': float(np.mean(a)),
                'std': float(np.std(a)),
                'median': float(np.median(a)),
                'q25': float(np.quantile(a, 0.25)),
                'q75': float(np.quantile(a, 0.75)),
            }

        summary = {
            'durations': _summ(durations),
            'peak_powers': _summ(peak_powers),
            'iei': _summ(iei)
        }

        return {
            'counts_by_type': counts_by_type,
            'durations': durations,
            'peak_powers': peak_powers,
            'iei': iei,
            'summary': summary
        }

    def plot_basic_stats(self, stats=None):
        """2x3 panel summary: counts by type, duration hist/box, peak power hist, IEI hist/box."""
        if stats is None:
            stats = self.analyze_basic_stats()

        durations = stats.get('durations', np.array([]))
        peak_powers = stats.get('peak_powers', np.array([]))
        iei = stats.get('iei', np.array([]))
        counts = stats.get('counts_by_type', {})

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.ravel()

        # Panel 1: Counts by group type
        types = list(counts.keys())
        vals = [counts[t] for t in types] if types else []
        axes[0].bar(types, vals, color=sns.color_palette('pastel', len(types) if types else 3))
        axes[0].set_title('Event counts by group type')
        axes[0].set_ylabel('Count')

        # Panel 2: Duration distribution
        if durations.size:
            sns.histplot(durations[np.isfinite(durations)], bins=30, ax=axes[1], color='skyblue')
        axes[1].set_title('Duration distribution (s)')

        # Panel 3: Peak power distribution
        if peak_powers.size:
            sns.histplot(peak_powers[np.isfinite(peak_powers)], bins=30, ax=axes[2], color='salmon')
        axes[2].set_title('Peak power (a.u.)')

        # Panel 4: Duration boxplot
        if durations.size:
            axes[3].boxplot(durations[np.isfinite(durations)], vert=True)
        axes[3].set_title('Duration (box)')

        # Panel 5: IEI distribution
        if iei.size:
            sns.histplot(iei[np.isfinite(iei)], bins=30, ax=axes[4], color='plum')
        axes[4].set_title('Inter-event intervals (s)')

        # Panel 6: IEI boxplot
        if iei.size:
            axes[5].boxplot(iei[np.isfinite(iei)], vert=True)
        axes[5].set_title('IEI (box)')

        plt.tight_layout()
        plt.show()

    def detect_events(self, channels='all', average_mode=False):
        """
        Detect SWR events either on individual channels or averaged signal.

        Parameters
        ----------
        channels : list, np.ndarray, or 'all'
            Channels to process. If 'all', uses all channels
        average_mode : bool
            If True, average signals across channels before detection

        Raises
        ------
        ValueError
            If channel indices are invalid
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

        print(f"✓ Detected {len(self.swr_events)} events")

    def _process_signal(self, signal_data, channel_id):
        """
        Process single channel or averaged signal for event detection.

        Parameters
        ----------
        signal_data : np.ndarray
            Signal data to process
        channel_id : int or str
            Channel identifier
        """
        # Initial signal processing
        if self.params.notch_freq:
            notched = self._notch_filter(signal_data, self.params.notch_freq)
        else:
            notched = signal_data

        # Ripple processing
        ripple_filtered = self._bandpass_filter(notched, *self.params.ripple_band)
        ripple_power = np.abs(signal.hilbert(ripple_filtered))
        smooth_ripple_power = np.convolve(ripple_power, np.ones(50) / 50, mode='same') ** 2
        ripple_threshold = (np.mean(smooth_ripple_power) +
                            self.params.threshold_multiplier * np.std(smooth_ripple_power))
        above_ripple = smooth_ripple_power > ripple_threshold

        # MUA processing
        if self.params.enable_mua and self.mua_data is not None:
            window_size = int(0.02 * self.fs)  # 20ms window
            smooth_mua = np.convolve(self.mua_data, np.ones(window_size) / window_size, mode='same')
            mua_threshold = (np.mean(smooth_mua) +
                             self.params.mua_threshold_multiplier * np.std(smooth_mua))
            above_mua = smooth_mua > mua_threshold
        else:
            smooth_mua = None
            above_mua = None

        # Sharp wave processing
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

        # Find threshold crossings
        crossings = np.diff(event_markers.astype(int))
        starts = np.where(crossings == 1)[0]
        ends = np.where(crossings == -1)[0]

        # Handle edge cases
        if len(ends) == 0 or len(starts) == 0:
            return
        if ends[0] < starts[0]:
            ends = ends[1:]
        if starts[-1] > ends[-1]:
            starts = starts[:-1]

        # Process each potential event
        for start, end in zip(starts, ends):
            duration = (end - start) / self.fs

            # Check duration criteria
            valid_duration = (duration >= self.params.mua_min_duration and
                              duration <= self.params.max_duration)

            if not valid_duration:
                continue

            # Determine event type
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

            # Classify event type
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

            # Velocity check
            if self.params.velocity_threshold is not None and self.velocity_data is not None:
                vel_window = self.velocity_data[start:end]
                if np.mean(vel_window) > self.params.velocity_threshold:
                    valid_event = False

            # Sharp wave validation
            if (valid_event and self.params.sharpwave_band and
                    event_type in ['ripple_only', 'ripple_mua']):
                sw_window = smooth_sw_power[start:end]
                if not np.any(sw_window < sw_threshold):
                    valid_event = False

            if not valid_event:
                continue

            # HMM-based edge refinement
            if self.params.use_hmm_edge_detection:
                margin_samples = int(self.params.hmm_margin * self.fs)
                win_start = max(0, start - margin_samples)
                win_end = min(len(smooth_ripple_power), end + margin_samples)
                refined_start, refined_end = self._refine_event_edges_hmm(
                    smooth_ripple_power, win_start, win_end)
                if refined_end > refined_start:
                    start = refined_start
                    end = refined_end
                    duration = (end - start) / self.fs

            # Extract traces
            win_samples = int(self.params.trace_window * self.fs)
            half_win = win_samples // 2
            trace_start = max(0, peak_idx - half_win)
            trace_end = min(len(notched), peak_idx + half_win)

            # Find peaks for classification
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

            # Create event dictionary
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
                'ripple_threshold': float(ripple_threshold),
                'peak_times': peak_times,
                'event_type': event_type,
                'raw_trace': notched[trace_start:trace_end],
                'ripple_trace': ripple_filtered[trace_start:trace_end],
                'mua_trace': smooth_mua[trace_start:trace_end] if smooth_mua is not None else None,
                'ripple_power': ripple_power[trace_start:trace_end],
                'ripple_power_smooth': smooth_ripple_power[trace_start:trace_end],
                'sharpwave_trace': sharpwave_filtered[trace_start:trace_end] if sharpwave_filtered is not None else None,
                'trace_timestamps': np.linspace(
                    peak_time - (peak_idx - trace_start) / self.fs,
                    peak_time + (trace_end - peak_idx) / self.fs,
                    trace_end - trace_start
                )
            }
            self.swr_events.append(new_event)

    def _notch_filter(self, data, notch_freq=60):
        """Apply notch filter to remove line noise."""
        nyquist = self.fs / 2
        b, a = signal.iirnotch(notch_freq / nyquist, 30)
        return signal.filtfilt(b, a, data)

    def _bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter."""
        nyquist = self.fs / 2
        b, a = signal.butter(3, [low_freq / nyquist, high_freq / nyquist], btype='band')
        return signal.filtfilt(b, a, data)

    def _refine_event_edges_hmm(self, smooth_envelope, window_start, window_end):
        """
        Refine event boundaries using HMM state transition detection.

        Parameters
        ----------
        smooth_envelope : np.ndarray
            Smoothed envelope signal (e.g., ripple power)
        window_start : int
            Start index for HMM analysis window
        window_end : int
            End index for HMM analysis window

        Returns
        -------
        tuple
            Refined (start_idx, end_idx) within the overall signal
        """
        segment = smooth_envelope[window_start:window_end]
        if len(segment) < 10:
            return window_start, window_end

        # Reshape for HMM
        observations = segment.reshape(-1, 1)

        try:
            # Fit 2-state Gaussian HMM
            model = GaussianHMM(n_components=2, covariance_type="diag",
                              n_iter=100, random_state=42)
            model.fit(observations)
            states = model.predict(observations)
        except Exception as e:
            print(f"HMM fitting failed: {e}")
            return window_start, window_end

        # Determine event state (higher mean value)
        state_means = [np.mean(observations[states == i]) for i in range(2)]
        event_state = np.argmax(state_means)

        # Find event state indices
        event_indices = np.where(states == event_state)[0]
        if event_indices.size == 0:
            return window_start, window_end

        refined_start = window_start + int(event_indices[0])
        refined_end = window_start + int(event_indices[-1])

        return refined_start, refined_end

    def _merge_close_events(self):
        """
        Merge events that occur within the merge_interval.

        Returns
        -------
        list
            List of merged events
        """
        if not self.swr_events:
            return []

        # Sort events by start time
        sorted_events = sorted(self.swr_events, key=lambda x: x['start_time'])

        merged_events = []
        current_group = [sorted_events[0]]

        for i in range(1, len(sorted_events)):
            current_event = sorted_events[i]
            last_event = current_group[-1]

            # Check if events are close enough to merge
            interval = current_event['start_time'] - last_event['end_time']

            if (interval <= self.params.merge_interval and
                current_event['channel'] == last_event['channel']):
                # Merge events
                merged_start = min(last_event['start_time'], current_event['start_time'])
                merged_end = max(last_event['end_time'], current_event['end_time'])
                merged_peak_time = (current_event['peak_time'] if current_event['peak_power'] > last_event['peak_power']
                                   else last_event['peak_time'])
                merged_peak_power = max(current_event['peak_power'], last_event['peak_power'])

                # Create merged event
                merged_event = {
                    'event_id': last_event['event_id'],  # Keep original ID
                    'channel': last_event['channel'],
                    'start_idx': int(merged_start * self.fs),
                    'end_idx': int(merged_end * self.fs),
                    'start_time': merged_start,
                    'end_time': merged_end,
                    'duration': merged_end - merged_start,
                    'peak_time': merged_peak_time,
                    'peak_power': merged_peak_power,
                    'peak_times': sorted(list(set(last_event['peak_times'] + current_event['peak_times']))),
                    'event_type': 'merged',  # Mark as merged
                    'raw_trace': None,  # Would need to extract new traces
                    'ripple_trace': None,
                    'mua_trace': None,
                    'ripple_power': None,
                    'sharpwave_trace': None,
                    'trace_timestamps': None
                }
                current_group[-1] = merged_event  # Replace last event with merged
            else:
                # Start new group
                merged_events.extend(current_group)
                current_group = [current_event]

        # Add final group
        if current_group:
            merged_events.extend(current_group)

        return merged_events

    def classify_events(self):
        """
        Classify events into singles, doubles, triples, etc. using advanced clustering.
        """
        if not self.swr_events:
            print("No events detected")
            return

        # Merge events that occur within the merge_interval
        self.swr_events = self._merge_close_events()
        if not self.swr_events:
            print("No events detected for classification.")
            return

        # Use improved classification method with DBSCAN clustering
        self.classify_events_improved()

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

        print("Improved Classification Summary:")
        print(f"Singles: {singles}")
        print(f"Doubles: {doubles}")
        print(f"Triples: {triples}")
        print(f"Multiples: {multiples}")

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

    def get_events_summary(self):
        """
        Return a pandas DataFrame with key event parameters.

        Returns
        -------
        pd.DataFrame or None
            Summary DataFrame if events exist, None otherwise
        """
        if not self.swr_events:
            print("No events detected")
            return None

        summary = []
        for event in self.swr_events:
            trace_ref = f"evt_{int(event['event_id'])}"
            event_summary = {
                'event_id': event['event_id'],
                'channel': event['channel'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'duration': event['duration'],
                'peak_time': event['peak_time'],
                'peak_power': event['peak_power'],
                'event_type': event['event_type'],
                'trace_ref': trace_ref,
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

    def get_channel_events(self, channel):
        """
        Get all events from a specific channel.

        Parameters
        ----------
        channel : int
            Channel index

        Returns
        -------
        list
            List of events from the specified channel
        """
        channel_events = [event for event in self.swr_events if event['channel'] == channel]
        if not channel_events:
            print(f"No events found for channel {channel}")
        return channel_events

    def get_event(self, event_id):
        """
        Get a specific event by its ID.

        Parameters
        ----------
        event_id : int
            Event ID to retrieve

        Returns
        -------
        dict or None
            Event dictionary if found, None otherwise
        """
        for event in self.swr_events:
            if event['event_id'] == event_id:
                return event
        print(f"Event ID {event_id} not found")
        return None

    def _build_traces_npz_payload(self):
        """Assemble a dict of per-event trace arrays suitable for np.savez."""
        payload = {}
        for e in self.swr_events:
            key = f"evt_{int(e['event_id'])}"
            # Times
            if e.get('trace_timestamps') is not None:
                payload[f"{key}_t"] = np.asarray(e['trace_timestamps'], dtype=float)
            # Raw and filtered
            if e.get('raw_trace') is not None:
                payload[f"{key}_raw"] = np.asarray(e['raw_trace'], dtype=float)
            if e.get('ripple_trace') is not None:
                payload[f"{key}_rip"] = np.asarray(e['ripple_trace'], dtype=float)
            # Power/envelope variants
            if e.get('ripple_power') is not None:
                payload[f"{key}_env"] = np.asarray(e['ripple_power'], dtype=float)
            if e.get('ripple_power_smooth') is not None:
                payload[f"{key}_env_s"] = np.asarray(e['ripple_power_smooth'], dtype=float)
            # MUA and sharpwave (optional)
            if e.get('mua_trace') is not None:
                payload[f"{key}_mua"] = np.asarray(e['mua_trace'], dtype=float)
            if e.get('sharpwave_trace') is not None:
                payload[f"{key}_sw"] = np.asarray(e['sharpwave_trace'], dtype=float)
        return payload

    def save_events(self, filename: str, include_traces: bool = False):
        """
        Save detected events to a summary table (CSV/Parquet) and optionally an NPZ with traces.

        Parameters
        ----------
        filename : str
            Output filename for the summary table. If endswith '.parquet', try Parquet,
            otherwise CSV. When include_traces=True, also writes '<stem>_traces.npz'.
        include_traces : bool
            When True, saves per-event trace arrays to an NPZ next to the table.

        Returns
        -------
        tuple[str, Optional[str]]
            (path_to_table, path_to_npz_or_None)
        """
        events_df = self.get_events_summary()
        if events_df is None:
            print("No events to save.")
            return None, None

        table_path = filename
        npz_path = None

        # Save table
        try:
            if table_path.lower().endswith(".parquet"):
                events_df.to_parquet(table_path, index=False)
            else:
                # Ensure .csv extension if none given
                root, ext = os.path.splitext(table_path)
                if ext == "":
                    table_path = root + ".csv"
                events_df.to_csv(table_path, index=False)
        except Exception as e:
            # Fallback to CSV if parquet failed
            root, _ = os.path.splitext(table_path)
            table_path = root + ".csv"
            events_df.to_csv(table_path, index=False)
            print(f"Parquet save failed ({e}); wrote CSV instead: {table_path}")

        print(f"✓ Saved {len(events_df)} events to {table_path}")

        # Optionally save traces
        if include_traces and len(self.swr_events) > 0:
            root, _ = os.path.splitext(table_path)
            npz_path = root + "_traces.npz"
            payload = self._build_traces_npz_payload()
            if payload:
                np.savez_compressed(npz_path, **payload)
                print(f"✓ Saved traces to {npz_path}")
            else:
                npz_path = None
                print("No trace payload to save.")

        return table_path, npz_path

    def update_params(self, **kwargs):
        """
        Update detection parameters.

        Parameters
        ----------
        **kwargs
            Parameter names and new values
        """
        self.params.update(**kwargs)
        self.params.validate_params()
        print("✓ Parameters updated successfully")

    def clear_events(self):
        """
        Clear all detected events.
        """
        self.swr_events = []
        self.event_counter = 0
        print("✓ All events cleared")

    def get_basic_stats(self):
        """
        Get basic statistics about detected events.

        Returns
        -------
        dict
            Dictionary containing basic statistics
        """
        if not self.swr_events:
            return None

        durations = [event['duration'] for event in self.swr_events]
        peak_powers = [event['peak_power'] for event in self.swr_events]
        event_types = [event['event_type'] for event in self.swr_events]

        return {
            'total_events': len(self.swr_events),
            'duration_stats': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            },
            'peak_power_stats': {
                'mean': np.mean(peak_powers),
                'std': np.std(peak_powers),
                'min': np.min(peak_powers),
                'max': np.max(peak_powers)
            },
            'event_type_counts': {
                event_type: event_types.count(event_type)
                for event_type in set(event_types)
            }
        }

    def filter_events(self, **criteria):
        """
        Filter events based on specified criteria.

        Parameters
        ----------
        **criteria
            Filtering criteria (e.g., event_type='ripple_only', min_duration=0.05)

        Returns
        -------
        list
            Filtered list of events
        """
        filtered_events = self.swr_events.copy()

        for key, value in criteria.items():
            if key == 'min_duration':
                filtered_events = [e for e in filtered_events if e['duration'] >= value]
            elif key == 'max_duration':
                filtered_events = [e for e in filtered_events if e['duration'] <= value]
            elif key == 'event_type':
                filtered_events = [e for e in filtered_events if e['event_type'] == value]
            elif key == 'min_peak_power':
                filtered_events = [e for e in filtered_events if e['peak_power'] >= value]
            elif key == 'channel':
                filtered_events = [e for e in filtered_events if e['channel'] == value]
            else:
                print(f"Warning: Unknown filter criterion '{key}'")

        return filtered_events

    def __len__(self):
        """Return number of detected events."""
        return len(self.swr_events)

    def __getitem__(self, index):
        """Get event by index."""
        return self.swr_events[index]

    def __iter__(self):
        """Iterate over events."""
        return iter(self.swr_events)
