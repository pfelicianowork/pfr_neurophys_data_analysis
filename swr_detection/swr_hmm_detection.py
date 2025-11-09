import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from sklearn.cluster import DBSCAN  # For improved clustering (if used)
from hmmlearn.hmm import GaussianHMM  # NEW: For HMM-based edge detection
from scipy.signal import butter, filtfilt, hilbert

class SWRHMMParams:
    def __init__(self,
                 # Required parameters
                 ripple_band=(150, 250),
                 threshold_multiplier=3,
                 min_duration=0.03,
                 max_duration=0.4,
                 # Hysteresis threshold parameters
                 use_hysteresis=False,            # Enable hysteresis thresholding
                 hysteresis_low_multiplier=2.0,   # Lower threshold multiplier (in SD)
                #  hysteresis_high_multiplier=3.5,  # Higher threshold multiplier (in SD)
                 hysteresis_confirmation_window=0.015,  # NEW: 15ms confirmation window
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
                 hmm_margin=0.15,                # Margin (in seconds) around candidate event for HMM analysis
                 # Global and multi-state HMM options
                 hmm_states=2,                   # Number of HMM states (legacy, ignored)
                 hmm_states_ripple=2,            # Number of HMM states for ripple power
                 hmm_states_mua=2,               # Number of HMM states for MUA
                 use_global_hmm=True,            # Train a global HMM and reuse for prediction
                 global_hmm_fraction=0.1,        # Fraction of recording to sample for global HMM training
                 # Multivariate HMM option
                 use_multivariate_hmm=False,     # If True, run a joint HMM over [ripple power, MUA]
                 zscore_mua=False,               # If True, z-score MUA before HMM detection
                 # NEW: Advanced feature engineering options
                 use_mua_derivative=False,       # Add rate of change as feature for MUA HMM
                 derivative_sigma=2,             # Smoothing for derivative computation (lower = sharper edges)
                 use_multiscale_features=False,  # Add multi-timescale smoothing features
                 multiscale_sigmas=[1, 3, 5],    # Gaussian smoothing scales (in samples)
                 use_robust_init=False,          # Use median/MAD instead of mean/std for HMM initialization
                 use_directional_hmms=False,     # Use separate HMMs for onset vs offset (only when use_global_hmm=False)
                 # NEW: Preprocessing parameters (Karlsson-style)
                 use_smoothing=True,             # Enable Gaussian smoothing of ripple power
                 smoothing_sigma=0.004,          # Smoothing kernel width in seconds (4ms default)
                 normalization_method='zscore',  # 'zscore' or 'median_mad'
                 normalization_time_range=None   # Optional (start, end) time range for baseline
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
        
        # Hysteresis parameters
        self.use_hysteresis = use_hysteresis
        self.hysteresis_low_multiplier = hysteresis_low_multiplier
        # self.hysteresis_high_multiplier = hysteresis_high_multiplier
        self.hysteresis_confirmation_window = hysteresis_confirmation_window  # NEW

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
        # Global and multi-state HMM options
        self.hmm_states = hmm_states  # legacy, ignored
        self.hmm_states_ripple = hmm_states_ripple
        self.hmm_states_mua = hmm_states_mua
        self.use_global_hmm = use_global_hmm
        self.global_hmm_fraction = global_hmm_fraction
        # Multivariate HMM
        self.use_multivariate_hmm = use_multivariate_hmm
        self.zscore_mua = zscore_mua
        
        # NEW: Advanced feature engineering options
        self.use_mua_derivative = use_mua_derivative
        self.derivative_sigma = derivative_sigma
        self.use_multiscale_features = use_multiscale_features
        self.multiscale_sigmas = multiscale_sigmas
        self.use_robust_init = use_robust_init
        self.use_directional_hmms = use_directional_hmms
        
        # NEW: Preprocessing parameters
        self.use_smoothing = use_smoothing
        self.smoothing_sigma = smoothing_sigma
        self.normalization_method = normalization_method
        self.normalization_time_range = normalization_time_range

    def update(self, **kwargs):
        """Update parameters with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    # ...existing code...

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
# ...existing code...

    # --- PATCH: Store both absolute and z-score thresholds in each event ---
    def _store_event_thresholds(self, mean_power_raw, std_power_raw, ripple_threshold, ripple_low_th, smooth_mua, mua_threshold):
        # For ripple thresholds
        if getattr(self.params, 'normalization_method', None):
            # Detection was in z-score units
            ripple_high_threshold_z = float(self.params.threshold_multiplier)
            ripple_low_threshold_z = float(getattr(self.params, 'hysteresis_low_multiplier', None)) if getattr(self.params, 'use_hysteresis', False) else None
            ripple_high_threshold = mean_power_raw + ripple_high_threshold_z * std_power_raw
            ripple_low_threshold = mean_power_raw + ripple_low_threshold_z * std_power_raw if ripple_low_threshold_z is not None else None
        else:
            # Detection was in absolute units
            ripple_high_threshold = ripple_threshold
            ripple_low_threshold = ripple_low_th if ripple_low_th is not None else None
            ripple_high_threshold_z = (ripple_high_threshold - mean_power_raw) / std_power_raw if std_power_raw > 0 else None
            ripple_low_threshold_z = (ripple_low_threshold - mean_power_raw) / std_power_raw if (ripple_low_threshold is not None and std_power_raw > 0) else None

        # For MUA thresholds
        if smooth_mua is not None and mua_threshold is not None:
            mua_mean = np.mean(smooth_mua)
            mua_std = np.std(smooth_mua) 
            mua_threshold_z = (mua_threshold - mua_mean) / mua_std if mua_std > 0 else None
        else:
            mua_threshold_z = None

        return {
            'ripple_high_threshold': float(ripple_high_threshold) if ripple_high_threshold is not None else None,
            'ripple_low_threshold': float(ripple_low_threshold) if ripple_low_threshold is not None else None,
            'ripple_high_threshold_z': float(ripple_high_threshold_z) if ripple_high_threshold_z is not None else None,
            'ripple_low_threshold_z': float(ripple_low_threshold_z) if ripple_low_threshold_z is not None else None,
            'mua_threshold': float(mua_threshold) if mua_threshold is not None else None,
            'mua_threshold_z': float(mua_threshold_z) if mua_threshold_z is not None else None,
        }

    # In your event creation code (inside detection loop), after computing thresholds, add:
    # thresholds_dict = self._store_event_thresholds(mean_power_raw, std_power_raw, ripple_threshold, ripple_low_th, smooth_mua, mua_threshold)
    # and then update the event dict with: new_event.update(thresholds_dict)
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

    def _compute_multi_method_durations(self, basic_start_idx, basic_end_idx, smooth_ripple_power, 
                                    smooth_mua, ripple_threshold, mua_threshold):
        """
        Compute duration estimates for a single candidate event using 4 different methods.
        
        This method receives the BASIC threshold boundaries (from high threshold crossing)
        and independently computes hysteresis expansion FROM those basic boundaries.
        
        Parameters:
        -----------
        basic_start_idx, basic_end_idx : int
            Event boundaries from basic threshold detection (high threshold only)
            These are NEVER modified and represent the ground truth basic detection
        
        Returns:
        --------
        durations : dict
            Contains start/end indices and durations for all 4 methods:
            - basic: from high threshold crossing (input boundaries)
            - hysteresis: expansion from basic using low threshold + confirmation window
            - hmm_ripple: HMM-based state detection on ripple power
            - hmm_mua: HMM-based state detection on MUA
        """
        durations = {}
        
        # METHOD 1: Basic Threshold (original detection)
        # These come directly from the input and represent pure threshold crossing
        durations['basic_start_idx'] = basic_start_idx
        durations['basic_end_idx'] = basic_end_idx
        durations['basic_duration'] = (basic_end_idx - basic_start_idx) / self.fs
        
        # METHOD 2: Enhanced Hysteresis with Look-Ahead Confirmation
        # Expands FROM basic boundaries using low threshold and confirmation window
        if getattr(self.params, 'use_hysteresis', False):
            mean_power = np.mean(smooth_ripple_power)
            std_power = np.std(smooth_ripple_power)
            low_th = mean_power + self.params.hysteresis_low_multiplier * std_power
            
            # Convert confirmation window from seconds to samples
            confirmation_samples = int(getattr(self.params, 'hysteresis_confirmation_window', 0.070) * self.fs)
            
            # === BACKWARD EXPANSION from basic_start_idx ===
            hyst_start = basic_start_idx
            while hyst_start > 0:
                if smooth_ripple_power[hyst_start - 1] > low_th:
                    hyst_start -= 1
                else:
                    # Look ahead: does power come back above threshold within window?
                    look_ahead_start = max(0, hyst_start - confirmation_samples)
                    look_ahead_window = smooth_ripple_power[look_ahead_start:hyst_start]
                    if len(look_ahead_window) > 0 and np.any(look_ahead_window > low_th):
                        # Power comes back above threshold within window, keep expanding
                        hyst_start -= 1
                    else:
                        # Power stays below for full window, stop expanding
                        break
            # Ensure hysteresis EXPANDS (never contracts)
            hyst_start = min(hyst_start, basic_start_idx)

            # === FORWARD EXPANSION from basic_end_idx ===
            hyst_end = basic_end_idx
            while hyst_end < len(smooth_ripple_power) - 1:
                if smooth_ripple_power[hyst_end] > low_th:
                    hyst_end += 1
                else:
                    # Look ahead to check for sustained drop
                    look_ahead_end = min(len(smooth_ripple_power), hyst_end + confirmation_samples)
                    look_ahead_window = smooth_ripple_power[hyst_end:look_ahead_end]
                    if len(look_ahead_window) > 0 and np.any(look_ahead_window > low_th):
                        # Power comes back above threshold within window, keep expanding
                        hyst_end += 1
                    else:
                        # Power stays below for full window, stop expanding
                        break
            # Ensure hysteresis EXPANDS (never contracts)
            hyst_end = max(hyst_end, basic_end_idx)
            
            durations['hysteresis_start_idx'] = hyst_start
            durations['hysteresis_end_idx'] = hyst_end
            durations['hysteresis_duration'] = (hyst_end - hyst_start) / self.fs
        else:
            # Hysteresis disabled, use basic
            durations['hysteresis_start_idx'] = basic_start_idx
            durations['hysteresis_end_idx'] = basic_end_idx
            durations['hysteresis_duration'] = durations['basic_duration']
        
        # METHOD 3: HMM Edge (Ripple Power)
        # HMM detects state transitions independently from basic boundaries
        if self.params.use_hmm_edge_detection:
            margin_samples = int(self.params.hmm_margin * self.fs)
            hmm_start = max(0, basic_start_idx - margin_samples)
            hmm_end = min(len(smooth_ripple_power), basic_end_idx + margin_samples)
            
            hmm_ripple_start, hmm_ripple_end = self._refine_event_edges_hmm(
                smooth_ripple_power, hmm_start, hmm_end, signal_type='ripple'
            )
            
            if hmm_ripple_start is not None and hmm_ripple_end is not None and hmm_ripple_end > hmm_ripple_start:
                durations['hmm_ripple_start_idx'] = hmm_ripple_start
                durations['hmm_ripple_end_idx'] = hmm_ripple_end
                durations['hmm_ripple_duration'] = (hmm_ripple_end - hmm_ripple_start) / self.fs
            else:
                durations['hmm_ripple_start_idx'] = basic_start_idx
                durations['hmm_ripple_end_idx'] = basic_end_idx
                durations['hmm_ripple_duration'] = durations['basic_duration']
        else:
            durations['hmm_ripple_start_idx'] = basic_start_idx
            durations['hmm_ripple_end_idx'] = basic_end_idx
            durations['hmm_ripple_duration'] = durations['basic_duration']
        
        # METHOD 4: HMM Edge (MUA)
        # HMM detects MUA state transitions independently from basic boundaries
        if self.params.use_hmm_edge_detection and smooth_mua is not None:
            margin_samples = int(self.params.hmm_margin * self.fs)
            hmm_start = max(0, basic_start_idx - margin_samples)
            hmm_end = min(len(smooth_mua), basic_end_idx + margin_samples)
            
            hmm_mua_start, hmm_mua_end = self._refine_event_edges_hmm(
                smooth_mua, hmm_start, hmm_end, signal_type='mua'
            )
            
            if hmm_mua_start is not None and hmm_mua_end is not None and hmm_mua_end > hmm_mua_start:
                durations['hmm_mua_start_idx'] = hmm_mua_start
                durations['hmm_mua_end_idx'] = hmm_mua_end
                durations['hmm_mua_duration'] = (hmm_mua_end - hmm_mua_start) / self.fs
            else:
                durations['hmm_mua_start_idx'] = basic_start_idx
                durations['hmm_mua_end_idx'] = basic_end_idx
                durations['hmm_mua_duration'] = durations['basic_duration']
        else:
            durations['hmm_mua_start_idx'] = basic_start_idx
            durations['hmm_mua_end_idx'] = basic_end_idx
            durations['hmm_mua_duration'] = durations['basic_duration']
        
        return durations

    def _notch_filter(self, data, notch_freq, Q=30):
        """
        Apply a notch filter to remove a specific frequency (e.g., line noise).
        notch_freq: frequency to remove (Hz)
        Q: quality factor (default 30)
        """
        from scipy.signal import iirnotch, filtfilt
        fs = getattr(self, 'fs', None) or getattr(self, 'sampling_frequency', None)
        if fs is None:
            raise AttributeError("Sampling frequency (fs) not set in SWRHMMDetector.")
        b, a = iirnotch(notch_freq, Q, fs)
        return filtfilt(b, a, data)
    
    def _bandpass_filter(self, data, lowcut, highcut, order=4):
        """
        Bandpass filter for LFP data.
        """
        fs = self.fs  # or self.sampling_frequency if that's your attribute
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def _gaussian_smooth(self, data, sigma):
        """
        Apply Gaussian smoothing to the data (Karlsson-style preprocessing).
        
        Parameters
        ----------
        data : np.ndarray
            Input signal to smooth
        sigma : float
            Standard deviation of Gaussian kernel in seconds
            
        Returns
        -------
        smoothed : np.ndarray
            Smoothed signal
        """
        from scipy.ndimage import gaussian_filter1d
        
        # Convert sigma from seconds to samples
        sigma_samples = sigma * self.fs
        
        # Apply Gaussian filter
        smoothed = gaussian_filter1d(data, sigma=sigma_samples, mode='reflect')
        
        return smoothed
    
    def _normalize_signal(self, data, method='zscore', time_range=None):
        """
        Normalize the signal using z-score or robust median/MAD method.
        
        Parameters
        ----------
        data : np.ndarray
            Input signal to normalize
        method : str
            Normalization method: 'zscore' or 'median_mad'
        time_range : tuple of (float, float), optional
            Time range (start, end) in seconds for computing baseline statistics
            
        Returns
        -------
        normalized : np.ndarray
            Normalized signal
        mean_val : float
            Mean or median used for normalization
        std_val : float
            Standard deviation or MAD used for normalization
        """
        # Select data for computing statistics
        if time_range is not None:
            start_idx = int(time_range[0] * self.fs)
            end_idx = int(time_range[1] * self.fs)
            baseline_data = data[start_idx:end_idx]
        else:
            baseline_data = data
        
        if method == 'zscore':
            mean_val = np.mean(baseline_data)
            std_val = np.std(baseline_data)
            normalized = (data - mean_val) / std_val
            
        elif method == 'median_mad':
            from scipy.stats import median_abs_deviation
            median_val = np.median(baseline_data)
            mad_val = median_abs_deviation(baseline_data)
            # Convert MAD to standard deviation equivalent (scale factor 1.4826)
            std_val = 1.4826 * mad_val if mad_val > 0 else 1.0
            normalized = (data - median_val) / std_val
            mean_val = median_val
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, mean_val, std_val
    
    def _initialize_hmm_robust(self, signal, n_components=2):
        """
        Initialize HMM with robust median/MAD statistics instead of mean/std.
        Makes initialization less sensitive to outliers and transient spikes.
        
        Parameters
        ----------
        signal : np.ndarray
            1D signal to analyze
        n_components : int
            Number of GMM components for initialization
            
        Returns
        -------
        tuple
            (means, covariances) for HMM initialization
        """
        from scipy.stats import median_abs_deviation
        from sklearn.mixture import GaussianMixture
        
        # Use median/MAD for robust statistics
        median = np.median(signal)
        mad = median_abs_deviation(signal, scale='normal')  # scale='normal' converts to std-like units
        
        # Initialize GMM component means based on median + multiples of MAD
        if n_components == 2:
            means_init = np.array([[median], [median + 3*mad]])
        elif n_components == 3:
            means_init = np.array([[median], [median + 2*mad], [median + 4*mad]])
        else:
            # For other n_components, space evenly
            means_init = np.linspace(median, median + 3*mad, n_components).reshape(-1, 1)
        
        # Fit GMM with initialized means
        gmm = GaussianMixture(n_components=n_components, means_init=means_init, 
                             covariance_type='diag', n_init=5, random_state=42)
        gmm.fit(signal.reshape(-1, 1))
        
        return gmm.means_, gmm.covariances_
    
    def _create_derivative_features(self, signal):
        """
        Add rate of change (derivative) as an additional feature.
        This helps HMM identify sharp onset/offset edges.
        
        Parameters
        ----------
        signal : np.ndarray
            1D signal (e.g., smoothed MUA)
            
        Returns
        -------
        np.ndarray
            2D array with shape (n_samples, 2): [signal, derivative]
        """
        from scipy.ndimage import gaussian_filter1d
        
        # Smooth the signal first to reduce noise in derivative
        deriv_sigma = getattr(self.params, 'derivative_sigma', 2)
        smooth_signal = gaussian_filter1d(signal, sigma=deriv_sigma)
        
        # Compute derivative
        derivative = np.gradient(smooth_signal)
        
        # Stack features
        features = np.column_stack([smooth_signal, derivative])
        
        return features
    
    def _create_multiscale_features(self, signal, scales=None):
        """
        Create features at multiple timescales to capture both fast and slow dynamics.
        Includes both smoothed signal and derivatives at each scale.
        
        Parameters
        ----------
        signal : np.ndarray
            1D signal
        scales : list of float, optional
            Gaussian smoothing sigmas (in samples). Default uses params.multiscale_sigmas
            
        Returns
        -------
        np.ndarray
            2D array with shape (n_samples, 2*n_scales): [smooth_scale1, deriv_scale1, smooth_scale2, ...]
        """
        from scipy.ndimage import gaussian_filter1d
        
        if scales is None:
            scales = getattr(self.params, 'multiscale_sigmas', [1, 3, 5])
        
        features = []
        for sigma in scales:
            smooth = gaussian_filter1d(signal, sigma=sigma)
            derivative = np.gradient(smooth)
            features.append(smooth)
            features.append(derivative)
        
        return np.column_stack(features)
    
    def _prepare_hmm_features(self, signal, signal_type='ripple'):
        """
        Prepare features for HMM based on enabled feature engineering options.
        Consolidates derivative and multiscale feature creation.
        
        Parameters
        ----------
        signal : np.ndarray
            1D signal (ripple power or MUA)
        signal_type : str
            'ripple' or 'mua'
            
        Returns
        -------
        np.ndarray
            Feature array with shape (n_samples, n_features)
        """
        # Start with base signal
        features = signal.reshape(-1, 1)
        
        # Only apply advanced features to MUA (ripple stays simple)
        if signal_type == 'mua':
            # Add derivative if enabled
            if getattr(self.params, 'use_mua_derivative', False):
                features = self._create_derivative_features(signal)
                # Debug: first call only
                if not hasattr(self, '_derivative_debug_shown'):
                    print(f"[HMM Feature Engineering] Using MUA derivative features: shape={features.shape}")
                    self._derivative_debug_shown = True
            
            # Add multiscale features if enabled (replaces derivative if both are on)
            if getattr(self.params, 'use_multiscale_features', False):
                features = self._create_multiscale_features(signal)
                # Debug: first call only
                if not hasattr(self, '_multiscale_debug_shown'):
                    scales = getattr(self.params, 'multiscale_sigmas', [1, 3, 5])
                    print(f"[HMM Feature Engineering] Using multiscale features: shape={features.shape}, scales={scales}")
                    self._multiscale_debug_shown = True
        
        return features
    
    def _refine_event_edges_hmm(self, smooth_envelope, window_start, window_end, signal_type='ripple'):
        """
        Refine the event boundaries using HMM state transition detection.
        Enhanced version with signal-type awareness for better MUA detection.
        
        Parameters
        ----------
        smooth_envelope : np.ndarray
            The smoothed signal (ripple power or MUA)
        window_start : int
            Start index of the window to analyze
        window_end : int
            End index of the window to analyze
        signal_type : str, default='ripple'
            Type of signal being analyzed ('ripple' or 'mua')
            
        Returns
        -------
        tuple
            (refined_start_idx, refined_end_idx) within the overall signal
        """
        segment = smooth_envelope[window_start:window_end]
        # Optionally z-score MUA before HMM detection
        if signal_type == 'mua' and getattr(self.params, 'zscore_mua', False):
            segment = (segment - np.mean(segment)) / np.std(segment)
        if len(segment) < 10:
            # Not enough samples to perform reliable HMM estimation.
            return window_start, window_end

        # NEW: Prepare features using advanced feature engineering if enabled
        observations = self._prepare_hmm_features(segment, signal_type=signal_type)

        # Determine desired number of states
        if signal_type == 'ripple':
            n_states = getattr(self.params, 'hmm_states_ripple', 2)
        elif signal_type == 'mua':
            n_states = getattr(self.params, 'hmm_states_mua', 2)
        else:
            n_states = 2

        # Use global pre-trained HMM if requested and available
        # Use global HMM only for MUA, always local for ripple
        if signal_type == 'mua':
            use_global = getattr(self.params, 'use_global_hmm', False)
        else:
            use_global = False
        model = self.global_hmms.get(signal_type, None) if use_global else None
        states = None
        
        # Try to use global HMM if available
        if model is not None:
            try:
                states = model.predict(observations)
            except Exception as e:
                print(f"Global HMM predict failed for {signal_type}: {e}")
                states = None

        # Otherwise, fit or initialize a local model
        if states is None:
            # For MUA and 3 states, prefer GMM-based initialization to capture baseline/shoulder/core
            if signal_type == 'mua' and n_states == 3:
                try:
                    from sklearn.mixture import GaussianMixture
                    
                    # Use robust initialization if enabled
                    if getattr(self.params, 'use_robust_init', False):
                        gm_means, gm_covs = self._initialize_hmm_robust(observations[:, 0], n_components=3)
                    else:
                        gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
                        gmm.fit(observations)
                        gm_means = gmm.means_
                        gm_covs = gmm.covariances_
                    
                    # Ensure proper shape for multi-dimensional features
                    n_features = observations.shape[1]
                    if gm_means.shape[1] == 1 and n_features > 1:
                        # Expand means to match feature dimensions (replicate across features)
                        gm_means = np.tile(gm_means, (1, n_features))
                        gm_covs = np.tile(gm_covs, (1, n_features))
                    
                    # Create HMM with init_params='st' to preserve our means/covars initialization
                    model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, 
                                       random_state=42, init_params='st')  # Only init start/transition probs
                    model.means_ = gm_means
                    # hmmlearn expects covars_ in different shapes depending on covariance_type
                    model.covars_ = gm_covs
                    # Conservative transition matrix (prefer staying in same state)
                    tm = np.eye(3) * 0.95
                    tm[tm == 0] = 0.025
                    model.transmat_ = tm
                    model.startprob_ = np.array([0.9, 0.05, 0.05])
                    states = model.predict(observations)
                except Exception:
                    # Fallback: fit a local HMM directly
                    try:
                        model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, random_state=42)
                        model.fit(observations)
                        states = model.predict(observations)
                    except Exception as e:
                        print(f"Local 3-state HMM failed (MUA): {e}")
                        return window_start, window_end
            else:
                # Default behavior: fit an n_state HMM locally
                try:
                    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=42)
                    model.fit(observations)
                    states = model.predict(observations)
                except Exception as e:
                    print(f"Local HMM fitting failed ({signal_type}): {e}")
                    return window_start, window_end

        # Map states to activity levels by mean observation
        # When using multi-dimensional features (e.g., signal + derivative),
        # rank states by the FIRST feature (the actual signal amplitude), not derivatives
        unique_states = np.unique(states)
        state_means = []
        for s in unique_states:
            if np.any(states == s):
                # Use only the first feature (signal amplitude) for ranking
                state_means.append((s, np.mean(observations[states == s, 0])))
            else:
                state_means.append((s, -np.inf))

        # Sort states by mean (low -> high)
        state_means_sorted = sorted(state_means, key=lambda x: x[1])
        sorted_states = [s for s, _ in state_means_sorted]

        if len(sorted_states) < n_states:
            # Ensure we have the expected number of states; fallback safe return
            return window_start, window_end

        if n_states == 2:
            # event is the higher-mean state
            event_state = sorted_states[-1]
            event_mask = (states == event_state)
        else:
            # n_states == 3: When using derivative features, we want SHARP edges
            # So we use ONLY the peak state (highest mean), not shoulder+core
            # This allows derivative information to define precise onset/offset
            if signal_type == 'mua' and getattr(self.params, 'use_mua_derivative', False):
                # Derivative mode: use only the peak state for sharp boundaries
                event_state = sorted_states[-1]
                event_mask = (states == event_state)
                if not hasattr(self, '_derivative_sharp_edge_debug'):
                    print(f"[HMM Edge Detection] Using derivative-based SHARP edge mode: only peak state (state {event_state})")
                    self._derivative_sharp_edge_debug = True
            else:
                # Non-derivative mode: treat shoulder + core (top two states) as event
                event_state_core = sorted_states[-1]
                event_state_shoulder = sorted_states[-2]
                event_mask = (states == event_state_core) | (states == event_state_shoulder)
                if not hasattr(self, '_derivative_sharp_edge_debug'):
                    print(f"[HMM Edge Detection] Using standard 3-state mode: shoulder+core (states {event_state_shoulder},{event_state_core})")
                    self._derivative_sharp_edge_debug = True

        event_indices = np.where(event_mask)[0]
        if event_indices.size == 0:
            return window_start, window_end

        refined_start = window_start + int(event_indices[0])
        refined_end = window_start + int(event_indices[-1])
        return refined_start, refined_end

    def _refine_event_edges_hmm_multivariate(self, smooth_ripple_power, smooth_mua, window_start, window_end):
        """
        Refine event boundaries using a multivariate HMM over [ripple power, MUA].

        Parameters
        ----------
        smooth_ripple_power : np.ndarray
            Smoothed ripple power envelope (1D)
        smooth_mua : np.ndarray
            Smoothed MUA (1D)
        window_start, window_end : int
            Index range to analyze (absolute indices)

        Returns
        -------
        tuple[int, int] or (None, None)
            Refined (start_idx, end_idx) within the overall signal.
        """
        seg_rip = smooth_ripple_power[window_start:window_end]
        seg_mua = smooth_mua[window_start:window_end]
        if len(seg_rip) < 10 or len(seg_rip) != len(seg_mua):
            return None, None

        X = np.column_stack([seg_rip, seg_mua])

        # Use MUA HMM states for multivariate HMM (usually for MUA refinement)
        n_states = getattr(self.params, 'hmm_states_mua', 2)
        # Use global joint HMM if requested
        use_global = getattr(self.params, 'use_global_hmm', False)
        model = self.global_hmms.get('joint', None) if use_global else None
        states = None
        # Use global joint HMM if available
        if getattr(self.params, 'use_global_hmm', False):
            model = self.global_hmms.get('joint', None)
            if model is not None:
                try:
                    states = model.predict(X)
                except Exception as e:
                    print(f"Global joint HMM predict failed: {e}")
                    states = None

        # Fit local model if needed
        if states is None:
            try:
                if n_states == 3:
                    # Try GMM initialization for stability
                    from sklearn.mixture import GaussianMixture
                    gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
                    gmm.fit(X)
                    # Create HMM with init_params='st' to preserve GMM initialization
                    model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, 
                                       random_state=42, init_params='st')
                    model.means_ = gmm.means_
                    model.covars_ = gmm.covariances_
                    tm = np.eye(3) * 0.95
                    tm[tm == 0] = 0.025
                    model.transmat_ = tm
                    model.startprob_ = np.array([0.9, 0.05, 0.05])
                    states = model.predict(X)
                else:
                    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=42)
                    model.fit(X)
                    states = model.predict(X)
            except Exception as e:
                # Fallback: simple local fit
                try:
                    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=42)
                    model.fit(X)
                    states = model.predict(X)
                except Exception as e2:
                    print(f"Local joint HMM failed: {e2}")
                    return None, None

        # Rank states by mean total activation (ripple + MUA)
        unique_states = np.unique(states)
        state_scores = []
        for s in unique_states:
            mask = (states == s)
            if np.any(mask):
                state_scores.append((s, np.mean(X[mask, 0] + X[mask, 1])))
            else:
                state_scores.append((s, -np.inf))
        ranked = [s for s, _ in sorted(state_scores, key=lambda t: t[1])]
        if len(ranked) < n_states:
            return None, None

        if n_states == 2:
            evt_state = ranked[-1]
            mask = (states == evt_state)
        else:
            core = ranked[-1]
            shoulder = ranked[-2]
            mask = (states == core) | (states == shoulder)

        idx = np.where(mask)[0]
        if idx.size == 0:
            return None, None
        return window_start + int(idx[0]), window_start + int(idx[-1])

    def _find_ripple_boundaries(self, ripple_power, start_idx, end_idx, threshold):
        """
        Find precise start/end times when ripple power crosses threshold.
        
        Parameters
        ----------
        ripple_power : np.ndarray
            Smoothed ripple power envelope
        start_idx : int
            Candidate event start index
        end_idx : int
            Candidate event end index
        threshold : float
            Ripple power threshold
            
        Returns
        -------
        tuple
            (start_sample, end_sample) where ripple power crosses threshold
        """
        # Extract segment
        segment = ripple_power[start_idx:end_idx]
        above = segment > threshold
        
        if not above.any():
            return start_idx, end_idx
        
        # Find threshold crossings
        crossings = np.diff(above.astype(int))
        starts = np.where(crossings == 1)[0]
        ends = np.where(crossings == -1)[0]
        
        # Handle edge cases
        if len(starts) == 0:
            if above[0]:  # Already above threshold at start
                starts = [0]
            else:
                return start_idx, end_idx
                
        if len(ends) == 0:
            if above[-1]:  # Still above threshold at end
                ends = [len(above) - 1]
            else:
                return start_idx, end_idx
        
        # Ensure we have paired crossings
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            ends = np.append(ends, len(above) - 1)
        
        if len(starts) == 0 or len(ends) == 0:
            return start_idx, end_idx
        
        # Return first crossing pair (main event)
        ripple_start = start_idx + starts[0]
        ripple_end = start_idx + ends[0] + 1  # +1 to include the last sample
        
        return ripple_start, ripple_end
    
    def _find_mua_boundaries(self, mua, start_idx, end_idx, threshold):
        """
        Find precise start/end times when MUA crosses threshold.

        Parameters
        ----------
        mua : np.ndarray
            Smoothed MUA signal
        start_idx : int
            Candidate event start index
        end_idx : int
            Candidate event end index
        threshold : float
            MUA threshold

        Returns
        -------
        tuple
            (start_sample, end_sample) where MUA crosses threshold
        """
        # Extract segment
        segment = mua[start_idx:end_idx]
        above = segment > threshold

        if not above.any():
            return start_idx, end_idx

        # Find threshold crossings
        crossings = np.diff(above.astype(int))
        starts = np.where(crossings == 1)[0]
        ends = np.where(crossings == -1)[0]

        # Handle edge cases
        if len(starts) == 0:
            if above[0]:  # Already above threshold at start
                starts = [0]
            else:
                return start_idx, end_idx

        if len(ends) == 0:
            if above[-1]:  # Still above threshold at end
                ends = [len(above) - 1]
            else:
                return start_idx, end_idx

        # Ensure we have paired crossings
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            ends = np.append(ends, len(above) - 1)

        if len(starts) == 0 or len(ends) == 0:
            return start_idx, end_idx

        # Return first crossing pair (main event)
        mua_start = start_idx + starts[0]
        mua_end = start_idx + ends[0] + 1  # +1 to include the last sample

        return mua_start, mua_end

    def _find_mua_boundary_expansion(self, smooth_mua, ripple_start_idx, ripple_end_idx,
                                     mua_threshold, expansion_factor=1.5):
        """
        Fall back to threshold-based MUA expansion when HMM fails.
        Uses threshold crossings with expanded window to find MUA boundaries.

        Parameters
        ----------
        smooth_mua : np.ndarray
            Z-score normalized MUA signal
        ripple_start_idx : int
            Ripple start time index
        ripple_end_idx : int
            Ripple end time index
        mua_threshold : float
            MUA threshold (z-score units)
        expansion_factor : float, default=1.5
            How much to expand the search window beyond ripple boundaries

        Returns
        -------
        tuple
            (mua_start_idx, mua_end_idx) with fallback to ripple boundaries if no crossings
        """
        # Expand search window around ripple event
        ripple_duration = ripple_end_idx - ripple_start_idx
        expansion = int(expansion_factor * ripple_duration)

        search_start = max(0, ripple_start_idx - expansion)
        search_end = min(len(smooth_mua), ripple_end_idx + expansion)

        # Extract segment for analysis
        segment = smooth_mua[search_start:search_end]
        above_threshold = segment > mua_threshold

        if not above_threshold.any():
            # No MUA activity detected, use ripple boundaries
            return ripple_start_idx, ripple_end_idx

        # Find threshold crossings in the expanded window
        crossings = np.diff(above_threshold.astype(int))
        starts = np.where(crossings == 1)[0]
        ends = np.where(crossings == -1)[0]

        # Handle edge cases
        if len(starts) == 0:
            if above_threshold[0]:  # Already above threshold
                starts = [0]
        if len(ends) == 0:
            if above_threshold[-1]:  # Still above threshold
                ends = [len(above_threshold) - 1]

        # Find the crossing pair that overlaps with or is closest to ripple window
        if len(starts) > 0 and len(ends) > 0:
            # Convert relative indices to absolute
            abs_starts = search_start + starts
            abs_ends = search_start + ends + 1  # +1 to include last sample

            # Find crossings that overlap with ripple window
            ripple_center = (ripple_start_idx + ripple_end_idx) / 2
            valid_crossings = []

            for s, e in zip(abs_starts, abs_ends):
                # Check if this crossing pair overlaps with ripple window
                overlap = min(e, ripple_end_idx) - max(s, ripple_start_idx) > 0
                if overlap:
                    valid_crossings.append((s, e))

            if valid_crossings:
                # Use the crossing pair with most overlap
                crossings_with_overlap = []
                for s, e in valid_crossings:
                    overlap_start = max(s, ripple_start_idx)
                    overlap_end = min(e, ripple_end_idx)
                    overlap_length = overlap_end - overlap_start
                    crossings_with_overlap.append(((s, e), overlap_length))

                # Select crossing with maximum overlap
                best_crossing = max(crossings_with_overlap, key=lambda x: x[1])[0]
                return best_crossing

        # Fallback: if no valid crossings, try to extend from ripple boundaries
        ripple_in_segment_start = ripple_start_idx - search_start
        ripple_in_segment_end = ripple_end_idx - search_start

        # Search backward from ripple start for where MUA drops below threshold
        start_idx = ripple_start_idx
        for i in range(max(0, ripple_in_segment_start), -1, -1):
            if segment[i] <= mua_threshold:
                # Found the point where MUA drops to or below threshold
                break
            start_idx = search_start + i

        # Search forward from ripple end for where MUA drops below threshold
        end_idx = ripple_end_idx
        for i in range(min(len(segment)-1, ripple_in_segment_end), len(segment)):
            if segment[i] <= mua_threshold:
                # Found the point where MUA drops to or below threshold
                end_idx = search_start + i
                break

        # Ensure we don't return bounds that are too far from ripple event
        max_expansion = int(2 * ripple_duration)
        final_start = max(ripple_start_idx - max_expansion, start_idx)
        final_end = min(ripple_end_idx + max_expansion, end_idx)

        return final_start, final_end

    def _compute_mua_edges(self, smooth_mua, seed_start=None, seed_end=None, peak_idx=None, margin_samples=0):
        """
        Compute MUA HMM edges using a provided seed window (start/end) or centered at peak_idx.

        Parameters
        ----------
        smooth_mua : np.ndarray
            Pre-smoothed MUA vector (1D, length == recording samples)
        seed_start, seed_end : int or None
            Optional seed window in absolute indices. If provided and valid, a margin will be applied.
        peak_idx : int or None
            Fallback center index (absolute) if no seed window is provided.
        margin_samples : int
            Extra samples to include on each side of the seed window (or around the peak) for HMM.

        Returns
        -------
        tuple[int|None, int|None]
            Absolute (start_idx, end_idx) for MUA derived via HMM, or (None, None) on failure.
        """
        n = len(smooth_mua)
        if n == 0:
            return None, None

        # Select analysis window
        if seed_start is not None and seed_end is not None and seed_end > seed_start:
            win_start = max(0, int(seed_start) - margin_samples)
            win_end = min(n, int(seed_end) + margin_samples)
        else:
            # Fallback: center on peak with a reasonable minimum half-width (>=150 ms)
            half = max(margin_samples, int(0.15 * self.fs))
            if peak_idx is None:
                peak_idx = n // 2
            win_start = max(0, int(peak_idx) - half)
            win_end = min(n, int(peak_idx) + half)

        seg = smooth_mua[win_start:win_end]
        if len(seg) < 10:
            return None, None

        try:
            s_rel, e_rel = self._refine_event_edges_hmm(seg, 0, len(seg), signal_type='mua')
        except Exception:
            return None, None

        s_idx = win_start + int(s_rel)
        e_idx = win_start + int(e_rel)
        if e_idx <= s_idx:
            return None, None
        return s_idx, e_idx

    def _detect_events_hysteresis(self, signal, high_threshold, low_threshold,
                              min_duration_samples, max_duration_samples):
        """
        Hysteresis thresholding with temporal confirmation:
        
        * Event begins when ripple crosses HIGH threshold.
        * Event ends ONLY if ripple stays below LOW threshold for a sustained window.
        * Prevents fragmentation when power briefly dips during an event.
        """
        events = []
        in_event = False
        start_idx = None

        confirmation_samples = int(
            getattr(self.params, 'hysteresis_confirmation_window', 0.015) * self.fs
        )

        below_low_counter = 0

        for i, value in enumerate(signal):
            
            # --- Event starts only when high threshold is crossed ---
            if not in_event and value > high_threshold:
                in_event = True
                start_idx = i
                below_low_counter = 0
                continue

            # --- Inside event ---
            if in_event:
                if value < low_threshold:
                    below_low_counter += 1

                    # Terminate if below low-th for long enough
                    if below_low_counter >= confirmation_samples:
                        end_idx = i - confirmation_samples
                        duration = end_idx - start_idx
                        if min_duration_samples <= duration <= max_duration_samples:
                            events.append((start_idx, end_idx))
                        in_event = False
                        start_idx = None
                        below_low_counter = 0
                else:
                    # reset counter if recovered above low threshold
                    below_low_counter = 0

        # If still inside an event at the end
        if in_event and start_idx is not None:
            end_idx = len(signal) - 1
            duration = end_idx - start_idx
            if min_duration_samples <= duration <= max_duration_samples:
                events.append((start_idx, end_idx))

        return events

    def __init__(self, lfp_data, fs, mua_data=None, velocity_data=None, params=None):
        self.multi_region = isinstance(lfp_data, dict)
        self.fs = fs
        self.params = params if params is not None else SWRHMMParams()
        self.event_counter = 0
    # Use self.params.hmm_states_ripple and self.params.hmm_states_mua directly in methods
        
        # Check for incompatible parameter combinations and warn user
        if getattr(self.params, 'use_global_hmm', False) and getattr(self.params, 'use_directional_hmms', False):
            print("\n" + "="*60)
            print("WARNING: use_directional_hmms=True is incompatible with use_global_hmm=True")
            print("Directional HMMs require local training for each event's onset/offset.")
            print("Disabling use_directional_hmms. To use directional HMMs, set use_global_hmm=False.")
            print("="*60 + "\n")
            self.params.use_directional_hmms = False

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

        # Storage for any global HMMs (e.g., {'mua': GaussianHMM(...)})
        self.global_hmms = {}

        # Compute global ripple power for z-score normalization
        if self.multi_region:
            # Average across regions and channels
            ripple_power_list = []
            for region, arr in self.lfp_data.items():
                mean_lfp = np.mean(arr, axis=0)
                ripple_filtered = self._bandpass_filter(mean_lfp, *self.params.ripple_band)
                ripple_envelope = np.abs(hilbert(ripple_filtered))
                ripple_power = ripple_envelope ** 2
                ripple_power_list.append(ripple_power)
            self.ripple_power = np.mean(np.vstack(ripple_power_list), axis=0)
        else:
            # Single region: average across channels
            mean_lfp = np.mean(self.lfp_data, axis=0)
            ripple_filtered = self._bandpass_filter(mean_lfp, *self.params.ripple_band)
            ripple_envelope = np.abs(hilbert(ripple_filtered))
            self.ripple_power = ripple_envelope ** 2

        # Apply smoothing if enabled
        if getattr(self.params, 'use_smoothing', False):
            self.ripple_power = self._gaussian_smooth(self.ripple_power, self.params.smoothing_sigma)

        # Apply normalization to global ripple power if enabled
        if getattr(self.params, 'normalization_method', None):
            self.ripple_power, _, _ = self._normalize_signal(
                self.ripple_power,
                method=self.params.normalization_method,
                time_range=getattr(self.params, 'normalization_time_range', None)
            )

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
        if getattr(self.params, 'use_hysteresis', False):
            print(f"Hysteresis thresholding: ENABLED")
            print(f"  High threshold: {getattr(self.params, 'hysteresis_high_multiplier', 3.5)} SD")
            print(f"  Low threshold: {getattr(self.params, 'hysteresis_low_multiplier', 2.0)} SD")
        else:
            print(f"Hysteresis thresholding: Disabled")
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
            print(f"HMM states (ripple): {getattr(self.params, 'hmm_states_ripple', 2)}")
            print(f"HMM states (MUA): {getattr(self.params, 'hmm_states_mua', 2)}")
            print(f"Use global HMM: {getattr(self.params, 'use_global_hmm', False)} (fraction={getattr(self.params, 'global_hmm_fraction', 0.05)})")
            print(f"Use multivariate HMM: {getattr(self.params, 'use_multivariate_hmm', False)}")
            print(f"Z-score MUA: {getattr(self.params, 'zscore_mua', False)}")
            
            # Show advanced feature engineering options
            if getattr(self.params, 'use_mua_derivative', False):
                print(f"  ✓ MUA derivative features enabled")
            if getattr(self.params, 'use_multiscale_features', False):
                scales = getattr(self.params, 'multiscale_sigmas', [1, 3, 5])
                print(f"  ✓ Multiscale features enabled (sigmas: {scales})")
            if getattr(self.params, 'use_robust_init', False):
                print(f"  ✓ Robust median/MAD initialization enabled")
            if getattr(self.params, 'use_directional_hmms', False):
                print(f"  ✓ Directional HMMs (onset/offset) enabled")

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

        # Prefer ripple-based start time for merge ordering
        # Ensure key is always numeric: fall back to legacy start_time when ripple_start_time is None/missing
        def _sort_key(ev):
            rst = ev.get('ripple_start_time')
            if rst is None:
                return ev.get('start_time', 0.0)
            return rst
        sorted_events = sorted(self.swr_events, key=_sort_key)
        merged = []
        current_merge = None

        for event in sorted_events:
            if current_merge is None:
                current_merge = event.copy()
                continue

            # Use ripple end/start times when available; otherwise fall back to legacy unified times
            prev_end = current_merge.get('ripple_end_time') or current_merge.get('end_time', 0.0)
            curr_start = event.get('ripple_start_time') or event.get('start_time', 0.0)
            interval = curr_start - prev_end

            if interval < self.params.merge_interval:
                # Update the merged event to include the full duration of all combined events
                # Maintain both ripple-only and legacy unified times
                current_merge['ripple_start_time'] = min(
                    current_merge.get('ripple_start_time') or current_merge.get('start_time', 0.0),
                    event.get('ripple_start_time') or event.get('start_time', 0.0)
                )
                current_merge['ripple_end_time'] = max(
                    current_merge.get('ripple_end_time') or current_merge.get('end_time', 0.0),
                    event.get('ripple_end_time') or event.get('end_time', 0.0)
                )
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

    def train_global_hmm(self, sig='mua', n_states=None, fraction=None, random_state=42):
        """
        Train a global Gaussian HMM on a sampled portion of the recording.

        Parameters
        ----------
        sig : str
            'mua', 'ripple', or 'joint' - which signal(s) to use for training
        n_states : int or None
            Number of HMM states (overrides params.hmm_states if provided)
        fraction : float or None
            Fraction of the recording to sample for training (overrides params.global_hmm_fraction)
        random_state : int
            RNG seed for reproducibility

        Returns
        -------
        model : GaussianHMM
            Trained HMM instance stored in self.global_hmms[sig]
        """
        if n_states is None:
            n_states = getattr(self.params, 'hmm_states', 2)
        if fraction is None:
            fraction = getattr(self.params, 'global_hmm_fraction', 0.05)

        # Extract signal vector
        if sig == 'mua':
            data_mua = self.mua_data
            if data_mua is None:
                raise ValueError("No MUA data available for global HMM training")
            data = data_mua
            is_joint = False
        elif sig == 'ripple':
            # compute ripple power across an example channel (mean across channels if multi)
            if self.multi_region:
                # concatenate across regions
                ripple_list = []
                for region, arr in self.lfp_data.items():
                    # average across channels
                    mean_lfp = np.mean(arr, axis=0)
                    ripple_filtered = self._bandpass_filter(mean_lfp, *self.params.ripple_band)
                    ripple_envelope = np.abs(hilbert(ripple_filtered))
                    ripple_power = ripple_envelope ** 2
                    ripple_list.append(ripple_power)
                data = np.mean(np.vstack(ripple_list), axis=0)
            else:
                mean_lfp = np.mean(self.lfp_data, axis=0)
                ripple_filtered = self._bandpass_filter(mean_lfp, *self.params.ripple_band)
                ripple_envelope = np.abs(hilbert(ripple_filtered))
                data = ripple_envelope ** 2
            is_joint = False
        elif sig == 'joint':
            # Build 2D features [ripple_power, MUA]
            if self.mua_data is None:
                raise ValueError("No MUA data available for joint global HMM training")
            if self.multi_region:
                ripple_list = []
                for region, arr in self.lfp_data.items():
                    mean_lfp = np.mean(arr, axis=0)
                    ripple_filtered = self._bandpass_filter(mean_lfp, *self.params.ripple_band)
                    ripple_envelope = np.abs(hilbert(ripple_filtered))
                    ripple_power = ripple_envelope ** 2
                    ripple_list.append(ripple_power)
                ripple_vec = np.mean(np.vstack(ripple_list), axis=0)
                # Assume self.mua_data is a 1D array for joint training; if it's region-specific, user should pass appropriate vector
                if not isinstance(self.mua_data, np.ndarray) or self.mua_data.ndim != 1:
                    raise ValueError("For multi-region joint training, provide a 1D MUA vector compatible with ripple time length")
                mua_vec = self.mua_data
            else:
                mean_lfp = np.mean(self.lfp_data, axis=0)
                ripple_filtered = self._bandpass_filter(mean_lfp, *self.params.ripple_band)
                ripple_envelope = np.abs(hilbert(ripple_filtered))
                ripple_vec = ripple_envelope ** 2
                mua_vec = self.mua_data
                if mua_vec is None:
                    raise ValueError("No MUA data available for joint global HMM training")
            if len(ripple_vec) != len(mua_vec):
                raise ValueError("Ripple and MUA vectors must have the same length for joint HMM training")
            data = np.column_stack([ripple_vec, mua_vec])
            is_joint = True
        else:
            raise ValueError("sig must be 'mua', 'ripple', or 'joint'")

        if data is None:
            raise ValueError(f"No data available for signal '{sig}' to train global HMM")

        # NEW APPROACH: Sample continuous event segments instead of random timepoints
        # This preserves temporal structure and event dynamics
        L = len(data) if not is_joint else data.shape[0]
        rng = np.random.RandomState(random_state)
        
        # Step 1: Find candidate events using threshold detection
        if is_joint:
            # For joint, use ripple power (first column) for threshold detection
            signal_for_threshold = data[:, 0]
        else:
            signal_for_threshold = data if data.ndim == 1 else data
        
        threshold = np.mean(signal_for_threshold) + self.params.threshold_multiplier * np.std(signal_for_threshold)
        above_threshold = signal_for_threshold > threshold
        
        # Find event boundaries
        event_segments = []
        in_event = False
        event_start = 0
        margin_samples = int(self.params.hmm_margin * self.fs)
        
        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_event:
                event_start = i
                in_event = True
            elif not above_threshold[i] and in_event:
                event_end = i
                in_event = False
                # Store event segment with margin
                seg_start = max(0, event_start - margin_samples)
                seg_end = min(L, event_end + margin_samples)
                if seg_end - seg_start > 10:  # Minimum segment length
                    event_segments.append((seg_start, seg_end))
        
        # Step 2: Sample events based on global_hmm_fraction
        segments = []
        if len(event_segments) > 0:
            n_events_to_use = max(1, int(len(event_segments) * fraction))
            n_events_to_use = min(n_events_to_use, len(event_segments))
            
            # Randomly select events
            selected_indices = rng.choice(len(event_segments), size=n_events_to_use, replace=False)
            
            for idx in selected_indices:
                seg_start, seg_end = event_segments[idx]
                if is_joint:
                    segments.append(data[seg_start:seg_end, :])
                else:
                    segments.append(data[seg_start:seg_end])
            
            # Step 3: Add baseline segments for contrast (same number as events)
            baseline_length = int(0.5 * self.fs)  # 0.5 second baseline segments
            n_baseline = min(n_events_to_use, int(len(event_segments) * 0.5))  # Half as many baselines
            
            for _ in range(n_baseline):
                attempts = 0
                while attempts < 100:
                    start_idx = rng.randint(0, max(1, L - baseline_length))
                    # Check if overlaps with any event
                    overlaps = any(start_idx < seg_end and start_idx + baseline_length > seg_start 
                                 for seg_start, seg_end in event_segments)
                    if not overlaps:
                        if is_joint:
                            segments.append(data[start_idx:start_idx + baseline_length, :])
                        else:
                            segments.append(data[start_idx:start_idx + baseline_length])
                        break
                    attempts += 1
            
            print(f"[Global HMM Training] Sampled {n_events_to_use} events + {n_baseline} baseline segments from {len(event_segments)} total events")
        else:
            # Fallback: use random continuous segments if no events found
            print(f"[Global HMM Training] Warning: No events found. Using random continuous segments.")
            sample_len = max(100, int(0.01 * L))
            n_samples = max(1, int((fraction * L) / sample_len))
            for _ in range(n_samples):
                start = rng.randint(0, max(1, L - sample_len))
                if is_joint:
                    segments.append(data[start:start + sample_len, :])
                else:
                    segments.append(data[start:start + sample_len])

        if not segments:
            raise RuntimeError("Failed to sample segments for global HMM training")

        if is_joint:
            training = np.vstack(segments)
        else:
            training_1d = np.concatenate(segments)
            # Apply feature engineering for MUA if enabled
            if sig == 'mua':
                training = self._prepare_hmm_features(training_1d, signal_type='mua')
            else:
                training = training_1d.reshape(-1, 1)

        # Use robust initialization if enabled
        if getattr(self.params, 'use_robust_init', False) and not is_joint:
            try:
                gm_means, gm_covs = self._initialize_hmm_robust(training[:, 0], n_components=n_states)
                # Expand for multi-dimensional features
                n_features = training.shape[1]
                if gm_means.shape[1] == 1 and n_features > 1:
                    gm_means = np.tile(gm_means, (1, n_features))
                    gm_covs = np.tile(gm_covs, (1, n_features))
                
                # Create HMM with init_params="" to preserve our robust initialization
                model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, 
                                   random_state=random_state, init_params='st')  # Only init start/transition
                model.means_ = gm_means
                model.covars_ = gm_covs
                model.fit(training)
            except Exception:
                # Fallback to standard fit
                model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=random_state)
                model.fit(training)
        else:
            model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=random_state)
            model.fit(training)
        
        self.global_hmms[sig] = model
        print(f"Trained global HMM for '{sig}' with {n_states} states on {len(training)} samples " +
              f"(features: {training.shape[1]})")
        return model

    # (Removed outdated detect_events placeholder and _detect_events_impl)
    def _process_signal(self, signal_data, channel_id):
        """Process a single channel or averaged signal for event detection."""
        # Initial signal processing
        if self.params.notch_freq:
            notched = self._notch_filter(signal_data, self.params.notch_freq)
        else:
            notched = signal_data

        # === Ripple processing with Karlsson-style preprocessing ===
        ripple_filtered = self._bandpass_filter(notched, *self.params.ripple_band)
        ripple_envelope = np.abs(hilbert(ripple_filtered))
        ripple_power = ripple_envelope ** 2
        
        # Apply smoothing if enabled (Karlsson method)
        if getattr(self.params, 'use_smoothing', False):
            ripple_power_smoothed = self._gaussian_smooth(ripple_power, self.params.smoothing_sigma)
        else:
            ripple_power_smoothed = ripple_power
        
        # ALWAYS store raw statistics BEFORE normalization
        # This ensures thresholds are computed from raw power envelope
        mean_power_raw = np.mean(ripple_power_smoothed)
        std_power_raw = np.std(ripple_power_smoothed)
        
        # Normalize ripple power if enabled (Karlsson method - OPTIONAL)
        if getattr(self.params, 'normalization_method', None):
            # Use global ripple power for consistent z-score normalization
            ripple_power_normalized, mean_val, std_dev = self._normalize_signal(
                self.ripple_power,
                method=self.params.normalization_method,
                time_range=getattr(self.params, 'normalization_time_range', None)
            )
            # Use normalized signal for detection
            smooth_ripple_power = ripple_power_normalized
            # CRITICAL: When using normalized signal, threshold must also be in normalized space
            # In normalized space (z-score or robust), threshold is simply the multiplier
            ripple_threshold = self.params.threshold_multiplier
        else:
            # No normalization - use smoothed signal directly
            smooth_ripple_power = ripple_power_smoothed
            # Threshold is computed from raw statistics
            ripple_threshold = mean_power_raw + self.params.threshold_multiplier * std_power_raw
        
        above_ripple = smooth_ripple_power > ripple_threshold

        # MUA processing (if enabled and provided)
        if self.params.enable_mua and self.mua_data is not None:
            # MUA data is already pre-processed (smoothed), use as-is
            smooth_mua = self.mua_data
            mua_threshold = (np.mean(smooth_mua) +
                             self.params.mua_threshold_multiplier * np.std(smooth_mua))
            above_mua = smooth_mua > mua_threshold
        else:
            smooth_mua = None
            mua_threshold = None
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

        # ===================================================================
        # STEP 1: ALWAYS run basic threshold detection first (independent)
        # ===================================================================
        # Basic detection detects ripple and MUA events SEPARATELY, then merges them
        # This prevents MUA from masking ripple events (the OR masking problem)
        
        # SUBSTEP 1A: Detect ripple-based events
        ripple_markers = above_ripple
        ripple_crossings = np.diff(ripple_markers.astype(int))
        ripple_starts = np.where(ripple_crossings == 1)[0]
        ripple_ends = np.where(ripple_crossings == -1)[0]
        
        # Handle edge cases for ripple
        if len(ripple_starts) > 0 and len(ripple_ends) > 0:
            if ripple_ends[0] < ripple_starts[0]:
                ripple_ends = ripple_ends[1:]
            if len(ripple_starts) > len(ripple_ends):
                ripple_starts = ripple_starts[:-1]
        
        # SUBSTEP 1B: Detect MUA-based events (if enabled)
        mua_starts = np.array([])
        mua_ends = np.array([])
        if self.params.enable_mua and self.mua_data is not None and above_mua is not None:
            mua_markers = above_mua
            mua_crossings = np.diff(mua_markers.astype(int))
            mua_starts = np.where(mua_crossings == 1)[0]
            mua_ends = np.where(mua_crossings == -1)[0]
            
            # Handle edge cases for MUA
            if len(mua_starts) > 0 and len(mua_ends) > 0:
                if mua_ends[0] < mua_starts[0]:
                    mua_ends = mua_ends[1:]
                if len(mua_starts) > len(mua_ends):
                    mua_starts = mua_starts[:-1]
        
        # SUBSTEP 1C: Merge and classify events
        basic_starts = []
        basic_ends = []
        event_types = []  # Track what triggered each event: 'ripple', 'mua', or 'both'
        
        # Add all ripple events
        for r_start, r_end in zip(ripple_starts, ripple_ends):
            duration = (r_end - r_start) / self.fs
            if duration >= self.params.min_duration and duration <= self.params.max_duration:
                # Check if there's overlapping MUA
                has_mua = False
                merged_start = r_start
                merged_end = r_end
                
                if len(mua_starts) > 0:
                    # Check if any MUA event overlaps with this ripple
                    for m_start, m_end in zip(mua_starts, mua_ends):
                        if not (r_end < m_start or r_start > m_end):  # Events overlap
                            has_mua = True
                            # Expand to include both signals
                            merged_start = min(merged_start, m_start)
                            merged_end = max(merged_end, m_end)
                
                basic_starts.append(merged_start)
                basic_ends.append(merged_end)
                event_types.append('both' if has_mua else 'ripple')
        
        # Add MUA-only events (those that don't overlap with ripple)
        if len(mua_starts) > 0:
            for m_start, m_end in zip(mua_starts, mua_ends):
                duration = (m_end - m_start) / self.fs
                if duration >= self.params.mua_min_duration and duration <= self.params.max_duration:
                    # Check if this MUA overlaps with any ripple event we already added
                    overlaps_ripple = False
                    for i, (b_start, b_end) in enumerate(zip(basic_starts, basic_ends)):
                        if not (m_end < b_start or m_start > b_end):
                            overlaps_ripple = True
                            break
                    
                    # Only add if it doesn't overlap (MUA-only event)
                    if not overlaps_ripple:
                        basic_starts.append(m_start)
                        basic_ends.append(m_end)
                        event_types.append('mua')
        
        # Convert to numpy arrays
        basic_starts = np.array(basic_starts) if basic_starts else np.array([])
        basic_ends = np.array(basic_ends) if basic_ends else np.array([])
        
        # Sort events by start time
        if len(basic_starts) > 0:
            sort_idx = np.argsort(basic_starts)
            basic_starts = basic_starts[sort_idx]
            basic_ends = basic_ends[sort_idx]
            event_types = [event_types[i] for i in sort_idx]
        
        # Store for iteration
        starts = basic_starts
        ends = basic_ends

        for event_idx, (start, end) in enumerate(zip(starts, ends)):
            duration = (end - start) / self.fs
            # Duration already validated during merge, but use type-specific minimum
            # Use MUA minimum duration for MUA-only candidates, otherwise use ripple min_duration
            detection_type = event_types[event_idx] if event_idx < len(event_types) else 'ripple'
            if detection_type == 'mua':
                min_dur = getattr(self.params, 'mua_min_duration', 0.0)
            else:
                min_dur = getattr(self.params, 'min_duration', 0.0)
            if duration < min_dur or duration > self.params.max_duration:
                # skip events that don't meet duration criteria for their type
                continue
            
            # Determine peak indices and validate event type
            # NOTE: Always check against ripple_threshold regardless of detection method
            ripple_event = False
            mua_event = False
            ripple_idx = None
            mua_idx = None
            ripple_peak = None
            mua_peak = None

            # Check if there's a ripple peak in this window
            window_power = smooth_ripple_power[start:end]
            if len(window_power) > 0 and np.max(window_power) > ripple_threshold:
                ripple_idx = start + np.argmax(window_power)
                ripple_peak = smooth_ripple_power[ripple_idx]
                ripple_event = True

            # Check for MUA if enabled
            if self.params.enable_mua and self.mua_data is not None and mua_threshold is not None:
                window_mua = smooth_mua[start:end]
                if len(window_mua) > 0 and np.max(window_mua) > mua_threshold:
                    mua_idx = start + np.argmax(window_mua)
                    mua_peak = smooth_mua[mua_idx]
                    mua_event = True

            # Determine final event type based on what peaks are present
            # if ripple_event and mua_event:
            #     event_type = 'ripple_mua'
            #     peak_idx = ripple_idx
            #     peak_power = ripple_peak
            # elif ripple_event:
            #     event_type = 'ripple_only'
            #     peak_idx = ripple_idx
            #     peak_power = ripple_peak
            # elif mua_event:
            #     event_type = 'mua_only'
            #     peak_idx = mua_idx
            #     peak_power = mua_peak
            # else:
            if ripple_event and mua_event:
                event_type = 'ripple_mua'
                peak_idx = ripple_idx
                # Always use normalized ripple power for peak_power
                peak_power = smooth_ripple_power[peak_idx]
            elif ripple_event:
                event_type = 'ripple_only'
                peak_idx = ripple_idx
                peak_power = smooth_ripple_power[peak_idx]
            elif mua_event:
                event_type = 'mua_only'
                peak_idx = mua_idx
                # If you want to use normalized MUA, normalize it here
                if getattr(self.params, 'normalization_method', None):
                    mua_mean = np.mean(smooth_mua)
                    mua_std = np.std(smooth_mua)
                    peak_power = (smooth_mua[peak_idx] - mua_mean) / mua_std
                else:
                    peak_power = smooth_mua[peak_idx]
                # proceed with mua_only events (do not skip)

            peak_time = peak_idx / self.fs if peak_idx is not None else None

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

            # === REMOVED: Early HMM refinement ===
            # The proper HMM edge detection happens later in the dual-boundary detection
            # Early refinement with small margins was causing overly narrow detections
            # The dual boundary detection below uses proper margins and validation
            
            # === NEW: Dual Boundary Detection (Ripple + MUA) ===
            ripple_start_idx, ripple_end_idx = None, None
            ripple_duration = None
            mua_start_idx, mua_end_idx = None, None
            mua_duration = None


            # Compute ripple-specific boundaries
            if ripple_event:
                ripple_start_idx, ripple_end_idx = self._find_ripple_boundaries(
                    smooth_ripple_power, start, end, ripple_threshold
                )
                if self.params.use_hmm_edge_detection and ripple_start_idx is not None and ripple_end_idx is not None:
                    hmm_win_start = max(0, ripple_start_idx - int(self.params.hmm_margin * self.fs))
                    hmm_win_end = min(len(smooth_ripple_power), ripple_end_idx + int(self.params.hmm_margin * self.fs))
                    ripple_start_idx_hmm, ripple_end_idx_hmm = self._refine_event_edges_hmm(
                        smooth_ripple_power, hmm_win_start, hmm_win_end
                    )
                    if ripple_end_idx_hmm > ripple_start_idx_hmm:
                        ripple_start_idx = ripple_start_idx_hmm
                        ripple_end_idx = ripple_end_idx_hmm
                if ripple_start_idx is not None and ripple_end_idx is not None:
                    ripple_duration = (ripple_end_idx - ripple_start_idx) / self.fs

            # Compute MUA-specific boundaries
            # Case A: Ripple present -> use ripple-centered window detection (existing behavior)
            if smooth_mua is not None and ripple_start_idx is not None and ripple_end_idx is not None:
                margin_samples = int(self.params.hmm_margin * self.fs)
                
                # Center search window on ripple event with margin for MUA expansion
                ripple_center = (ripple_start_idx + ripple_end_idx) // 2
                ripple_length = ripple_end_idx - ripple_start_idx
                
                # Create search window: ripple length + margins on both sides
                search_half_width = (ripple_length // 2) + margin_samples
                mua_search_start = max(0, ripple_center - search_half_width)
                mua_search_end = min(len(smooth_mua), ripple_center + search_half_width)
                
                # Extract MUA segment for analysis
                mua_segment = smooth_mua[mua_search_start:mua_search_end]
                
                # Step 1: Let HMM detect MUA boundaries independently (no constraints yet)
                if self.params.use_hmm_edge_detection and len(mua_segment) >= 10:
                    # Apply HMM to detect elevated MUA state boundaries
                    mua_start_rel, mua_end_rel = self._refine_event_edges_hmm(
                        mua_segment, 0, len(mua_segment), signal_type='mua'
                    )
                    
                    # Convert relative indices to absolute
                    if mua_start_rel is not None and mua_end_rel is not None:
                        mua_start_idx_hmm = mua_search_start + mua_start_rel
                        mua_end_idx_hmm = mua_search_start + mua_end_rel
                    else:
                        mua_start_idx_hmm = None
                        mua_end_idx_hmm = None
                else:
                    # Fallback: use threshold-based detection
                    mua_start_idx_hmm, mua_end_idx_hmm = self._find_mua_boundaries(
                        smooth_mua, mua_search_start, mua_search_end, mua_threshold
                    )
                
                # Step 2: Post-HMM validation - ensure MUA encompasses ripple
                if mua_start_idx_hmm is not None and mua_end_idx_hmm is not None:
                    # Validation: MUA should at minimum cover ripple window
                    # Allow MUA to expand beyond ripple, but ensure ripple is included
                    mua_start_idx = min(mua_start_idx_hmm, ripple_start_idx)
                    mua_end_idx = max(mua_end_idx_hmm, ripple_end_idx)
                    
                    # STEP 3: HMM validation and quality check
                    if mua_end_idx <= mua_start_idx:
                        # Invalid detection, use ripple boundaries
                        mua_start_idx = ripple_start_idx
                        mua_end_idx = ripple_end_idx

                    # VALIDATION: Check if HMM result is reasonable for MUA characteristics
                    mua_duration_hmm = (mua_end_idx - mua_start_idx) / self.fs
                    mua_duration_threshold = mua_duration_hmm  # We might need to recalculate threshold on the segmented data

                    # If HMM detects very short MUA duration relative to ripple, it probably failed
                    if ripple_duration is not None and mua_duration_hmm < ripple_duration * 0.8:
                        print(f"Event {self.event_counter + 1}: MUA HMM detection too short "
                             f"({mua_duration_hmm*1000:.1f}ms < {ripple_duration*1000:.1f}ms), "
                             f"falling back to threshold-based detection")

                        # FALLBACK: Use threshold crossing to expand MUA window
                        mua_start_idx, mua_end_idx = self._find_mua_boundary_expansion(
                            smooth_mua, ripple_start_idx, ripple_end_idx,
                            mua_threshold, expansion_factor=2.0
                        )
                else:
                    # HMM failed to detect, use threshold-based expansion
                    mua_start_idx, mua_end_idx = self._find_mua_boundary_expansion(
                        smooth_mua, ripple_start_idx, ripple_end_idx,
                        mua_threshold, expansion_factor=1.5
                    )
                
                # Step 3: Calculate MUA duration
                if mua_start_idx is not None and mua_end_idx is not None:
                    mua_duration = (mua_end_idx - mua_start_idx) / self.fs
                    
                    # Step 4: Final sanity check - MUA duration should be >= ripple duration
                    if ripple_duration is not None and mua_duration < ripple_duration:
                        print(f"WARNING: Event {self.event_counter + 1}: MUA duration "
                              f"({mua_duration*1000:.1f}ms) < Ripple duration "
                              f"({ripple_duration*1000:.1f}ms). Using ripple boundaries.")
                        mua_start_idx = ripple_start_idx
                        mua_end_idx = ripple_end_idx
                        mua_duration = ripple_duration
                else:
                    # Fallback to ripple boundaries
                    mua_start_idx = ripple_start_idx
                    mua_end_idx = ripple_end_idx
                    mua_duration = ripple_duration if ripple_duration is not None else (ripple_end_idx - ripple_start_idx) / self.fs
            # Case B: No ripple edges (mua_only). Run HMM seeding from MUA around the peak.
            elif smooth_mua is not None and event_type == 'mua_only':
                margin_samples = int(self.params.hmm_margin * self.fs)
                # Build a seed window from contiguous supra-threshold region around the MUA peak
                n = len(smooth_mua)
                # Safety for missing peak
                pk = int(peak_idx) if peak_idx is not None else (start + end) // 2
                pk = max(0, min(n - 1, pk))

                # Local threshold based on global MUA stats (already computed earlier)
                mua_threshold_local = (np.mean(smooth_mua) + self.params.mua_threshold_multiplier * np.std(smooth_mua))

                left = pk
                while left > 0 and smooth_mua[left] > mua_threshold_local:
                    left -= 1
                right = pk
                while right < n - 1 and smooth_mua[right] > mua_threshold_local:
                    right += 1

                seed_s, seed_e = (left, right) if right > left else (max(0, pk - 1), min(n, pk + 1))

                ms, me = self._compute_mua_edges(
                    smooth_mua=smooth_mua,
                    seed_start=seed_s,
                    seed_end=seed_e,
                    peak_idx=pk,
                    margin_samples=margin_samples,
                )

                if ms is None or me is None:
                    # Fallback: Minimal window around peak
                    half = max(margin_samples, int(0.1 * self.fs))
                    ms = max(0, pk - half)
                    me = min(n, pk + half)

                if me > ms:
                    mua_start_idx, mua_end_idx = ms, me
                    mua_duration = (mua_end_idx - mua_start_idx) / self.fs

            # Compute combined boundaries (union of ripple and MUA)
            if ripple_start_idx is not None and mua_start_idx is not None:
                combined_start_idx = min(ripple_start_idx, mua_start_idx)
                combined_end_idx = max(ripple_end_idx, mua_end_idx)
            elif ripple_start_idx is not None:
                combined_start_idx = ripple_start_idx
                combined_end_idx = ripple_end_idx
            elif mua_start_idx is not None:
                combined_start_idx = mua_start_idx
                combined_end_idx = mua_end_idx
            else:
                combined_start_idx = start
                combined_end_idx = end

            if combined_start_idx is not None and combined_end_idx is not None:
                combined_duration = (combined_end_idx - combined_start_idx) / self.fs
            else:
                combined_duration = None
            # ====================================================

            # Extract traces based on a trace window
            win_samples = int(self.params.trace_window * self.fs)
            half_win = win_samples // 2
            if peak_idx is not None:
                trace_start = max(0, peak_idx - half_win)
                trace_end = min(len(notched), peak_idx + half_win)
            else:
                trace_start = max(0, start - half_win)
                trace_end = min(len(notched), end + half_win)

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

            # === NEW: Compute all 4 duration estimates using multi-method approach ===
            # This gives us: basic threshold, hysteresis, HMM ripple, HMM MUA
            multi_durations = self._compute_multi_method_durations(
                start, end, smooth_ripple_power, smooth_mua, 
                ripple_threshold, mua_threshold
            )
            
            # --- NEW: compute low-threshold (hysteresis) value for storage ---
            # if getattr(self.params, 'use_hysteresis', False):
            #     ripple_low_th = mean_power_raw + getattr(self.params, 'hysteresis_low_multiplier', 2.0) * std_power_raw
            # else:
            #     ripple_low_th = None
            if getattr(self.params, 'use_hysteresis', False):
                if getattr(self.params, 'normalization_method', None):
                    # Use the normalized z-score multiplier
                    ripple_low_th = getattr(self.params, 'hysteresis_low_multiplier', 2.0)
                else:
                    # Use the raw, absolute value
                    ripple_low_th = mean_power_raw + getattr(self.params, 'hysteresis_low_multiplier', 2.0) * std_power_raw
            else:
                ripple_low_th = None

            # --- NEW: compute dominant ripple frequency & power for the event ---
            try:
                from scipy.signal import welch
                # prefer HMM ripple bounds, then basic bounds, then trace window
                psd_seg = None
                if multi_durations.get('hmm_ripple_start_idx') is not None and multi_durations.get('hmm_ripple_end_idx') is not None:
                    s_idx = int(multi_durations['hmm_ripple_start_idx'])
                    e_idx = int(multi_durations['hmm_ripple_end_idx'])
                    if e_idx > s_idx:
                        psd_seg = ripple_filtered[s_idx:e_idx]
                if psd_seg is None:
                    # fallback to trace window around peak
                    try:
                        seg_s = trace_start
                        seg_e = trace_end
                        psd_seg = ripple_filtered[seg_s:seg_e]
                    except Exception:
                        psd_seg = ripple_filtered[start:end] if end > start else ripple_filtered

                if len(psd_seg) >= 8:
                    nperseg = min(256, max(8, len(psd_seg)))
                    f, Pxx = welch(psd_seg, fs=self.fs, nperseg=nperseg)
                    idx = np.argmax(Pxx)
                    ripple_peak_freq = float(f[idx])
                    ripple_peak_freq_power = float(Pxx[idx])
                else:
                    ripple_peak_freq = None
                    ripple_peak_freq_power = None
            except Exception:
                ripple_peak_freq = None
                ripple_peak_freq_power = None

            # Simple frequency classification (relative to band midpoint)
            try:
                band_mid = 0.5 * (self.params.ripple_band[0] + self.params.ripple_band[1])
                if ripple_peak_freq is not None:
                    ripple_freq_class = 'low' if ripple_peak_freq < band_mid else 'high'
                else:
                    ripple_freq_class = None
            except Exception:
                ripple_freq_class = None
            
            self.event_counter += 1
            new_event = {
                'event_id': self.event_counter,
                'channel': channel_id,

                # Original unified boundaries (for backward compatibility)
                'start_idx': start,
                'end_idx': end,
                'start_time': start / self.fs,
                'end_time': end / self.fs,
                'duration': duration,

                # === Multi-method duration estimates (NEW) ===
                # Basic threshold (original detection method)
                'basic_start_idx': multi_durations['basic_start_idx'],
                'basic_end_idx': multi_durations['basic_end_idx'],
                'basic_duration': multi_durations['basic_duration'],
                'basic_start_time': multi_durations['basic_start_idx'] / self.fs,  # FIXED
                'basic_end_time': multi_durations['basic_end_idx'] / self.fs,      # FIXED
                
                # Hysteresis threshold (dual-threshold refinement)
                'hysteresis_start_idx': multi_durations['hysteresis_start_idx'],
                'hysteresis_end_idx': multi_durations['hysteresis_end_idx'],
                'hysteresis_duration': multi_durations['hysteresis_duration'],
                'hysteresis_start_time': multi_durations['hysteresis_start_idx'] / self.fs if multi_durations['hysteresis_start_idx'] is not None else None,
                'hysteresis_end_time': multi_durations['hysteresis_end_idx'] / self.fs if multi_durations['hysteresis_end_idx'] is not None else None,
                
                # HMM Ripple power (state-based ripple detection)
                'hmm_ripple_start_idx': multi_durations['hmm_ripple_start_idx'],
                'hmm_ripple_end_idx': multi_durations['hmm_ripple_end_idx'],
                'hmm_ripple_duration': multi_durations['hmm_ripple_duration'],
                
                # HMM MUA (state-based MUA detection)
                'hmm_mua_start_idx': multi_durations['hmm_mua_start_idx'],
                'hmm_mua_end_idx': multi_durations['hmm_mua_end_idx'],
                'hmm_mua_duration': multi_durations['hmm_mua_duration'],
                'mua_high_start': multi_durations['hmm_mua_start_idx'] / self.fs if multi_durations['hmm_mua_start_idx'] is not None else None,
                'mua_high_end': multi_durations['hmm_mua_end_idx'] / self.fs if multi_durations['hmm_mua_end_idx'] is not None else None,              
                # Legacy ripple-specific boundaries (kept for backward compatibility)
                'ripple_start_idx': ripple_start_idx,
                'ripple_end_idx': ripple_end_idx,
                'ripple_start_time': ripple_start_idx / self.fs if ripple_start_idx is not None else None,
                'ripple_end_time': ripple_end_idx / self.fs if ripple_end_idx is not None else None,
                'ripple_duration': ripple_duration,

                # Legacy MUA-specific boundaries (kept for backward compatibility)
                'mua_start_idx': mua_start_idx,
                'mua_end_idx': mua_end_idx,
                'mua_start_time': mua_start_idx / self.fs if mua_start_idx is not None else None,
                'mua_end_time': mua_end_idx / self.fs if mua_end_idx is not None else None,
                'mua_duration': mua_duration,

                # Legacy combined boundaries (kept for backward compatibility)
                'combined_start_idx': combined_start_idx,
                'combined_end_idx': combined_end_idx,
                'combined_start_time': combined_start_idx / self.fs if combined_start_idx is not None else None,
                'combined_end_time': combined_end_idx / self.fs if combined_end_idx is not None else None,
                'combined_duration': combined_duration,

                'peak_time': peak_time,
                'peak_power': peak_power,
                'peak_times': peak_times,
                'event_type': event_type,  # 'ripple_only', 'mua_only', or 'ripple_mua' (based on peaks)
                'detection_type': detection_type,  # 'ripple', 'mua', or 'both' (based on initial detection)
                # --- NEW: thresholds and spectral summaries ---
                'ripple_high_threshold': float(ripple_threshold) if ripple_threshold is not None else None,
                'ripple_low_threshold': float(ripple_low_th) if ripple_low_th is not None else None,
                'mua_threshold': float(mua_threshold) if mua_threshold is not None else None,
                'ripple_peak_freq': ripple_peak_freq,
                'ripple_peak_freq': ripple_peak_freq,
                'ripple_peak_freq_power': ripple_peak_freq_power,
                'ripple_peak_amplitude': float(ripple_peak) if 'ripple_peak' in locals() and ripple_peak is not None else None,
                'mua_peak_amplitude': float(mua_peak) if 'mua_peak' in locals() and mua_peak is not None else None,
                'mua_peak_idx': int(mua_idx) if 'mua_idx' in locals() and mua_idx is not None else None,
                'mua_peak_time': (int(mua_idx) / self.fs) if 'mua_idx' in locals() and mua_idx is not None else None,
                'ripple_freq_class': ripple_freq_class,
                'raw_trace': notched[trace_start:trace_end],
                'ripple_trace': ripple_filtered[trace_start:trace_end],
                'mua_trace': smooth_mua[trace_start:trace_end] if smooth_mua is not None else None,
                'ripple_power': smooth_ripple_power[trace_start:trace_end],
                'sharpwave_trace': sharpwave_filtered[trace_start:trace_end] if sharpwave_filtered is not None else None,
                'trace_timestamps': np.linspace(
                    peak_time - (peak_idx - trace_start) / self.fs if peak_time is not None and peak_idx is not None else 0,
                    peak_time + (trace_end - peak_idx) / self.fs if peak_time is not None and peak_idx is not None else (trace_end - trace_start) / self.fs,
                    trace_end - trace_start
                )
            }
            # --- Store canonical thresholds (raw and z) into the event dict ---
            try:
                thresholds_dict = self._store_event_thresholds(
                    mean_power_raw, std_power_raw, ripple_threshold, ripple_low_th, smooth_mua, mua_threshold
                )
            except Exception:
                thresholds_dict = {}
            new_event.update(thresholds_dict)
            # Add optional multivariate HMM joint bounds if enabled and MUA present
            if getattr(self.params, 'use_multivariate_hmm', False) and smooth_mua is not None:
                margin_samples = int(self.params.hmm_margin * self.fs)
                # Define search window around union of ripple and MUA bounds if available
                base_start = start
                base_end = end
                if ripple_start_idx is not None and ripple_end_idx is not None:
                    base_start = min(base_start, ripple_start_idx)
                    base_end = max(base_end, ripple_end_idx)
                if mua_start_idx is not None and mua_end_idx is not None:
                    base_start = min(base_start, mua_start_idx)
                    base_end = max(base_end, mua_end_idx)
                win_start = max(0, base_start - margin_samples)
                win_end = min(len(smooth_ripple_power), base_end + margin_samples)

                try:
                    j_s, j_e = self._refine_event_edges_hmm_multivariate(
                        smooth_ripple_power, smooth_mua, win_start, win_end
                    )
                except Exception as e:
                    j_s, j_e = None, None

                if j_s is not None and j_e is not None and j_e > j_s:
                    new_event['joint_start_idx'] = j_s
                    new_event['joint_end_idx'] = j_e
                    new_event['joint_start_time'] = j_s / self.fs
                    new_event['joint_end_time'] = j_e / self.fs
                    new_event['joint_duration'] = (j_e - j_s) / self.fs
                else:
                    new_event['joint_start_idx'] = None
                    new_event['joint_end_idx'] = None
                    new_event['joint_start_time'] = None
                    new_event['joint_end_time'] = None
                    new_event['joint_duration'] = None

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
                    if (prev_event['channel'] == event['channel']):
                        prev_end = prev_event.get('ripple_end_time', prev_event.get('end_time'))
                        curr_start = event.get('ripple_start_time') or event.get('start_time')
                        if prev_end is not None and curr_start is not None and (curr_start - prev_end) < self.params.single_separation:
                            is_single = False
            
                # Check next event if exists
                if event_idx < len(self.swr_events) - 1:
                    next_event = self.swr_events[event_idx + 1]
                    if (next_event['channel'] == event['channel']):
                        next_start = next_event.get('ripple_start_time') or next_event.get('start_time')
                        curr_end = event.get('ripple_end_time', event.get('end_time'))
                        if next_start is not None and curr_end is not None and (next_start - curr_end) < self.params.single_separation:
                            is_single = False
        
            group_type = 'single' if is_single else 'unclassified'
        elif group_size == 2:
            group_type = 'double'
        elif group_size == 3:
            group_type = 'triple'
        else:
            group_type = 'multiple'
    
        # Calculate inter-event intervals using ripple power boundaries only
        intervals = []
        for i in range(1, len(group)):
            prev_end = group[i-1].get('ripple_end_time') or group[i-1].get('end_time', 0.0)
            curr_start = group[i].get('ripple_start_time') or group[i].get('start_time', 0.0)
            interval = curr_start - prev_end
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
                'peak_count': len(event.get('peak_times', [1])),
                # Save both ripple and MUA boundaries for downstream analysis
                'ripple_start_idx': event.get('ripple_start_idx'),
                'ripple_end_idx': event.get('ripple_end_idx'),
                'ripple_start_time': event.get('ripple_start_time'),
                'ripple_end_time': event.get('ripple_end_time'),
                'mua_start_idx': event.get('mua_start_idx'),
                'mua_end_idx': event.get('mua_end_idx'),
                'mua_start_time': event.get('mua_start_time'),
                'mua_end_time': event.get('mua_end_time'),
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
            # Prefer ripple-based boundaries for classification
            curr_start = current_event.get('ripple_start_time') or current_event.get('start_time')
            prev_end = prev_event.get('ripple_end_time', prev_event.get('end_time'))
            interval = curr_start - prev_end if (curr_start is not None and prev_end is not None) else float('inf')
        
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
        Includes a 4th plot to compare consensus duration vs. HMM-detected MUA duration.

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
        
        # Determine which event list to use
        events_to_inspect = []
        if self.multi_region:
             for region_events in self.swr_events.values():
                 events_to_inspect.extend(region_events)
        else:
             events_to_inspect = self.swr_events

        for event in events_to_inspect:
            if 'classification' not in event:
                continue
                
            if (event['classification']['group_type'] == group_type and 
                event['classification']['group_id'] not in group_ids_seen):
                
                # Get all events in this group
                group_id = event['classification']['group_id']
                group_events = [e for e in events_to_inspect if 'classification' in e and e['classification']['group_id'] == group_id]
                examples.append(group_events)
                group_ids_seen.add(group_id)
                
            if len(examples) >= n_examples:
                break

        if not examples:
            print(f"No {group_type} events found")
            return

        # Create figure
        n_rows = len(examples)
        # --- MODIFIED: Widen the figure for 4 plots ---
        fig = plt.figure(figsize=(20, 4*n_rows)) 

        for i, group in enumerate(examples):
            # Sort events by time
            # Prefer ripple-based ordering for visualization
            group = sorted(group, key=lambda x: (x.get('ripple_start_time') if x.get('ripple_start_time') is not None else x.get('start_time', 0.0)))

            # Find time window
            start_time = group[0]['start_time'] - window_size/4
            end_time = group[-1]['end_time'] + window_size/4

            # Get data indices
            # Handle multi-region or single-region timepoints
            total_timepoints = self.n_timepoints[group[0]['region']] if self.multi_region else self.n_timepoints
            start_idx = max(0, int(start_time * self.fs))
            end_idx = min(total_timepoints, int(end_time * self.fs))
            
            t = np.arange(start_idx, end_idx) / self.fs

            # --- MODIFIED: Use 4 columns ---
            # --- Plot 1: Raw Signal ---
            ax1 = plt.subplot(n_rows, 4, i*4 + 1)
            
            # Determine which LFP data to pull from
            if self.multi_region:
                region = group[0]['region']
                lfp_source = self.lfp_data[region]
            else:
                lfp_source = self.lfp_data

            if isinstance(group[0]['channel'], int):
                signal = lfp_source[group[0]['channel'], start_idx:end_idx]
            else:
                signal = np.mean(lfp_source[:, start_idx:end_idx], axis=0)
            
            ax1.plot(t, signal, 'k')

            # Highlight events (using combined consensus duration)
            for event in group:
                ax1.axvspan(event['combined_start_time'], event['combined_end_time'], color='yellow', alpha=0.3)
                ax1.axvline(x=event['peak_time'], color='r', linestyle='--')
            
            ax1.set_title(f"Example {i+1}: Raw Signal (Ch {group[0]['channel']})")
            ax1.set_ylabel('Amplitude')

            # --- MODIFIED: Use 4 columns ---
            # --- Plot 2: Ripple Power ---
            ax2 = plt.subplot(n_rows, 4, i*4 + 2)
            for event in group:
                if event['ripple_trace'] is not None:
                    ax2.plot(event['trace_timestamps'], event['ripple_power'], 'r')
                    # Shade HMM ripple duration
                    if event['ripple_start_time'] is not None:
                         ax2.axvspan(event['ripple_start_time'], event['ripple_end_time'], color='cyan', alpha=0.4, label='HMM Ripple')
                    # Shade joint HMM duration (if available)
                    if event.get('joint_start_time') is not None and event.get('joint_end_time') is not None:
                        ax2.axvspan(event['joint_start_time'], event['joint_end_time'], color='orange', alpha=0.3, label='HMM Joint')
                    ax2.axvline(x=event['peak_time'], color='r', linestyle='--')
            ax2.set_title("Ripple Power (HMM Ripple shaded)")
            ax2.set_ylabel('Power')

            # --- MODIFIED: Use 4 columns ---
            # --- Plot 3: MUA with Consensus Edges ---
            ax3 = plt.subplot(n_rows, 4, i*4 + 3)
            for event in group:
                if event['mua_trace'] is not None:
                    ax3.plot(event['trace_timestamps'], event['mua_trace'], 'g')
                    # Shade combined (consensus) duration
                    ax3.axvspan(event['combined_start_time'], event['combined_end_time'], color='yellow', alpha=0.3, label='Consensus')
                    ax3.axvline(x=event['peak_time'], color='r', linestyle='--')
            ax3.set_title("MUA (Consensus Edges)")
            ax3.set_ylabel('Rate')
            if ax3.get_legend_handles_labels()[0]:
                ax3.legend()

            # --- NEW: Plot 4: MUA with HMM-Detected Edges ---
            ax4 = plt.subplot(n_rows, 4, i*4 + 4)
            for event in group:
                if event['mua_trace'] is not None:
                    # Plot the same MUA trace
                    ax4.plot(event['trace_timestamps'], event['mua_trace'], 'g')
                    
                    # --- KEY CHANGE: Plot HMM MUA boundaries ---
                    if event['mua_start_time'] is not None and event['mua_end_time'] is not None:
                        ax4.axvspan(event['mua_start_time'], event['mua_end_time'], 
                                    color='magenta', alpha=0.4, label='HMM MUA')
                    # Plot joint HMM span on MUA axis as well
                    if event.get('joint_start_time') is not None and event.get('joint_end_time') is not None:
                        ax4.axvspan(event['joint_start_time'], event['joint_end_time'], 
                                    color='orange', alpha=0.3, label='HMM Joint')
                    
                    ax4.axvline(x=event['peak_time'], color='r', linestyle='--')
            
            ax4.set_title("MUA (HMM Edges)")
            ax4.set_ylabel('Rate')
            # Add a legend if any HMM events were found
            handles, labels = ax4.get_legend_handles_labels()
            if handles:
                # Deduplicate labels
                uniq = dict(zip(labels, handles))
                ax4.legend(uniq.values(), uniq.keys())

            # Add event information
            durations = [f"{event['combined_duration']*1000:.1f}ms" for event in group]
            info_text = [
                f"Events: {len(group)}",
                f"Type: {group[0]['event_type']}",
                f"Duration: {', '.join(durations)}"
            ]
            if len(group) > 1:
                intervals = group[0]['classification']['inter_event_intervals']
                interval_text = [f"{interval*1000:.1f}ms" for interval in intervals]
                info_text.append(f"Intervals: {', '.join(interval_text)}")
            
            ax1.text(0.02, 0.98, '\n'.join(info_text), transform=ax1.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_basic_stats(self, stats):
        """
        Plot basic statistics from detected SWR events with multi-method comparison.
        
        Shows:
        - Event type distribution (pie chart)
        - Duration comparison boxplot (all 4 methods on same N events)
        - 4 histogram panels (one per detection method)
        
        Parameters
        ----------
        stats : dict
            Dictionary returned from analyze_basic_stats()
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not stats or stats['event_counts']['total'] == 0:
            print("No stats to plot. Run analyze_basic_stats() first.")
            return

        # Create figure with 2 rows
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
        
        # Color scheme for methods
        method_colors = {
            'basic': '#3498db',
            'hysteresis': '#e74c3c',
            'hmm_ripple': '#2ecc71',
            'hmm_mua': '#f39c12'
        }
        
        method_labels = {
            'basic': 'Basic\nThreshold',
            'hysteresis': 'Hysteresis',
            'hmm_ripple': 'HMM\nRipple',
            'hmm_mua': 'HMM\nMUA'
        }
        
        # === Row 1: Event Type Pie + Duration Boxplot ===
        
        # Plot 1: Event Type Distribution (Pie Chart)
        ax_pie = fig.add_subplot(gs[0, 0])
        labels = []
        sizes = []
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F39C12', '#9B59B6']
        
        for event_type, count in stats['event_counts']['by_type'].items():
            if count > 0:
                labels.append(event_type.replace('_', ' ').title())
                sizes.append(count)
        
        if sizes:
            wedges, texts, autotexts = ax_pie.pie(
                sizes, labels=labels, autopct='%1.1f%%', 
                startangle=90, colors=colors_pie[:len(sizes)]
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax_pie.set_title(
                f'Event Type Distribution\n(N={stats["event_counts"]["total"]} events)', 
                fontsize=12, fontweight='bold'
            )
        ax_pie.axis('equal')
        
        # Plot 2: Duration Comparison Boxplot (spans 2 columns)
        ax_box = fig.add_subplot(gs[0, 1:3])
        
        if 'durations_by_method' in stats:
            durations_data = []
            labels_list = []
            colors_list = []

            # Build list of methods to include in the boxplot depending on enabled features
            methods_for_box = ['basic']
            if stats.get('hysteresis_used', False):
                methods_for_box.append('hysteresis')
            if stats.get('hmm_used', False):
                methods_for_box += ['hmm_ripple', 'hmm_mua']

            for method in methods_for_box:
                durs = stats['durations_by_method'].get(method, np.array([]))
                if len(durs) > 0:
                    durations_data.append(durs * 1000)  # Convert to ms
                    labels_list.append(method_labels[method])
                    colors_list.append(method_colors[method])
            
            if durations_data:
                bp = ax_box.boxplot(
                    durations_data, 
                    labels=labels_list,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8, markeredgecolor='darkred'),
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5)
                )
                
                for patch, color in zip(bp['boxes'], colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax_box.set_ylabel('Duration (ms)', fontsize=12, fontweight='bold')
                ax_box.set_title(
                    f'Duration Comparison by Detection Method\n(N={len(durations_data[0])} events, After Cleanup)',
                    fontsize=12, fontweight='bold'
                )
                ax_box.grid(axis='y', alpha=0.3, linestyle='--')
                ax_box.spines['top'].set_visible(False)
                ax_box.spines['right'].set_visible(False)
                
                # Add mean values as text
                for i, (data, label) in enumerate(zip(durations_data, labels_list)):
                    mean_val = np.mean(data)
                    median_val = np.median(data)
                    ax_box.text(
                        i + 1, ax_box.get_ylim()[1] * 0.95,
                        f'μ={mean_val:.1f}\nM={median_val:.1f}',
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
        
        # Plot 3: Violin plot alternative (spans 2 columns)
        ax_violin = fig.add_subplot(gs[0, 3:5])
        
        if 'durations_by_method' in stats:
            durations_data = []
            labels_list = []
            colors_list = []

            # Build list of methods to include in the violin depending on enabled features
            methods_for_violin = ['basic']
            if stats.get('hysteresis_used', False):
                methods_for_violin.append('hysteresis')
            if stats.get('hmm_used', False):
                methods_for_violin += ['hmm_ripple', 'hmm_mua']

            for method in methods_for_violin:
                durs = stats['durations_by_method'].get(method, np.array([]))
                if len(durs) > 0:
                    durations_data.append(durs * 1000)
                    labels_list.append(method_labels[method])
                    colors_list.append(method_colors[method])
            
            if durations_data:
                positions = np.arange(len(labels_list))
                parts = ax_violin.violinplot(
                    durations_data,
                    positions=positions,
                    showmeans=True,
                    showmedians=True,
                    widths=0.7
                )
                
                for pc, color in zip(parts['bodies'], colors_list):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.5)
                
                # Style the other elements
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                    if partname in parts:
                        vp = parts[partname]
                        vp.set_edgecolor('black')
                        vp.set_linewidth(1.5)
                
                ax_violin.set_xticks(positions)
                ax_violin.set_xticklabels(labels_list)
                ax_violin.set_ylabel('Duration (ms)', fontsize=12, fontweight='bold')
                ax_violin.set_title(
                    f'Duration Distribution by Method\n(Violin Plot)',
                    fontsize=12, fontweight='bold'
                )
                ax_violin.grid(axis='y', alpha=0.3, linestyle='--')
                ax_violin.spines['top'].set_visible(False)
                ax_violin.spines['right'].set_visible(False)
        
        # === Row 2: 4 Histograms ===
        
        histogram_configs = [
            ('basic', 0, 'Basic Threshold Duration'),
            ('hysteresis', 1, 'Hysteresis Duration'),
            ('hmm_ripple', 2, 'HMM Ripple Power Duration'),
            ('hmm_mua', 3, 'HMM MUA Duration')
        ]
        
        if 'durations_by_method' in stats:
            for method, col_idx, title in histogram_configs:
                ax = fig.add_subplot(gs[1, col_idx])
                durs = stats['durations_by_method'].get(method, np.array([]))
                
                if len(durs) > 0:
                    durs_ms = durs * 1000
                    color = method_colors[method]
                    
                    # Plot histogram
                    n, bins, patches = ax.hist(
                        durs_ms, bins=40, alpha=0.7, color=color,
                        edgecolor='black', linewidth=1
                    )
                    
                    # Add mean and median lines
                    mean_val = np.mean(durs_ms)
                    median_val = np.median(durs_ms)
                    
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_val:.1f}ms')
                    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2,
                            label=f'Median: {median_val:.1f}ms')
                    
                    ax.set_xlabel('Duration (ms)', fontsize=10, fontweight='bold')
                    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
                    ax.set_title(f'{title}\n(N={len(durs)})', fontsize=11, fontweight='bold')
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                else:
                    # If a method was disabled, show a clear "analysis not done" message
                    if method in ('hmm_ripple', 'hmm_mua') and not stats.get('hmm_used', False):
                        ax.text(0.5, 0.5, 'Analysis Not Done\n(HMM disabled)', ha='center', va='center',
                                transform=ax.transAxes, fontsize=12, color='gray', weight='bold')
                    elif method == 'hysteresis' and not stats.get('hysteresis_used', False):
                        ax.text(0.5, 0.5, 'Analysis Not Done\n(Hysteresis disabled)', ha='center', va='center',
                                transform=ax.transAxes, fontsize=12, color='gray', weight='bold')
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes, fontsize=12)
                    ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Add overall title
        plt.suptitle(
            'SWR Detection Multi-Method Comparison (After Cleanup)',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*80)
        print("MULTI-METHOD DURATION COMPARISON SUMMARY")
        print("="*80)
        if 'durations_by_method' in stats:
            for method in ['basic', 'hysteresis', 'hmm_ripple', 'hmm_mua']:
                # If HMM disabled, explicitly state that instead of printing stats
                if method in ('hmm_ripple', 'hmm_mua') and not stats.get('hmm_used', False):
                    print(f"\n{method.upper().replace('_', ' ')}:")
                    print(f"  HMM was not used for this run.")
                    continue

                durs = stats['durations_by_method'].get(method, np.array([]))
                if len(durs) > 0:
                    print(f"\n{method.upper().replace('_', ' ')}:")
                    print(f"  N events: {len(durs)}")
                    print(f"  Mean: {np.mean(durs)*1000:.2f} ms")
                    print(f"  Median: {np.median(durs)*1000:.2f} ms")
                    print(f"  Std: {np.std(durs)*1000:.2f} ms")
                    print(f"  Range: [{np.min(durs)*1000:.2f}, {np.max(durs)*1000:.2f}] ms")
        print("="*80 + "\n")

    def analyze_basic_stats(self):
        """
        Analyzes basic statistics from detected SWR events and classification.
        
        NOTE: These statistics are calculated AFTER event cleanup/filtering:
        - Events that don't meet duration criteria are excluded
        - Events below quality thresholds are excluded
        - Merged/consolidated events are counted as single events
        
        This ensures accurate duration estimates reflecting only valid, high-quality events.
        
        Calculates stats for:
        - Consensus events (by classification)
        - HMM-edge detected events (ripple & MUA)
        - Hysteresis-detected events (if enabled)
        - Basic threshold events (MUA + Ripple)
        
        Prints a summary of the findings.
        """
        print("\n" + "="*80)
        print("ANALYZING BASIC STATISTICS (AFTER CLEANUP)")
        print("="*80)
        
        all_stats = {
            'event_counts': {
                'total': 0,
                'by_type': {'ripple_mua': 0, 'ripple_only': 0, 'mua_only': 0},
                'by_method': {
                    'basic_threshold': 0,
                    'hysteresis': 0,
                    'hmm_ripple': 0,
                    'hmm_mua': 0
                }
            },
            'durations': {
                'single': [],
                'double': [],
                'triple': [],
                'multiple': [],
                'unclassified': []
            },
            'hmm_durations': {
                'ripple': [],
                'mua': []
            },
            'method_durations': {
                'basic_threshold': [],
                'hysteresis': [],
                'hmm_ripple': [],
                'hmm_mua': []
            },
            # === NEW: Multi-method duration arrays (same N events, 4 estimates each) ===
            'durations_by_method': {
                'basic': [],
                'hysteresis': [],
                'hmm_ripple': [],
                'hmm_mua': []
            }
        }

        # Record whether HMM edge-detection was enabled for this detector
        all_stats['hmm_used'] = bool(getattr(self.params, 'use_hmm_edge_detection', False))
        # Record whether hysteresis was enabled for this detector
        all_stats['hysteresis_used'] = bool(getattr(self.params, 'use_hysteresis', False))

        # Determine which events to process (handles multi-region)
        events_to_process = []
        if self.multi_region:
             print("Multi-region detected. Aggregating events from all regions for stats.")
             for region_events in self.swr_events.values():
                 events_to_process.extend(region_events)
        else:
             events_to_process = self.swr_events

        if not events_to_process:
             print("No events found to analyze.")
             return all_stats

        all_stats['event_counts']['total'] = len(events_to_process)

        for event in events_to_process:
            # 1. Event Type Stats
            if event['event_type'] in all_stats['event_counts']['by_type']:
                all_stats['event_counts']['by_type'][event['event_type']] += 1
            
            # 2. Classification Duration Stats (using combined duration for backward compatibility)
            if 'classification' in event:
                group_type = event['classification'].get('group_type', 'unclassified')
                if group_type in all_stats['durations']:
                    all_stats['durations'][group_type].append(event['combined_duration'])
            else:
                 all_stats['durations']['unclassified'].append(event['combined_duration'])

            # === NEW: Extract all 4 duration estimates from the SAME event ===
            # This ensures fair comparison - same N events, different methods
            if 'basic_duration' in event and event['basic_duration'] is not None:
                all_stats['durations_by_method']['basic'].append(event['basic_duration'])
                all_stats['event_counts']['by_method']['basic_threshold'] += 1
                
            if 'hysteresis_duration' in event and event['hysteresis_duration'] is not None:
                # Only collect hysteresis durations if hysteresis was enabled
                if all_stats.get('hysteresis_used', False):
                    all_stats['durations_by_method']['hysteresis'].append(event['hysteresis_duration'])
                    all_stats['event_counts']['by_method']['hysteresis'] += 1
                
            # Only record HMM-derived durations if HMM edge detection was actually used
            if all_stats.get('hmm_used', False):
                if 'hmm_ripple_duration' in event and event['hmm_ripple_duration'] is not None:
                    all_stats['durations_by_method']['hmm_ripple'].append(event['hmm_ripple_duration'])
                    all_stats['event_counts']['by_method']['hmm_ripple'] += 1
                if 'hmm_mua_duration' in event and event['hmm_mua_duration'] is not None:
                    all_stats['durations_by_method']['hmm_mua'].append(event['hmm_mua_duration'])
                    all_stats['event_counts']['by_method']['hmm_mua'] += 1

            # 3. Legacy HMM Duration Stats (for backward compatibility)
            if event.get('ripple_duration') is not None and event['ripple_duration'] > 0:
                all_stats['hmm_durations']['ripple'].append(event['ripple_duration'])
                all_stats['method_durations']['hmm_ripple'].append(event['ripple_duration'])
                
            if event.get('mua_duration') is not None and event['mua_duration'] > 0:
                all_stats['hmm_durations']['mua'].append(event['mua_duration'])
                all_stats['method_durations']['hmm_mua'].append(event['mua_duration'])
        
        # Convert lists to numpy arrays for easier plotting
        for key, val in all_stats['durations'].items():
            all_stats['durations'][key] = np.array(val)
        for key, val in all_stats['hmm_durations'].items():
            all_stats['hmm_durations'][key] = np.array(val)
        for key, val in all_stats['method_durations'].items():
            all_stats['method_durations'][key] = np.array(val)
        # Convert multi-method durations to arrays
        for key, val in all_stats['durations_by_method'].items():
            all_stats['durations_by_method'][key] = np.array(val)

        # If HMM was disabled, ensure HMM-related arrays/counts are explicitly empty/zero
        if not all_stats.get('hmm_used', False):
            all_stats['durations_by_method']['hmm_ripple'] = np.array([])
            all_stats['durations_by_method']['hmm_mua'] = np.array([])
            all_stats['hmm_durations']['ripple'] = np.array([])
            all_stats['hmm_durations']['mua'] = np.array([])
            all_stats['method_durations']['hmm_ripple'] = np.array([])
            all_stats['method_durations']['hmm_mua'] = np.array([])
            all_stats['event_counts']['by_method']['hmm_ripple'] = 0
            all_stats['event_counts']['by_method']['hmm_mua'] = 0

        # If hysteresis was disabled, ensure hysteresis-related arrays/counts are explicitly empty/zero
        if not all_stats.get('hysteresis_used', False):
            all_stats['durations_by_method']['hysteresis'] = np.array([])
            all_stats['method_durations']['hysteresis'] = np.array([])
            all_stats['event_counts']['by_method']['hysteresis'] = 0

        # --- Print Summary ---
        print(f"\nTotal candidate events detected: {all_stats['event_counts']['total']}")
        print("\n" + "-"*80)
        print("EVENT COUNTS BY TYPE")
        print("-"*80)
        print(f"  Ripple & MUA:       {all_stats['event_counts']['by_type']['ripple_mua']}")
        print(f"  Ripple Only:        {all_stats['event_counts']['by_type']['ripple_only']}")
        print(f"  MUA Only:           {all_stats['event_counts']['by_type']['mua_only']}")

        # === NEW: Multi-Method Duration Comparison ===
        print("\n" + "-"*80)
        print(f"DURATION ESTIMATES BY METHOD (Same N={all_stats['event_counts']['total']} events)")
        print("-"*80)
        
        method_labels = {
            'basic': 'Basic Threshold',
            'hysteresis': 'Hysteresis (High=3.5SD, Low=2.0SD)',
            'hmm_ripple': 'HMM Edge (Ripple Power)',
            'hmm_mua': 'HMM Edge (MUA)'
        }
        
        for method_key, method_label in method_labels.items():
            # If hysteresis or HMM were disabled, report that instead of printing empty arrays
            if method_key == 'hysteresis' and not all_stats.get('hysteresis_used', False):
                print(f"  {method_label}: Hysteresis was not used.")
                continue
            if method_key in ('hmm_ripple', 'hmm_mua') and not all_stats.get('hmm_used', False):
                print(f"  {method_label}: HMM was not used.")
                continue

            durs = all_stats['durations_by_method'][method_key]
            if len(durs) > 0:
                print(f"  {method_label}:")
                print(f"      Durations (s): {durs}")
                print(f"      Mean: {np.mean(durs)*1000:.2f} ms, Median: {np.median(durs)*1000:.2f} ms")
                print(f"      N={len(durs)} events")
            else:
                print(f"  {method_label}: No events")

        # Classification Duration Stats (legacy)
        print("\n" + "-"*80)
        print("CONSENSUS EVENT DURATIONS (by Classification)")
        print("Note: After cleanup - only events meeting quality criteria")
        print("-"*80)
        for group_type, durations in all_stats['durations'].items():
            if len(durations) > 0:
                print(f"  {group_type.capitalize()} (N={len(durations)}):")
                print(f"      Mean: {np.mean(durations)*1000:.2f} ms, Median: {np.median(durations)*1000:.2f} ms")

        # HMM Duration Stats
        hmm_ripple_count = len(all_stats['hmm_durations']['ripple'])
        hmm_mua_count = len(all_stats['hmm_durations']['mua'])
        
        print("\n" + "-"*80)
        print("HMM-EDGE EVENT DURATIONS (Ripple Power)")
        print("Note: After cleanup - precise boundaries from HMM state detection")
        print("-"*80)
        if hmm_ripple_count > 0:
            durs = all_stats['hmm_durations']['ripple']
            print(f"  HMM Ripple (N={hmm_ripple_count}):")
            print(f"      Mean: {np.mean(durs)*1000:.2f} ms, Median: {np.median(durs)*1000:.2f} ms")
        else:
            print("  No HMM Ripple events detected")
            
        if hmm_mua_count > 0:
            durs = all_stats['hmm_durations']['mua']
            print(f"  HMM MUA (N={hmm_mua_count}):")
            print(f"      Mean: {np.mean(durs)*1000:.2f} ms, Median: {np.median(durs)*1000:.2f} ms")

        # Method-specific durations
        print("\n" + "-"*80)
        print("DURATIONS BY DETECTION METHOD")
        print("Note: After cleanup - comparing different detection algorithms")
        print("-"*80)
        
        method_labels = {
            'basic_threshold': 'Basic Threshold (MUA + Ripple)',
            'hysteresis': 'Hysteresis (Ripple Power)',
            'hmm_ripple': 'HMM Edge (Ripple Power)',
            'hmm_mua': 'HMM Edge (MUA)'
        }
        
        for method, label in method_labels.items():
            durs = all_stats['method_durations'][method]
            if len(durs) > 0:
                print(f"  {label} (N={len(durs)}):")
                print(f"      Mean: {np.mean(durs)*1000:.2f} ms, Median: {np.median(durs)*1000:.2f} ms")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")

        return all_stats

    def analyze_dual_boundaries(self):
        """
        Analyze differences between ripple and MUA boundaries.
        This method computes statistics about the temporal relationship between
        ripple oscillations and population firing dynamics.
        
        Returns
        -------
        dict
            Dictionary containing:
            - ripple_durations: Array of ripple-specific durations (seconds)
            - mua_durations: Array of MUA-specific durations (seconds)
            - combined_durations: Array of combined (union) durations (seconds)
            - temporal_offsets_start: Array of (MUA_start - Ripple_start) offsets (seconds)
            - temporal_offsets_end: Array of (MUA_end - Ripple_end) offsets (seconds)
            - mean_offset_start: Mean start time offset (positive = MUA starts earlier)
            - mean_offset_end: Mean end time offset (positive = MUA ends later)
            - events_with_both: Number of events with both ripple and MUA boundaries
        """
        if not self.swr_events:
            print("No events detected")
            return None
        
        ripple_durations = []
        mua_durations = []
        combined_durations = []
        temporal_offsets_start = []
        temporal_offsets_end = []
        
        for event in self.swr_events:
            # Collect ripple durations
            if event.get('ripple_duration') is not None:
                ripple_durations.append(event['ripple_duration'])
            
            # Collect MUA durations
            if event.get('mua_duration') is not None:
                mua_durations.append(event['mua_duration'])
            
            # Collect combined durations
            if event.get('combined_duration') is not None:
                combined_durations.append(event['combined_duration'])
            
            # Calculate temporal offsets (only for events with both boundaries)
            if (event.get('ripple_start_time') is not None and 
                event.get('mua_start_time') is not None):
                # Positive offset = MUA starts before ripple
                offset_start = event['mua_start_time'] - event['ripple_start_time']
                temporal_offsets_start.append(offset_start)
                
                # Positive offset = MUA ends after ripple
                offset_end = event['mua_end_time'] - event['ripple_end_time']
                temporal_offsets_end.append(offset_end)
        
        # Compile statistics
        stats = {
            'ripple_durations': np.array(ripple_durations),
            'mua_durations': np.array(mua_durations),
            'combined_durations': np.array(combined_durations),
            'temporal_offsets_start': np.array(temporal_offsets_start),
            'temporal_offsets_end': np.array(temporal_offsets_end),
            'events_with_both': len(temporal_offsets_start),
            'events_with_ripple': len(ripple_durations),
            'events_with_mua': len(mua_durations),
            'total_events': len(self.swr_events)
        }
        
        # Calculate summary statistics
        if temporal_offsets_start:
            stats['mean_offset_start'] = np.mean(temporal_offsets_start)
            stats['std_offset_start'] = np.std(temporal_offsets_start)
            stats['median_offset_start'] = np.median(temporal_offsets_start)
        else:
            stats['mean_offset_start'] = None
            stats['std_offset_start'] = None
            stats['median_offset_start'] = None
        
        if temporal_offsets_end:
            stats['mean_offset_end'] = np.mean(temporal_offsets_end)
            stats['std_offset_end'] = np.std(temporal_offsets_end)
            stats['median_offset_end'] = np.median(temporal_offsets_end)
        else:
            stats['mean_offset_end'] = None
            stats['std_offset_end'] = None
            stats['median_offset_end'] = None
        
        # Print summary
        print("\n" + "="*60)
        print("DUAL BOUNDARY ANALYSIS")
        print("="*60)
        
        print(f"\nTotal events: {stats['total_events']}")
        print(f"Events with ripple boundaries: {stats['events_with_ripple']}")
        print(f"Events with MUA boundaries: {stats['events_with_mua']}")
        print(f"Events with both: {stats['events_with_both']}")
        
        if ripple_durations:
            print(f"\nRipple Duration: {np.mean(ripple_durations)*1000:.1f} ± {np.std(ripple_durations)*1000:.1f} ms")
        
        if mua_durations:
            print(f"MUA Duration: {np.mean(mua_durations)*1000:.1f} ± {np.std(mua_durations)*1000:.1f} ms")
        
        if combined_durations:
            print(f"Combined Duration: {np.mean(combined_durations)*1000:.1f} ± {np.std(combined_durations)*1000:.1f} ms")
        
        if temporal_offsets_start:
            print(f"\nTemporal Offset (Start): {stats['mean_offset_start']*1000:.1f} ± {stats['std_offset_start']*1000:.1f} ms")
            print(f"  (Positive = MUA starts before ripple)")
            print(f"Temporal Offset (End): {stats['mean_offset_end']*1000:.1f} ± {stats['std_offset_end']*1000:.1f} ms")
            print(f"  (Positive = MUA ends after ripple)")
        
        return stats
    
    def plot_dual_boundary_comparison(self, stats=None):
        """
        Plot comprehensive comparison of ripple vs MUA durations and temporal offsets.
        
        Parameters
        ----------
        stats : dict, optional
            Statistics from analyze_dual_boundaries(). If None, will compute automatically.
        """
        if stats is None:
            stats = self.analyze_dual_boundaries()
        
        if stats is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        
        # Plot 1: Duration comparison scatter
        ax1 = axes[0, 0]
        if len(stats['ripple_durations']) > 0 and len(stats['mua_durations']) > 0:
            # Find events with both measurements
            ripple_both = []
            mua_both = []
            for event in self.swr_events:
                if event.get('ripple_duration') is not None and event.get('mua_duration') is not None:
                    ripple_both.append(event['ripple_duration'] * 1000)
                    mua_both.append(event['mua_duration'] * 1000)
            
            if ripple_both:
                ax1.scatter(ripple_both, mua_both, alpha=0.6, s=50, color='#3498DB')
                max_val = max(max(ripple_both), max(mua_both))
                ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Unity Line', alpha=0.7)
                ax1.set_xlabel('Ripple Duration (ms)', fontsize=11, fontweight='bold')
                ax1.set_ylabel('MUA Duration (ms)', fontsize=11, fontweight='bold')
                ax1.set_title('Ripple vs MUA Duration', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temporal offset histogram (start times)
        ax2 = axes[0, 1]
        if len(stats['temporal_offsets_start']) > 0:
            offsets_ms = stats['temporal_offsets_start'] * 1000
            ax2.hist(offsets_ms, bins=30, color='#9B59B6', alpha=0.7, edgecolor='black')
            ax2.axvline(0, color='k', linestyle='--', linewidth=2, label='No Offset')
            ax2.axvline(stats['mean_offset_start'] * 1000, color='r', linestyle='-', 
                       linewidth=2, label=f"Mean: {stats['mean_offset_start']*1000:.1f} ms")
            ax2.set_xlabel('Start Time Offset (ms)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax2.set_title('MUA Start - Ripple Start', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Duration distributions (box plot)
        ax3 = axes[1, 0]
        duration_data = []
        duration_labels = []
        if len(stats['ripple_durations']) > 0:
            duration_data.append(stats['ripple_durations'] * 1000)
            duration_labels.append('Ripple')
        if len(stats['mua_durations']) > 0:
            duration_data.append(stats['mua_durations'] * 1000)
            duration_labels.append('MUA')
        if len(stats['combined_durations']) > 0:
            duration_data.append(stats['combined_durations'] * 1000)
            duration_labels.append('Combined')
        
        if duration_data:
            bp = ax3.boxplot(duration_data, labels=duration_labels, patch_artist=True,
                            notch=True, showmeans=True)
            colors = ['#FF6B6B', '#2ECC71', '#F39C12']
            for patch, color in zip(bp['boxes'], colors[:len(duration_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax3.set_ylabel('Duration (ms)', fontsize=11, fontweight='bold')
            ax3.set_title('Duration Distributions', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Temporal offset histogram (end times)
        ax4 = axes[1, 1]
        if len(stats['temporal_offsets_end']) > 0:
            offsets_ms = stats['temporal_offsets_end'] * 1000
            ax4.hist(offsets_ms, bins=30, color='#E74C3C', alpha=0.7, edgecolor='black')
            ax4.axvline(0, color='k', linestyle='--', linewidth=2, label='No Offset')
            ax4.axvline(stats['mean_offset_end'] * 1000, color='r', linestyle='-', 
                       linewidth=2, label=f"Mean: {stats['mean_offset_end']*1000:.1f} ms")
            ax4.set_xlabel('End Time Offset (ms)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax4.set_title('MUA End - Ripple End', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def visualize_events(self, spike_data=None, spike_times_by_region=None, region_colors=None, remove_spines=True):
        """
        Create interactive visualization of detected events including MUA data and spike rasters.
        
        Parameters
        ----------
        spike_data : list of arrays, optional
            Legacy spike data format (list of spike times per unit)
        spike_times_by_region : dict, optional
            Dictionary mapping region names to dictionaries of unit spike times
            Format: {'CA1': {unit_id: spike_times_array, ...}, 'PFC': {...}, ...}
        region_colors : dict, optional
            Dictionary mapping region names to colors for spike raster
            Default: {'CA1': '#FF6B6B', 'RTC': '#4ECDC4', 'PFC': '#45B7D1'}
        remove_spines : bool, default=True
            Whether to remove top and right spines for cleaner appearance
        """
        if not self.swr_events:
            print("No events detected")
            return

        # Default region colors
        if region_colors is None:
            region_colors = {
                'CA1': '#FF6B6B',  # Red
                'RTC': '#4ECDC4',  # Teal
                'PFC': '#45B7D1',  # Blue
            }

        # Calculate minimum window size to accommodate all events
        max_event_duration = max([e['duration'] for e in self.swr_events])
        min_window = max(0.3, max_event_duration + 0.2)  # Add 0.2s buffer

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

        # Add slider for window length with dynamic minimum
        win_slider = widgets.FloatSlider(
            value=max(0.6, min_window),  # Start with safe default
            min=min_window,
            max=3.0,
            step=0.1,
            description='Window (s):',
            continuous_update=False,
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

                # Update slider minimum if current event exceeds it
                event_duration = event['duration']
                required_min = event_duration + 0.2
                if required_min > win_slider.min:
                    win_slider.min = required_min
                    if win_slider.value < required_min:
                        win_slider.value = required_min

                # Determine number of subplots based on available data
                # Base: 1 raw + 4 method panels + spike raster + MUA + context = 8 panels
                n_plots = 5  # Base count (used for subplot positioning logic)
                n_panels_total = 8  # Actual panel count: 1 raw + 4 methods + 3 data panels
                if self.velocity_data is not None:
                    n_panels_total += 1

                # Set subplot parameters for maximum spacing before creating figure
                plt.rcParams['figure.subplot.hspace'] = 1.0
                plt.rcParams['figure.subplot.bottom'] = 0.1
                plt.rcParams['figure.subplot.top'] = 0.95
                
                # Create figure with adjusted height (smaller per-panel height due to more panels)
                fig = plt.figure(figsize=(14, 2.5 * n_panels_total))
                
                # Disable all grids globally for this figure
                plt.rcParams['axes.grid'] = False
                
                # Calculate time window based on slider value
                window_size = win_slider.value
                peak_time = event['peak_time']
                t_start = peak_time - window_size / 2
                t_end = peak_time + window_size / 2

                # Plot 1: Raw trace and ripple/MUA
                ax1 = plt.subplot(n_plots, 1, 1)
                
                # Use slider window for raw trace
                mask = (event['trace_timestamps'] >= t_start) & (event['trace_timestamps'] <= t_end)
                time_windowed = event['trace_timestamps'][mask]
                raw_windowed = event['raw_trace'][mask]
                
                ax1.plot(time_windowed, raw_windowed, 'k', linewidth=1, label='Raw')

                if event['sharpwave_trace'] is not None:
                    sw_windowed = event['sharpwave_trace'][mask]
                    ax1.plot(time_windowed, sw_windowed, 'g',
                            linewidth=1, label='Sharp Wave')

                # Add highlight for event duration
                ax1.axvspan(event['start_time'], event['end_time'],
                            color='yellow', alpha=0.3, label='Event')
                ax1.axvline(x=event['peak_time'], color='r',
                            linestyle='--', linewidth=1.5, label='Peak')
                ax1.set_title(f"Event {event['event_id']} - Channel {event['channel']} "
                            f"(Type: {event['event_type']})", fontsize=12, fontweight='bold')
                ax1.legend(loc='upper right', frameon=False, fontsize=9)
                ax1.set_ylabel('Amplitude (μV)', fontsize=10)
                ax1.set_xlim(t_start, t_end)
                ax1.tick_params(axis='x', labelbottom=False)
                ax1.grid(False)
                
                # Remove spines if requested
                if remove_spines:
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['left'].set_linewidth(1.5)
                    ax1.spines['bottom'].set_linewidth(1.5)

                # Plot 2: Multi-Method Boundary Comparison (4 panels stacked)
                # This shows all 4 detection methods for edge detection quality assessment
                method_configs = [
                    ('Basic Threshold', 'start_time', 'end_time', '#f39c12', 'Basic'),
                    ('Hysteresis', 'hysteresis_start_time', 'hysteresis_end_time', '#3498db', 'Hyst'),
                    ('HMM Ripple', 'combined_start_time', 'combined_end_time', '#2ecc71', 'HMM-R'),
                    ('HMM MUA', 'mua_high_start', 'mua_high_end', '#e74c3c', 'HMM-M'),
                ]
                
                for method_idx, (method_label, start_key, end_key, color, short_label) in enumerate(method_configs):
                    ax_method = plt.subplot(n_plots + 3, 1, 2 + method_idx)  # Insert 4 panels after raw trace
                    
                    # Plot ripple power as reference
                    if event['ripple_power'] is not None:
                        power_windowed = event['ripple_power'][mask]
                        ax_method.plot(time_windowed, power_windowed, 'gray',
                                linewidth=1, alpha=0.4, label='Ripple Power')
                    
                    # Get boundaries for this method
                    start_time = event.get(start_key, None)
                    end_time = event.get(end_key, None)
                    
                    # Highlight detected boundaries
                    if start_time is not None and end_time is not None:
                        ax_method.axvspan(start_time, end_time, color=color, alpha=0.3, 
                                        label=f'{short_label} Duration')
                        duration_ms = (end_time - start_time) * 1000
                        ax_method.text(0.98, 0.85, f'{duration_ms:.1f}ms', 
                                    transform=ax_method.transAxes,
                                    fontsize=9, fontweight='bold', 
                                    ha='right', va='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    else:
                        ax_method.text(0.5, 0.5, f'No {short_label} boundaries', 
                                    transform=ax_method.transAxes,
                                    fontsize=10, color=color, alpha=0.7,
                                    ha='center', va='center')
                    
                    # Mark peak
                    ax_method.axvline(x=event['peak_time'], color='red', 
                                    linestyle='--', linewidth=1, alpha=0.5)
                    
                    ax_method.set_ylabel(short_label, fontsize=9, fontweight='bold')
                    ax_method.set_title(method_label, fontsize=10, fontweight='bold', pad=3)
                    ax_method.set_xlim(t_start, t_end)
                    ax_method.legend(loc='upper left', frameon=False, fontsize=8)
                    
                    # Clean appearance
                    if remove_spines:
                        ax_method.spines['top'].set_visible(False)
                        ax_method.spines['right'].set_visible(False)
                        ax_method.spines['left'].set_linewidth(1.5)
                        ax_method.spines['bottom'].set_linewidth(1.5)
                    
                    ax_method.tick_params(axis='x', labelbottom=False)
                    ax_method.grid(False)

                # Plot 6: Spike raster (adjusted index due to 4 method panels inserted)
                ax_spike = plt.subplot(n_plots + 3, 1, 6)  # Now at position 6 (after 1 raw + 4 methods)
                
                spike_plotted = False
                
                # Try region-based spike data first (preferred)
                if spike_times_by_region is not None:
                    all_spike_times = []
                    all_colors = []
                    all_labels = []
                    y_offset = 0
                    
                    for region, units_dict in spike_times_by_region.items():
                        region_color = region_colors.get(region, 'k')
                        
                        for unit_id, spike_times in units_dict.items():
                            spikes = np.asarray(spike_times)
                            mask = (spikes >= t_start) & (spikes <= t_end)
                            spikes_in_window = spikes[mask]
                            
                            if len(spikes_in_window) > 0:
                                all_spike_times.append(spikes_in_window)
                                all_colors.append(region_color)
                                all_labels.append(f"{region}-U{unit_id}")
                                y_offset += 1
                    
                    if all_spike_times:
                        ax_spike.eventplot(all_spike_times, colors=all_colors, 
                                        lineoffsets=np.arange(len(all_spike_times)),
                                        linelengths=0.8, linewidths=1.5)
                        ax_spike.set_ylim(-0.5, len(all_spike_times) - 0.5)
                        ax_spike.set_yticks(np.arange(len(all_spike_times)))
                        ax_spike.set_yticklabels(all_labels, fontsize=8)
                        spike_plotted = True
                
                # Fallback to legacy spike_data format
                elif spike_data is not None:
                    spike_times_in_window = []
                    for unit_spikes in spike_data:
                        spikes = np.array(unit_spikes)
                        mask = (spikes >= t_start) & (spikes <= t_end)
                        spike_times_in_window.append(spikes[mask])
                    
                    if any(len(s) > 0 for s in spike_times_in_window):
                        ax_spike.eventplot(spike_times_in_window, colors='k', 
                                        lineoffsets=np.arange(len(spike_times_in_window)),
                                        linelengths=0.8, linewidths=1.5)
                        ax_spike.set_ylabel('Units', fontsize=10)
                        spike_plotted = True
                
                # Show message if no spikes
                if not spike_plotted:
                    ax_spike.text(0.5, 0.5, 'No spike data available', 
                                ha='center', va='center', transform=ax_spike.transAxes,
                                fontsize=10, color='gray')
                    ax_spike.set_ylabel('Units', fontsize=10)
                
                # Add event markers
                ax_spike.axvspan(event['start_time'], event['end_time'], 
                            color='yellow', alpha=0.3, zorder=0)
                ax_spike.axvline(x=event['peak_time'], color='r', 
                            linestyle='--', linewidth=1.5, zorder=1)
                ax_spike.set_title('Spike Raster (by Region)', fontsize=11, fontweight='bold')
                ax_spike.set_xlim(t_start, t_end)
                ax_spike.tick_params(axis='x', labelbottom=False)
                
                # Remove spines if requested
                if remove_spines:
                    ax_spike.spines['top'].set_visible(False)
                    ax_spike.spines['right'].set_visible(False)
                    ax_spike.spines['left'].set_linewidth(1.5)
                    ax_spike.spines['bottom'].set_linewidth(1.5)
                
                ax_spike.grid(False)

                # Plot 7: MUA trace with HMM-detected boundaries
                ax3 = plt.subplot(n_plots + 3, 1, 7)  # Adjusted for 4 method panels
                if event['mua_trace'] is not None:
                    # Use slider window for MUA
                    mask = (event['trace_timestamps'] >= t_start) & (event['trace_timestamps'] <= t_end)
                    time_mua = event['trace_timestamps'][mask]
                    mua_windowed = event['mua_trace'][mask]
                    
                    ax3.plot(time_mua, mua_windowed, 
                            color='#2ECC71', linewidth=1.5, label='MUA')
                    ax3.fill_between(time_mua, 0, mua_windowed,
                                    color='#2ECC71', alpha=0.3)

                    # Overlay HMM state sequence for MUA if available
                    if 'mua_hmm_states' in event and event['mua_hmm_states'] is not None:
                        hmm_states = event['mua_hmm_states']
                        # Ensure state sequence matches window length
                        if len(hmm_states) == len(time_mua):
                            import matplotlib.colors as mcolors
                            state_colors = np.array([mcolors.to_rgba('lightgray', 0.3), mcolors.to_rgba('magenta', 0.3)])
                            # Plot colored background for each state
                            for state_val in np.unique(hmm_states):
                                state_mask = (hmm_states == state_val)
                                ax3.fill_between(time_mua, ax3.get_ylim()[0], ax3.get_ylim()[1], where=state_mask,
                                                color=state_colors[state_val], alpha=0.3, step='mid', zorder=0)
                            # Add legend entry for HMM states
                            from matplotlib.patches import Patch
                            handles = [Patch(facecolor='lightgray', alpha=0.3, label='HMM State 0'),
                                       Patch(facecolor='magenta', alpha=0.3, label='HMM State 1')]
                            # Add to existing legend if present
                            leg = ax3.get_legend()
                            if leg:
                                for h in handles:
                                    leg._legend_box._children.append(h)
                            else:
                                ax3.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)
                        else:
                            ax3.text(0.5, 0.9, 'HMM state length mismatch', transform=ax3.transAxes,
                                     fontsize=8, color='red', ha='center', va='center')
                    
                    # Show consensus (combined) boundaries in yellow
                    ax3.axvspan(event['start_time'], event['end_time'],
                                color='yellow', alpha=0.3, label='Consensus')
                    
                    # Show HMM-detected MUA boundaries in magenta
                    if event['mua_start_time'] is not None and event['mua_end_time'] is not None:
                        ax3.axvspan(event['mua_start_time'], event['mua_end_time'],
                                    color='magenta', alpha=0.4, label='HMM MUA', zorder=2)
                        # Add vertical lines at HMM boundaries for clarity
                        ax3.axvline(x=event['mua_start_time'], color='magenta',
                                linestyle=':', linewidth=2, alpha=0.8)
                        ax3.axvline(x=event['mua_end_time'], color='magenta',
                                linestyle=':', linewidth=2, alpha=0.8)
                    
                    ax3.axvline(x=event['peak_time'], color='r',
                                linestyle='--', linewidth=1.5, label='Peak')
                    ax3.set_xlim([t_start, t_end])
                    ax3.legend(loc='upper right', frameon=False, fontsize=9)
                    ax3.set_ylabel('Spike Rate (Hz)', fontsize=10)
                    ax3.set_title('Multi-Unit Activity (HMM MUA Edges)', fontsize=11, fontweight='bold')
                else:
                    ax3.text(0.5, 0.5, 'No MUA data', ha='center', va='center',
                            transform=ax3.transAxes, fontsize=10, color='gray')
                    ax3.set_ylabel('Spike Rate (Hz)', fontsize=10)
                    ax3.set_title('Multi-Unit Activity', fontsize=11, fontweight='bold')
                
                ax3.tick_params(axis='x', labelbottom=False)
                
                # Remove spines if requested
                if remove_spines:
                    ax3.spines['top'].set_visible(False)
                    ax3.spines['right'].set_visible(False)
                    ax3.spines['left'].set_linewidth(1.5)
                    ax3.spines['bottom'].set_linewidth(1.5)
                
                ax3.grid(False)

                # Plot 8: Context
                ax4 = plt.subplot(n_plots + 3, 1, 8)  # Adjusted for 4 method panels
                # Use slider window for context view
                peak_idx = int(event['peak_time'] * self.fs)
                context_start = max(0, int(peak_idx - self.fs * win_slider.value / 2))
                context_end = min(self.n_timepoints, int(peak_idx + self.fs * win_slider.value / 2))
                time_context = np.arange(context_start, context_end) / self.fs

                if isinstance(event['channel'], int):
                    signal_context = self.lfp_data[event['channel'], context_start:context_end]
                else:
                    signal_context = np.mean(self.lfp_data[:, context_start:context_end], axis=0)

                ax4.plot(time_context, signal_context, 'k', linewidth=1, label='LFP Signal')
                ax4.axvspan(event['start_time'], event['end_time'],
                            color='yellow', alpha=0.3, label='Detected Event')
                ax4.axvline(x=event['peak_time'], color='r',
                            linestyle='--', linewidth=1.5, label='Peak')
                ax4.set_xlim([t_start, t_end])
                ax4.set_title(f'Event Context (±{win_slider.value/2:.2f}s)', 
                            fontsize=11, fontweight='bold')
                ax4.set_ylabel('Amplitude (μV)', fontsize=10)
                ax4.set_xlabel('Time (s)', fontsize=10)
                ax4.legend(loc='upper right', frameon=False, fontsize=9)
                
                # Remove spines if requested
                if remove_spines:
                    ax4.spines['top'].set_visible(False)
                    ax4.spines['right'].set_visible(False)
                    ax4.spines['left'].set_linewidth(1.5)
                    ax4.spines['bottom'].set_linewidth(1.5)
                
                ax4.grid(False)

                # Plot 9: Velocity data if available
                if self.velocity_data is not None:
                    ax5 = plt.subplot(n_plots + 3, 1, 9)  # Adjusted for 4 method panels
                    velocity_trace = self.velocity_data[context_start:context_end]
                    ax5.plot(time_context, velocity_trace, color='#3498DB', 
                            linewidth=1.5, label='Velocity')
                    ax5.fill_between(time_context, 0, velocity_trace,
                                    color='#3498DB', alpha=0.2)

                    if self.params.velocity_threshold is not None:
                        ax5.axhline(y=self.params.velocity_threshold, color='red',
                                    linestyle='--', linewidth=2, alpha=0.7,
                                    label=f'Threshold ({self.params.velocity_threshold} cm/s)')

                    ax5.axvspan(event['start_time'], event['end_time'],
                                color='yellow', alpha=0.3)
                    ax5.axvline(x=event['peak_time'], color='r',
                                linestyle='--', linewidth=1.5)
                    ax5.set_xlim([t_start, t_end])
                    ax5.set_title('Velocity', fontsize=11, fontweight='bold')
                    ax5.set_xlabel('Time (s)', fontsize=10)
                    ax5.set_ylabel('Velocity (cm/s)', fontsize=10)
                    ax5.legend(loc='upper right', frameon=False, fontsize=9)
                    
                    # Remove spines if requested
                    if remove_spines:
                        ax5.spines['top'].set_visible(False)
                        ax5.spines['right'].set_visible(False)
                        ax5.spines['left'].set_linewidth(1.5)
                        ax5.spines['bottom'].set_linewidth(1.5)
                    
                    ax5.grid(False)

                    # Calculate mean velocity during event
                    event_start_idx = int(event['start_time'] * self.fs)
                    event_end_idx = int(event['end_time'] * self.fs)
                    mean_velocity = np.mean(self.velocity_data[event_start_idx:event_end_idx])

                plt.tight_layout(h_pad=6)
                plt.subplots_adjust(hspace=1.0)  # Even more vertical spacing

                # Print event details
                print(f"\nEvent {event['event_id']} Details:")
                print(f"Channel: {event['channel']}")
                print(f"Event type: {event['event_type']}")
                print(f"\nConsensus Boundaries:")
                print(f"  Start time: {event['start_time']:.3f} s")
                print(f"  Peak time: {event['peak_time']:.3f} s")
                print(f"  End time: {event['end_time']:.3f} s")
                print(f"  Duration: {(event['end_time'] - event['start_time']) * 1000:.1f} ms")
                
                # Print HMM-detected MUA boundaries if available
                if event['mua_start_time'] is not None and event['mua_end_time'] is not None:
                    print(f"\nHMM MUA Boundaries:")
                    print(f"  Start time: {event['mua_start_time']:.3f} s")
                    print(f"  End time: {event['mua_end_time']:.3f} s")
                    print(f"  Duration: {event['mua_duration'] * 1000:.1f} ms")
                
                print(f"\nPeak power: {event['peak_power']:.2f}")
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

        def on_win_change(change):
            plot_event(event_input.value)

        # Connect handlers
        prev_button.on_click(on_prev_clicked)
        next_button.on_click(on_next_clicked)
        event_input.observe(on_value_change, names='value')
        win_slider.observe(on_win_change, names='value')

        # Create layout
        buttons = widgets.HBox([prev_button, event_input, next_button, win_slider])

        # Display everything
        display(widgets.VBox([buttons, out]))

        # Show initial plot
        plot_event(0)

        # Print instructions
        print("\n" + "="*60)
        print("INTERACTIVE SWR EVENT BROWSER")
        print("="*60)
        print("\nNavigation Controls:")
        print("  • Click 'Previous' and 'Next' buttons to browse events")
        print("  • Type event number directly in the input box")
        print("  • Adjust 'Window (s)' slider to change event window length")
        print("\nFeatures:")
        print("  ✓ Dynamic window adjustment (prevents trace overflow)")
        print("  ✓ Region-colored spike rasters (if spike_times_by_region provided)")
        print("  ✓ Clean axes without box spines")
        print("  ✓ Enhanced multi-panel visualization")
        print("  ✓ HMM MUA boundary visualization (magenta shading)")
        print(f"\nTotal Events: {len(self.swr_events)}")
        print("="*60 + "\n")

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
            # Use MUA directly if already smoothed
            smooth_mua = self.mua_data
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

    def get_events_dataframe(self):
        """
        Return a pandas DataFrame with numeric per-event summary, including new spectral fields.
        """
        import pandas as pd
        if not self.swr_events:
            return pd.DataFrame()
        rows = []
        for ev in self.swr_events:
            rows.append({
                'event_id': ev.get('event_id'),
                'channel': ev.get('channel'),
                'start_time': ev.get('start_time'),
                'end_time': ev.get('end_time'),
                'duration': ev.get('combined_duration') or ev.get('duration'),
                'peak_time': ev.get('peak_time'),
                'peak_power': ev.get('peak_power'),
                'event_type': ev.get('event_type'),
                'ripple_high_threshold': ev.get('ripple_high_threshold'),
                'ripple_low_threshold': ev.get('ripple_low_threshold'),
                'mua_threshold': ev.get('mua_threshold'),
                'ripple_peak_freq': ev.get('ripple_peak_freq'),
                'ripple_peak_freq_power': ev.get('ripple_peak_freq_power'),
                'ripple_freq_class': ev.get('ripple_freq_class'),
                'group_type': ev.get('classification', {}).get('group_type') if ev.get('classification') else None,
                'group_size': ev.get('classification', {}).get('group_size') if ev.get('classification') else None
            })
        return pd.DataFrame(rows)
 
    def get_event_notes_dataframe(self):
        """
        Return a DataFrame where per-event raw arrays and thresholds are stored.
        Columns contain object arrays (numpy arrays) for traces.
        Useful for exporting or for saving to HDF/feather where arrays are supported.
        """
        import pandas as pd
        if not self.swr_events:
            return pd.DataFrame()
        rows = []
        for ev in self.swr_events:
            # Export both 'mua_trace' and legacy 'mua_vector' keys to ensure
            # notebooks and older code can find the MUA data under either name.
            # If per-event MUA was not stored at detection time but the detector
            # has a global mua_data vector, synthesize an excerpt here for plotting.
            mua_excerpt = ev.get('mua_trace')
            if mua_excerpt is None and getattr(self, 'mua_data', None) is not None:
                # try to build timestamps for the event window
                ts = ev.get('trace_timestamps', None)
                if ts is not None:
                    timestamps = np.asarray(ts)
                    # map timestamps to sample indices (assumes mua_data aligned to t=0)
                    idxs = np.round((timestamps - timestamps[0]) * getattr(self, 'fs', 1.0)).astype(int)
                    idxs = np.clip(idxs, 0, len(self.mua_data) - 1)
                    mua_excerpt = np.asarray(self.mua_data)[idxs]
                else:
                    # fallback: use basic_start/end times
                    start_t = ev.get('basic_start_time', ev.get('start_time', None))
                    end_t = ev.get('basic_end_time', ev.get('end_time', None))
                    if start_t is None or end_t is None:
                        peak = ev.get('peak_time', 0.0)
                        w = getattr(self.params, 'trace_window', 0.5)
                        start_t = peak - w / 2.0
                        end_t = peak + w / 2.0
                    fs = getattr(self, 'fs', 1.0)
                    n_samples = max(1, int(round((end_t - start_t) * fs)))
                    mua_idxs = np.linspace(int(round(start_t * fs)), int(round(end_t * fs)), n_samples).astype(int)
                    mua_idxs = np.clip(mua_idxs, 0, len(self.mua_data) - 1)
                    mua_excerpt = np.asarray(self.mua_data)[mua_idxs]

            rows.append({
                'event_id': ev.get('event_id'),
                'channel': ev.get('channel'),
                'raw_lfp': ev.get('raw_trace'),              # numpy array (object dtype)
                'filtered_lfp': ev.get('ripple_trace'),     # bandpassed ripple trace (object)
                'ripple_power': ev.get('ripple_power'),     # power envelope (object)
                'ripple_high_threshold': ev.get('ripple_high_threshold'),
                'ripple_low_threshold': ev.get('ripple_low_threshold'),
                'ripple_high_threshold_z': ev.get('ripple_high_threshold_z'),
                'ripple_low_threshold_z': ev.get('ripple_low_threshold_z'),
                'mua_trace': mua_excerpt,                    # preferred key for per-event MUA
                'mua_vector': mua_excerpt,                   # legacy name kept for compatibility
                'mua_threshold': ev.get('mua_threshold'),
                'mua_threshold_z': ev.get('mua_threshold_z'),
                'ripple_peak_amplitude': ev.get('ripple_peak_amplitude') if ev.get('ripple_peak_amplitude') is not None else ev.get('ripple_peak_freq_power'),
                'mua_peak_amplitude': ev.get('mua_peak_amplitude'),
                'trace_timestamps': ev.get('trace_timestamps')
            })
        return pd.DataFrame(rows)

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
            channel_events.sort(key=lambda x: (x.get('ripple_start_time') if x.get('ripple_start_time') is not None else x.get('start_time', 0.0)))
            # Extract event start times as the feature for clustering (prefer ripple boundaries)
            times = np.array([(e.get('ripple_start_time') if e.get('ripple_start_time') is not None else e.get('start_time', 0.0)) for e in channel_events]).reshape(-1, 1)

            # Skip DBSCAN if there are no events for this channel
            if times.shape[0] == 0:
                print(f"No events to classify for channel {ch}.")
                continue

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
                    group_events.sort(key=lambda x: (x.get('ripple_start_time') if x.get('ripple_start_time') is not None else x.get('start_time', 0.0)))
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
                        prev_end = group_events[i-1].get('ripple_end_time') or group_events[i-1].get('end_time', 0.0)
                        curr_start = group_events[i].get('ripple_start_time') or group_events[i].get('start_time', 0.0)
                        intervals.append(curr_start - prev_end)
    
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
        # Optional: pre-train global HMMs if requested
        if getattr(self.params, 'use_hmm_edge_detection', False) and getattr(self.params, 'use_global_hmm', False):
            # Train per-signal global HMMs if missing
            try:
                if 'ripple' not in self.global_hmms:
                    self.train_global_hmm(sig='ripple')
            except Exception as e:
                print(f"Global ripple HMM training skipped: {e}")
            try:
                if getattr(self.params, 'enable_mua', False) and self.mua_data is not None and 'mua' not in self.global_hmms:
                    self.train_global_hmm(sig='mua')
            except Exception as e:
                print(f"Global MUA HMM training skipped: {e}")
            try:
                if getattr(self.params, 'use_multivariate_hmm', False) and getattr(self.params, 'enable_mua', False) and self.mua_data is not None and 'joint' not in self.global_hmms:
                    self.train_global_hmm(sig='joint')
            except Exception as e:
                print(f"Global joint HMM training skipped: {e}")

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


def launch_event_inspector(detector=None, events_df=None, df_notes=None, plot_window_sec=0.4):
    """
    Launch an interactive Jupyter widget to inspect SWR events.

    Parameters
    - detector: SWRHMMDetector instance (optional). If None, will try to get 'detector' from the notebook globals.
    - events_df: pandas.DataFrame of events (optional). If None, will try to get 'events_df' from notebook globals or build from detector.swr_events.
    - df_notes: pandas.DataFrame with per-event notes/traces (optional). If None, will try to get 'df_notes' from notebook globals or create via detector.get_event_notes_dataframe().
    - plot_window_sec: float, seconds to show around the event peak when per-event traces are missing.

    Returns: the widget layout (ipywidgets.VBox). Call from a notebook cell to display.
    """
    # Local notebook-only imports to avoid import-time side effects
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from scipy.signal import hilbert

    # Helper to get notebook globals only when needed
    try:
        user_ns = get_ipython().user_ns
    except Exception:
        user_ns = {}

    # Resolve detector/events_df/df_notes
    if detector is None:
        detector = user_ns.get('detector', None)

    if events_df is None:
        events_df = user_ns.get('events_df', None)
        if (events_df is None or getattr(events_df, 'empty', True)) and detector is not None:
            try:
                events_df = pd.DataFrame(getattr(detector, 'swr_events', []))
                # do not overwrite user_ns automatically; return created df to caller via variable if desired
            except Exception:
                events_df = pd.DataFrame([])

    if df_notes is None and detector is not None:
        df_notes = user_ns.get('df_notes', None)
        if (df_notes is None or getattr(df_notes, 'empty', True)):
            try:
                df_notes = detector.get_event_notes_dataframe()
            except Exception:
                df_notes = pd.DataFrame([])

    # Basic validation
    if detector is None:
        raise ValueError("detector not provided and not found in notebook globals.")
    if events_df is None or getattr(events_df, 'empty', True):
        raise ValueError("events_df not provided and could not be created from detector.swr_events.")

    # helper: safely pick first non-None value from event for a list of keys
    def first_present(event, keys):
        for k in keys:
            if k in event:
                val = event.get(k)
                if val is not None:
                    return val
        return None

    # Core plotting function (mirrors your notebook code)
    def plot_event(event_index, events_df_local, detector_local, plot_window_sec_local=plot_window_sec):
        try:
            event = events_df_local.loc[event_index]

            peak_time = float(event['peak_time'])
            event_id = event.get('event_id', event_index)
            event_type = event.get('event_type', event.get('type', 'unknown'))
            channel = int(event.get('channel', 0))

            basic_start = event.get('basic_start_time', event.get('start_time'))
            basic_end = event.get('basic_end_time', event.get('end_time'))
            basic_dur = event.get('basic_duration', np.nan)

            hyst_start = event.get('hysteresis_start_time')
            hyst_end = event.get('hysteresis_end_time')
            hyst_dur = event.get('hysteresis_duration', np.nan)
            ripple_peak = event.get('ripple_peak_amplitude', event.get('peak_power', np.nan))
            ripple_high_th = event.get('ripple_high_threshold')
            ripple_low_th = event.get('ripple_low_threshold')

            mua_peak = event.get('mua_peak_amplitude', np.nan)
            mua_th = event.get('mua_threshold')

            fs = detector_local.fs
            plot_window_samples = int(plot_window_sec_local * fs)
            peak_sample = int(peak_time * fs)

            n_timepoints = detector_local.lfp_data.shape[1]
            start_sample = max(0, peak_sample - plot_window_samples // 2)
            end_sample = min(n_timepoints, peak_sample + plot_window_samples // 2)

            time_vec_lfp = np.arange(start_sample, end_sample) / fs
            time_vec_power = time_vec_lfp

            per_raw = first_present(event, ['raw_lfp', 'raw_trace', 'raw_lfp_trace', 'raw'])
            per_filtered = first_present(event, ['filtered_lfp', 'filtered', 'ripple_trace', 'ripple_trace_snippet'])
            per_power = event.get('ripple_power')
            per_mua = first_present(event, ['mua_trace', 'mua_vector', 'mua'])
            per_timestamps = first_present(event, ['trace_timestamps', 'timestamps', 'time_vector'])

            used_normalized = False

            if any(x is not None for x in (per_raw, per_filtered, per_power, per_mua)):
                L = None
                for x in (per_filtered, per_raw, per_power, per_mua):
                    if x is not None:
                        L = len(x)
                        break
                if L is None:
                    L = max(1, end_sample - start_sample)

                window = getattr(detector_local.params, 'trace_window', plot_window_sec_local)
                if per_timestamps is None:
                    per_timestamps = np.linspace(peak_time - window / 2, peak_time + window / 2, L)
                else:
                    per_timestamps = np.asarray(per_timestamps)

                if per_raw is not None:
                    lfp_snippet = np.asarray(per_raw)
                else:
                    raw_lfp_channel = detector_local.lfp_data[channel, :]
                    notched_full = detector_local._notch_filter(raw_lfp_channel, detector_local.params.notch_freq)
                    lfp_snippet = notched_full[start_sample:end_sample]

                if per_filtered is not None:
                    ripple_snippet = np.asarray(per_filtered)
                else:
                    if 'notched_full' not in locals():
                        raw_lfp_channel = detector_local.lfp_data[channel, :]
                        notched_full = detector_local._notch_filter(raw_lfp_channel, detector_local.params.notch_freq)
                    ripple_filtered_full = detector_local._bandpass_filter(notched_full, *detector_local.params.ripple_band)
                    ripple_snippet = ripple_filtered_full[start_sample:end_sample]

                if per_power is not None:
                    power_snippet = np.asarray(per_power)
                    time_vec_power = per_timestamps
                else:
                    analytic_signal_full = hilbert(ripple_filtered_full)
                    ripple_power_raw_full = np.abs(analytic_signal_full)
                    smooth_ripple_power_full = detector_local._gaussian_smooth(ripple_power_raw_full, detector_local.params.smoothing_sigma)
                    power_snippet = smooth_ripple_power_full[start_sample:end_sample]
                    if len(power_snippet) == len(per_timestamps):
                        time_vec_power = per_timestamps
                    else:
                        time_vec_power = time_vec_lfp

                if per_mua is not None:
                    mua_snippet = np.asarray(per_mua)
                else:
                    if getattr(detector_local, 'mua_data', None) is not None:
                        smooth_mua_full = detector_local._gaussian_smooth(detector_local.mua_data, detector_local.params.smoothing_sigma)
                        mua_snippet = smooth_mua_full[start_sample:end_sample]
                    else:
                        mua_snippet = None

                time_vec_lfp = per_timestamps

                if per_power is not None:
                    rp_ref = getattr(detector_local, 'ripple_power_global', None)
                    if rp_ref is None:
                        rp_ref = getattr(detector_local, 'ripple_power', None)
                    if rp_ref is not None:
                        rp_ref = np.asarray(rp_ref)
                        ref_mean = np.mean(rp_ref)
                        ref_std = np.std(rp_ref) if np.std(rp_ref) > 0 else 1.0
                        power_snippet = (power_snippet - ref_mean) / ref_std
                        used_normalized = True
                    else:
                        ref_mean = np.nanmean(power_snippet)
                        ref_std = np.nanstd(power_snippet) if np.nanstd(power_snippet) > 0 else 1.0
                        power_snippet = (power_snippet - ref_mean) / ref_std
                        used_normalized = True

            else:
                raw_lfp_channel = detector_local.lfp_data[channel, :]
                notched_full = detector_local._notch_filter(raw_lfp_channel, detector_local.params.notch_freq)
                ripple_filtered_full = detector_local._bandpass_filter(notched_full, *detector_local.params.ripple_band)
                analytic_signal_full = hilbert(ripple_filtered_full)
                ripple_power_raw_full = np.abs(analytic_signal_full)
                smooth_ripple_power_full = detector_local._gaussian_smooth(ripple_power_raw_full, detector_local.params.smoothing_sigma)

                if getattr(detector_local, 'mua_data', None) is not None:
                    smooth_mua_full = detector_local._gaussian_smooth(detector_local.mua_data, detector_local.params.smoothing_sigma)
                    mua_snippet = smooth_mua_full[start_sample:end_sample]
                else:
                    smooth_mua_full = None
                    mua_snippet = None

                lfp_snippet = notched_full[start_sample:end_sample]
                ripple_snippet = ripple_filtered_full[start_sample:end_sample]
                power_snippet = smooth_ripple_power_full[start_sample:end_sample]
                time_vec_power = time_vec_lfp
                used_normalized = False

                if getattr(detector_local.params, 'normalization_method', None) and smooth_ripple_power_full is not None:
                    ref = smooth_ripple_power_full
                    ref_mean = np.mean(ref)
                    ref_std = np.std(ref) if np.std(ref) > 0 else 1.0
                    power_snippet = (power_snippet - ref_mean) / ref_std
                    used_normalized = True

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

            try:
                x1 = time_vec_lfp if len(time_vec_lfp) == len(lfp_snippet) else (np.arange(len(lfp_snippet)) / fs + time_vec_lfp[0])
            except Exception:
                x1 = np.arange(len(lfp_snippet)) / fs + (time_vec_lfp[0] if len(time_vec_lfp) > 0 else 0)
            ax1.plot(x1, lfp_snippet, color='k')
            ax1.axvline(peak_time, color='r', linestyle='--', label=f"Peak: {peak_time:.3f} s")
            if basic_start is not None and basic_end is not None:
                ax1.axvspan(basic_start, basic_end, alpha=0.3, color='orange', label=f"Basic: {basic_dur*1000:.1f} ms")
            ax1.set_title("Panel 1: Notched LFP")
            ax1.legend(loc='upper right')
            ax1.set_ylabel("Voltage (uV)")

            try:
                x2 = time_vec_lfp if len(time_vec_lfp) == len(ripple_snippet) else (np.arange(len(ripple_snippet)) / fs + time_vec_lfp[0])
            except Exception:
                x2 = np.arange(len(ripple_snippet)) / fs + (time_vec_lfp[0] if len(time_vec_lfp) > 0 else 0)
            ax2.plot(x2, ripple_snippet, color='b')
            if basic_start is not None and basic_end is not None:
                ax2.axvspan(basic_start, basic_end, alpha=0.3, color='gray')
            ax2.axvline(peak_time, color='r', linestyle='--')
            ax2.set_title("Panel 2: Ripple-Filtered LFP")
            ax2.legend(loc='upper right')
            ax2.set_ylabel("Voltage (uV)")

            if power_snippet is not None:
                try:
                    xp = time_vec_power
                except Exception:
                    xp = time_vec_lfp
                label = 'Ripple Power (normalized)' if used_normalized else 'Ripple Power'
                ax3.plot(xp, power_snippet, color='g', lw=1.0, label=label)

            if ripple_high_th is not None:
                ax3.axhline(ripple_high_th, color='r', ls=':', label=f"High Th: {ripple_high_th:.2f}")
            if ripple_low_th is not None:
                ax3.axhline(ripple_low_th, color='orange', ls=':', label=f"Low Th: {ripple_low_th:.2f}")
            if hyst_start is not None and hyst_end is not None:
                ax3.axvspan(hyst_start, hyst_end, color='lightblue', alpha=0.25, label='Hysteresis')

            try:
                if (ripple_peak is not None and not (isinstance(ripple_peak, float) and np.isnan(ripple_peak))) and (power_snippet is not None):
                    idx = int(np.argmin(np.abs(np.asarray(xp) - peak_time)))
                    peak_val_plot = float(power_snippet[idx])
                    ax3.scatter([peak_time], [peak_val_plot], color='red', s=60, zorder=10, edgecolor='black', label='Ripple peak')
            except Exception:
                pass

            ax3.set_title("Panel 3: Ripple Power")
            ax3.legend(loc='upper right')
            ax3.set_ylabel("Power (z-score)" if used_normalized else "Power (a.u.)")

            if mua_snippet is not None:
                try:
                    xm = time_vec_lfp if len(time_vec_lfp) == len(mua_snippet) else (np.arange(len(mua_snippet)) / fs + time_vec_lfp[0])
                except Exception:
                    xm = np.arange(len(mua_snippet)) / fs + (time_vec_lfp[0] if len(time_vec_lfp) > 0 else 0)
                ax4.plot(xm, mua_snippet, color='purple', lw=0.9, label="MUA")
                if mua_th is not None:
                    ax4.axhline(mua_th, color='red', ls=':', label=f"MUA Th: {mua_th:.2f}")

                try:
                    if mua_peak is not None and not (isinstance(mua_peak, float) and np.isnan(mua_peak)):
                        if len(mua_snippet) > 0:
                            mua_peak_idx = int(np.argmax(mua_snippet))
                            mua_peak_time = xm[mua_peak_idx]
                            mua_plot_val = float(mua_snippet[mua_peak_idx])
                        else:
                            mua_peak_time = peak_time
                            mua_plot_val = float(mua_peak)
                    stored_mua_time = event.get('mua_peak_time', None)
                    if stored_mua_time is not None:
                        mua_peak_time = float(stored_mua_time)
                        try:
                            mua_plot_val = float(mua_snippet[int(np.argmin(np.abs(np.asarray(xm) - mua_peak_time)))])
                        except Exception:
                            mua_plot_val = float(mua_peak) if mua_peak is not None else np.nan
                    else:
                        if len(mua_snippet) > 0:
                            mua_peak_idx = int(np.argmax(mua_snippet))
                            mua_peak_time = xm[mua_peak_idx]
                            mua_plot_val = float(mua_snippet[mua_peak_idx])
                        else:
                            mua_peak_time = peak_time
                            mua_plot_val = float(mua_peak) if mua_peak is not None else np.nan
                    ax4.scatter([mua_peak_time], [mua_plot_val], color='red', s=60, zorder=10, edgecolor='black', label='MUA peak')
                    if event.get('combined_start_time') is not None and event.get('combined_end_time') is not None:
                        if not (event['combined_start_time'] <= mua_peak_time <= event['combined_end_time']):
                            ax4.scatter([mua_peak_time], [mua_plot_val], facecolors='none', edgecolors='red', s=120, linewidths=1.2, zorder=9)
                except Exception:
                    pass

                ax4.set_title("Panel 4: MUA")
                ax4.legend(loc='upper right')
                ax4.set_ylabel("MUA Firing Rate (a.u.)")
            else:
                ax4.text(0.5, 0.5, 'No MUA available', transform=ax4.transAxes, ha='center', va='center', fontsize=10, color='gray')
                ax4.set_ylabel("MUA")

            ax4.set_xlabel("Time (s)")
            fig.suptitle(f"Event ID: {event_id}  |  Type: '{event_type}'  (Channel {channel})", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        except Exception as e:
            print(f"Error plotting event {event_index}: {e}")
            import traceback
            traceback.print_exc()

    # Build widget UI
    n_events = len(events_df)
    out = widgets.Output()
    current_index = widgets.IntText(value=0, description='Event Index:', layout=widgets.Layout(width='160px'))
    prev_btn = widgets.Button(description='Previous', icon='arrow-left')
    next_btn = widgets.Button(description='Next', icon='arrow-right')
    total_events_label = widgets.Label(f"of {n_events - 1}")

    def on_prev_click(b):
        current_index.value = max(0, current_index.value - 1)

    def on_next_click(b):
        current_index.value = min(n_events - 1, current_index.value + 1)

    def on_index_change(change):
        events_df_local = events_df
        detector_local = detector
        if events_df_local is None or getattr(events_df_local, 'empty', True):
            with out:
                out.clear_output(wait=True)
                print("--- WIDGET ERROR ---")
                print("Error: 'events_df' is missing or empty.")
            return

        new_idx = change.get('new', None)
        if new_idx is None:
            return
        if not (0 <= int(new_idx) < len(events_df_local)):
            with out:
                out.clear_output(wait=True)
                print(f"Error: Index {new_idx} is out of bounds (0 to {len(events_df_local)-1}).")
            return

        with out:
            out.clear_output(wait=True)
            try:
                plot_event(int(new_idx), events_df_local, detector_local)
            except Exception as e:
                print("--- AN ERROR OCCURRED ---")
                print(f"Failed to plot event {new_idx}: {e}")
                import traceback
                traceback.print_exc()

    prev_btn.on_click(on_prev_click)
    next_btn.on_click(on_next_click)
    try:
        current_index.unobserve(on_index_change, names='value')
    except Exception:
        pass
    current_index.observe(on_index_change, names='value')

    controls = widgets.HBox([prev_btn, next_btn, current_index, total_events_label])
    widget_layout = widgets.VBox([controls, out])
    display(widget_layout)

    # trigger first plot
    on_index_change({'new': int(current_index.value)})

    # return widget for caller if they want to keep reference
    return widget_layout

