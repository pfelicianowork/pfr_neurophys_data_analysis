"""
params.py - Configuration parameters for SWR detection

This module defines the SWRParams class which contains all configuration
parameters for sharp wave ripple detection and analysis.
"""

import numpy as np


class SWRParams:
    """
    Configuration parameters for SWR detection and analysis.

    This class encapsulates all parameters needed for detecting sharp wave ripples,
    including signal processing, event detection, classification, and analysis settings.
    """

    def __init__(self,
                 # Core detection parameters
                 ripple_band=(150, 250),
                 threshold_multiplier=3,
                 min_duration=0.03,
                 max_duration=0.4,

                 # Signal processing parameters
                 notch_freq=60,
                 sharpwave_band=None,
                 velocity_threshold=None,
                 trace_window=0.2,

                 # Event detection parameters
                 duration_std_threshold=0.1,
                 min_event_separation=0.05,
                 merge_threshold=0.8,

                 # Classification parameters
                 single_separation=0.2,
                 burst_min_interval=0.07,
                 burst_max_interval=0.2,
                 merge_interval=0.07,

                 # MUA parameters
                 mua_threshold_multiplier=2.5,
                 mua_min_duration=0.02,
                 enable_mua=True,

                 # Advanced classification parameters
                 adaptive_classification=False,
                 dbscan_eps=0.2,
                 dbscan_min_samples=1,

                 # HMM edge detection parameters
                 use_hmm_edge_detection=True,
                 hmm_margin=0.1,

                 # Analysis parameters
                 analysis_window=1.0,
                 smoothing_kernel=0.05):
        """
        Initialize SWR detection parameters.

        Parameters
        ----------
        ripple_band : tuple
            Frequency band for ripple detection in Hz (low, high)
        threshold_multiplier : float
            Threshold multiplier in standard deviations
        min_duration : float
            Minimum event duration in seconds
        max_duration : float
            Maximum event duration in seconds
        notch_freq : float or None
            Notch filter frequency for line noise removal
        sharpwave_band : tuple or None
            Frequency band for sharp wave detection
        velocity_threshold : float or None
            Velocity threshold for movement filtering
        trace_window : float
            Window size for trace extraction in seconds
        duration_std_threshold : float
            Duration standard deviation threshold
        min_event_separation : float
            Minimum separation between events in seconds
        merge_threshold : float
            Threshold for merging close events
        single_separation : float
            Minimum separation for single event classification
        burst_min_interval : float
            Minimum interval for burst classification
        burst_max_interval : float
            Maximum interval for burst classification
        merge_interval : float
            Interval for merging close events
        mua_threshold_multiplier : float
            MUA detection threshold multiplier
        mua_min_duration : float
            Minimum MUA event duration
        enable_mua : bool
            Whether to enable MUA detection
        adaptive_classification : bool
            Whether to use adaptive classification
        dbscan_eps : float
            DBSCAN epsilon parameter
        dbscan_min_samples : int
            DBSCAN minimum samples parameter
        use_hmm_edge_detection : bool
            Whether to use HMM-based edge detection
        hmm_margin : float
            Margin for HMM analysis in seconds
        analysis_window : float
            Window size for analysis in seconds
        smoothing_kernel : float
            Smoothing kernel width in seconds
        """

        # Core detection parameters
        self.ripple_band = ripple_band
        self.threshold_multiplier = threshold_multiplier
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Signal processing parameters
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

        # Advanced classification parameters
        self.adaptive_classification = adaptive_classification
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # HMM edge detection parameters
        self.use_hmm_edge_detection = use_hmm_edge_detection
        self.hmm_margin = hmm_margin

        # Analysis parameters
        self.analysis_window = analysis_window
        self.smoothing_kernel = smoothing_kernel

    def update(self, **kwargs):
        """
        Update parameters with new values.

        Parameters
        ----------
        **kwargs
            Parameter names and new values

        Raises
        ------
        ValueError
            If an invalid parameter name is provided
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def copy(self):
        """
        Create a copy of the current parameters.

        Returns
        -------
        SWRParams
            Copy of current parameters
        """
        return SWRParams(**self.__dict__)

    def get_core_params(self):
        """
        Get core detection parameters as dictionary.

        Returns
        -------
        dict
            Core parameters for detection
        """
        return {
            'ripple_band': self.ripple_band,
            'threshold_multiplier': self.threshold_multiplier,
            'min_duration': self.min_duration,
            'max_duration': self.max_duration
        }

    def get_processing_params(self):
        """
        Get signal processing parameters as dictionary.

        Returns
        -------
        dict
            Signal processing parameters
        """
        return {
            'notch_freq': self.notch_freq,
            'sharpwave_band': self.sharpwave_band,
            'velocity_threshold': self.velocity_threshold,
            'trace_window': self.trace_window
        }

    def get_classification_params(self):
        """
        Get classification parameters as dictionary.

        Returns
        -------
        dict
            Classification parameters
        """
        return {
            'single_separation': self.single_separation,
            'burst_min_interval': self.burst_min_interval,
            'burst_max_interval': self.burst_max_interval,
            'merge_interval': self.merge_interval,
            'adaptive_classification': self.adaptive_classification,
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples
        }

    def get_mua_params(self):
        """
        Get MUA-related parameters as dictionary.

        Returns
        -------
        dict
            MUA parameters
        """
        return {
            'mua_threshold_multiplier': self.mua_threshold_multiplier,
            'mua_min_duration': self.mua_min_duration,
            'enable_mua': self.enable_mua
        }

    def get_hmm_params(self):
        """
        Get HMM-related parameters as dictionary.

        Returns
        -------
        dict
            HMM parameters
        """
        return {
            'use_hmm_edge_detection': self.use_hmm_edge_detection,
            'hmm_margin': self.hmm_margin
        }

    def validate_params(self):
        """
        Validate parameter values.

        Raises
        ------
        ValueError
            If any parameter values are invalid
        """
        # Validate frequency bands
        if self.ripple_band[0] >= self.ripple_band[1]:
            raise ValueError("Ripple band low frequency must be < high frequency")

        if self.sharpwave_band is not None:
            if self.sharpwave_band[0] >= self.sharpwave_band[1]:
                raise ValueError("Sharpwave band low frequency must be < high frequency")

        # Validate duration limits
        if self.min_duration <= 0:
            raise ValueError("min_duration must be > 0")

        if self.max_duration <= self.min_duration:
            raise ValueError("max_duration must be > min_duration")

        # Validate threshold multipliers
        if self.threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be > 0")

        if self.mua_threshold_multiplier <= 0:
            raise ValueError("mua_threshold_multiplier must be > 0")

        # Validate intervals
        if self.burst_min_interval <= 0:
            raise ValueError("burst_min_interval must be > 0")

        if self.burst_max_interval <= self.burst_min_interval:
            raise ValueError("burst_max_interval must be > burst_min_interval")

        if self.single_separation <= 0:
            raise ValueError("single_separation must be > 0")

        if self.merge_interval <= 0:
            raise ValueError("merge_interval must be > 0")

        print("✓ All parameters validated successfully")

    def print_summary(self):
        """
        Print a formatted summary of all parameters.
        """
        print("\nSWR Detection Parameters Summary")
        print("=" * 50)

        print("\nCore Detection Parameters:")
        print(f"  Ripple band: {self.ripple_band} Hz")
        print(f"  Threshold multiplier: {self.threshold_multiplier} SD")
        print(f"  Duration limits: {self.min_duration}-{self.max_duration} s")

        print("\nSignal Processing:")
        print(f"  Notch frequency: {self.notch_freq if self.notch_freq else 'Disabled'} Hz")
        print(f"  Sharp wave band: {self.sharpwave_band if self.sharpwave_band else 'Disabled'} Hz")
        print(f"  Velocity threshold: {self.velocity_threshold if self.velocity_threshold else 'Disabled'} cm/s")
        print(f"  Trace window: {self.trace_window} s")

        print("\nClassification Parameters:")
        print(f"  Single separation: {self.single_separation * 1000:.0f} ms")
        print(f"  Burst intervals: {self.burst_min_interval * 1000:.0f}-{self.burst_max_interval * 1000:.0f} ms")
        print(f"  Merge interval: {self.merge_interval * 1000:.0f} ms")
        print(f"  Adaptive classification: {self.adaptive_classification}")

        print("\nMUA Parameters:")
        print(f"  MUA detection: {'Enabled' if self.enable_mua else 'Disabled'}")
        if self.enable_mua:
            print(f"  MUA threshold: {self.mua_threshold_multiplier} SD")
            print(f"  MUA min duration: {self.mua_min_duration * 1000:.0f} ms")

        print("\nHMM Edge Detection:")
        print(f"  HMM edge detection: {'Enabled' if self.use_hmm_edge_detection else 'Disabled'}")
        if self.use_hmm_edge_detection:
            print(f"  HMM margin: {self.hmm_margin} s")

        print("\nAnalysis Parameters:")
        print(f"  Analysis window: {self.analysis_window} s")
        print(f"  Smoothing kernel: {self.smoothing_kernel} s")

        print("=" * 50)

    def save_to_file(self, filename):
        """
        Save parameters to a JSON file.

        Parameters
        ----------
        filename : str
            Output filename
        """
        import json

        # Convert parameters to dictionary
        params_dict = {}
        for key in self.__dict__:
            value = getattr(self, key)
            # Convert numpy types to native Python types
            if isinstance(value, np.ndarray):
                params_dict[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                params_dict[key] = value.item()
            else:
                params_dict[key] = value

        with open(filename, 'w') as f:
            json.dump(params_dict, f, indent=2)

        print(f"Parameters saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        """
        Load parameters from a JSON file.

        Parameters
        ----------
        filename : str
            Input filename

        Returns
        -------
        SWRParams
            Loaded parameters
        """
        import json

        with open(filename, 'r') as f:
            params_dict = json.load(f)

        return cls(**params_dict)


# Default parameter presets for different use cases
PRESETS = {
    'conservative': SWRParams(
        ripple_band=(150, 250),
        threshold_multiplier=3.5,
        min_duration=0.04,
        max_duration=0.3,
        mua_threshold_multiplier=3.0,
        use_hmm_edge_detection=True,
        hmm_margin=0.15
    ),

    'sensitive': SWRParams(
        ripple_band=(125, 300),
        threshold_multiplier=2.5,
        min_duration=0.025,
        max_duration=0.5,
        mua_threshold_multiplier=2.0,
        use_hmm_edge_detection=True,
        hmm_margin=0.1
    ),

    'hippocampal': SWRParams(
        ripple_band=(150, 250),
        threshold_multiplier=3,
        min_duration=0.03,
        max_duration=0.4,
        sharpwave_band=(2, 10),
        mua_threshold_multiplier=2.5,
        enable_mua=True,
        use_hmm_edge_detection=True,
        hmm_margin=0.1
    ),

    'cortical': SWRParams(
        ripple_band=(100, 200),
        threshold_multiplier=2.8,
        min_duration=0.025,
        max_duration=0.35,
        mua_threshold_multiplier=2.2,
        enable_mua=True,
        use_hmm_edge_detection=True,
        hmm_margin=0.12
    )
}
