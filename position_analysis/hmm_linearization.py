from typing import Optional
import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None  # type: ignore


essage = "hmmlearn is not installed. Install it to use HMM-based segmentation."


def hmm_segment_boundaries(signal_envelope: np.ndarray,
                           n_states: int = 2,
                           random_state: int = 42) -> Optional[np.ndarray]:
    """
    Fit a GaussianHMM to a 1D envelope (e.g., ripple power or distance-to-edge).
    Returns state sequence (ints) or None if hmmlearn is not available.
    """
    if GaussianHMM is None:
        print(essage)
        return None
    if signal_envelope.ndim != 1 or signal_envelope.size < 5:
        raise ValueError("signal_envelope must be a 1D vector with sufficient length")
    X = signal_envelope.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=random_state)
    model.fit(X)
    states = model.predict(X)
    return states
