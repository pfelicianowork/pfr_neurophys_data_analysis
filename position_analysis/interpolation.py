from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d


def interpolate_position(
    time: np.ndarray,
    position: np.ndarray,
    target_fs: Optional[float] = None,
    target_vector: Optional[np.ndarray] = None,
    interp_func: Optional[interp1d] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate position to a target sampling rate or target time vector.
    - Provide either target_fs or target_vector.
    - Returns (new_time, new_position).
    """
    if not isinstance(time, np.ndarray) or not isinstance(position, np.ndarray):
        raise TypeError("time and position must be numpy arrays")
    if time.ndim != 1 or position.ndim != 1:
        raise ValueError("time and position must be 1D arrays")
    if len(time) != len(position):
        raise ValueError("time and position must have the same length")

    if target_fs is not None and target_vector is None:
        duration = float(time[-1] - time[0])
        num_samples = max(int(round(duration * target_fs)), 2)
        target_vector = np.linspace(time[0], time[-1], num_samples)
    elif target_vector is not None and target_fs is None:
        if not isinstance(target_vector, np.ndarray):
            raise TypeError("target_vector must be a numpy array")
    else:
        raise ValueError("Provide either target_fs or target_vector, not both")

    if interp_func is None:
        interp_func = interp1d(time, position, kind='linear', fill_value="extrapolate", bounds_error=False)

    new_time = target_vector
    new_position = interp_func(new_time)
    return new_time, new_position
