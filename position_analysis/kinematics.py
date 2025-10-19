import numpy as np


def estimate_velocity(time: np.ndarray, position: np.ndarray) -> np.ndarray:
    """
    Estimate velocity from position data using numerical differentiation.
    Returns velocity with same length as input by edge-padding the gradient.
    """
    if len(time) != len(position):
        raise ValueError("Time and position arrays must have the same length")
    vel = np.gradient(position, time)
    return vel
