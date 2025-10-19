from __future__ import annotations

from typing import Dict, List

import numpy as np


def _gaussian_kernel(sigma_samples: float, truncate: float = 4.0) -> np.ndarray:
    """Return a normalized 1D Gaussian kernel in samples.

    Parameters
    ----------
    sigma_samples : float
        Standard deviation of the Gaussian in samples.
    truncate : float
        Truncate kernel at +/- truncate * sigma.
    """
    sigma_samples = float(max(sigma_samples, 1e-6))
    radius = int(truncate * sigma_samples)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_samples) ** 2)
    s = k.sum()
    if s > 0:
        k /= s
    return k


def compute_region_mua_from_spikes(
    spike_times_by_region: Dict[str, List[np.ndarray]],
    t_lfp: np.ndarray,
    sigma: float = 0.05,
    normalize: str = "sum",
) -> Dict[str, np.ndarray]:
    """
    Build region-wise MUA/firing-rate vectors aligned to the LFP timebase from spike timestamps.

    Parameters
    ----------
    spike_times_by_region : dict
        Mapping {region: [spike_times_sec_per_unit, ...]} where each spike_times array is in seconds.
    t_lfp : np.ndarray
        1D time vector (seconds) for the LFP timeline.
    sigma : float
        Gaussian smoothing width in seconds (default 0.05 -> 50 ms).
    normalize : str
        "sum" to sum rates across units (population), or "mean" for average per unit.

    Returns
    -------
    dict
        {region: 1D np.ndarray} of length len(t_lfp).
    """
    t_lfp = np.asarray(t_lfp, float).squeeze()
    if t_lfp.ndim != 1 or t_lfp.size < 2:
        raise ValueError("t_lfp must be a 1D time vector with length > 1")

    # Derive bin edges from t_lfp assuming near-uniform sampling
    dt = float(np.median(np.diff(t_lfp)))
    edges = np.r_[t_lfp - 0.5 * dt, t_lfp[-1] + 0.5 * dt]

    # Smoothing kernel in samples
    k = _gaussian_kernel(sigma_samples=max(sigma / dt, 1e-6))

    out: Dict[str, np.ndarray] = {}
    for region, units in spike_times_by_region.items():
        units = units or []
        if len(units) == 0:
            out[region] = np.zeros_like(t_lfp, dtype=float)
            continue

        # Bin spikes per unit onto LFP timeline
        counts_per_unit = []
        for st in units:
            st = np.asarray(st, float).squeeze()
            if st.size == 0:
                counts_per_unit.append(np.zeros_like(t_lfp, dtype=float))
                continue
            c, _ = np.histogram(st, bins=edges)
            counts_per_unit.append(c.astype(float))

        if counts_per_unit:
            counts = np.vstack(counts_per_unit)
        else:
            counts = np.zeros((0, t_lfp.size), dtype=float)

        if normalize == "mean" and counts.shape[0] > 0:
            rate = counts.mean(axis=0) / dt
        else:
            rate = counts.sum(axis=0) / dt

        # Smooth with Gaussian kernel
        rate_sm = np.convolve(rate, k, mode="same")
        out[region] = rate_sm

    return out
