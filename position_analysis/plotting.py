from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt



def plot_position_data(time: np.ndarray, position: np.ndarray,
                       title: str = 'Position vs Time', xlabel: str = 'Time (s)',
                       ylabel: str = 'Position (cm)') -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(time, position, label='Position')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_interpolated_comparison(original_time: np.ndarray, original_pos: np.ndarray,
                                 interp_time: np.ndarray, interp_pos: np.ndarray,
                                 title: str = 'Original vs Interpolated Position') -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(original_time, original_pos, 'b-', lw=2.0, label=f'Original ({len(original_time)} pts)')
    plt.plot(interp_time, interp_pos, 'r-', lw=1.0, label=f'Interpolated ({len(interp_time)} pts)')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (cm)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_velocity_comparison(time: np.ndarray, velocity: np.ndarray,
                             title: str = 'Estimated Velocity') -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(time, velocity, 'r-', lw=1.0, label=f'Estimated Velocity ({len(velocity)} pts)')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# position_analysis/plotting.py



def plot_interpolation_corr(tvel, linVel, t_lfp, velocity, show=True, return_corr=True):
    """
    Plot and compare original and interpolated velocity traces on both position and LFP timelines.
    Returns (corr_pos, corr_lfp).
    """
    # Ensure 1D arrays
    tvel = np.asarray(tvel).astype(float).squeeze()
    linVel = np.asarray(linVel).astype(float).squeeze()
    t_lfp = np.asarray(t_lfp).astype(float).squeeze()
    velocity = np.asarray(velocity).astype(float).squeeze()

    # Overlapping window
    t0 = max(tvel[0], t_lfp[0])
    t1 = min(tvel[-1], t_lfp[-1])
    mask_pos = (tvel >= t0) & (tvel <= t1)
    mask_lfp = (t_lfp >= t0) & (t_lfp <= t1)

    tvel_ov = tvel[mask_pos]
    linVel_ov = linVel[mask_pos]
    tlfp_ov = t_lfp[mask_lfp]
    vel_lfp_ov = velocity[mask_lfp]

    # Project both ways so curves share the same x-axis
    vel_back_on_pos = np.interp(tvel_ov, tlfp_ov, vel_lfp_ov)   # interpolated -> pos timeline
    linVel_on_lfp   = np.interp(tlfp_ov, tvel_ov, linVel_ov)    # original -> LFP timeline

    # Quick correlation on the overlap (pos timeline)
    def safe_corr(a, b):
        a = a - np.nanmean(a); b = b - np.nanmean(b)
        da = np.nanstd(a); db = np.nanstd(b)
        return np.nan if (da == 0 or db == 0) else float(np.nansum(a*b) / (len(a)*da*db))

    corr_pos = safe_corr(linVel_ov, vel_back_on_pos)
    corr_lfp = safe_corr(linVel_on_lfp, vel_lfp_ov)

    # Plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 3), sharex=False)

    # A) On position timeline
    axs[0].plot(tvel_ov, linVel_ov, lw=1.0, label="Original (pos timeline)")
    axs[0].plot(tvel_ov, vel_back_on_pos, lw=0.8, label="Interpolated→pos timeline")
    axs[0].set_title(f"Overlay on pos timeline (r={corr_pos:.4f})")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity")
    axs[0].legend(loc="upper right")

    # B) On LFP timeline
    axs[1].plot(tlfp_ov, vel_lfp_ov, lw=1.0, label="Interpolated (LFP timeline)")
    axs[1].plot(tlfp_ov, linVel_on_lfp, lw=0.8, label="Original→LFP timeline")
    axs[1].set_title(f"Overlay on LFP timeline (r={corr_lfp:.4f})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity")
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_corr:
        return corr_pos, corr_lfp


def plot_interpolation_alignment(t_original, vel_original, t_interp, vel_interp, show=True):
    """
    Plot original and interpolated velocity traces for visual inspection.
    Returns Pearson correlation coefficients for both traces.
    """
    # Compute correlation coefficients
    from scipy.stats import pearsonr
    # Interpolate original velocity to interpolated timeline for fair comparison
    vel_original_on_interp = np.interp(t_interp, t_original, vel_original)
    corr_pos = pearsonr(vel_original, np.interp(t_original, t_interp, vel_interp))[0]
    corr_lfp = pearsonr(vel_original_on_interp, vel_interp)[0]

    if show:
        plt.figure(figsize=(10, 4))
        plt.plot(t_original, vel_original, label='Original velocity', alpha=0.7)
        plt.plot(t_interp, vel_interp, label='Interpolated velocity', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.title('Original vs Interpolated Velocity')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return corr_pos, corr_lfp