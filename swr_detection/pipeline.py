from __future__ import annotations
# --- Ripple bandpass filter utility ---
import re
from typing import Dict, List, Optional, Tuple, Union
from math import ceil
import copy
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from . import SWRDetector, SWRParams, PRESETS
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # optional for nicer palettes
except Exception:  # pragma: no cover
    sns = None


__all__ = [
    "find_region_channels",
    "build_region_lfp",
    "interpolate_velocity_to_lfp",
    "interpolate_signal_to_lfp",
    "compute_immobility_mask",
    "detector_events_to_df",
    "pick_time_column",
    "detect_swr_by_region",
    "quick_overlay_plot",
    "create_region_event_browser",
    "build_spike_times_by_region",
    "compute_region_mua_from_spikes",
    "summarize_region_df",
    "analyze_events_by_region",
    "plot_events_by_region_basic",
]

def bandpass_ripple_lfp(lfp_array, fs, ripple_band=(150, 250)):
    """
    Bandpass filter LFP array in the ripple band (default: 150-250 Hz).
    Returns the filtered trace.
    """
    b, a = butter(4, [ripple_band[0]/(fs/2), ripple_band[1]/(fs/2)], btype='band')
    return filtfilt(b, a, lfp_array)

def _get_unit_spike_times(sa, unit_id) -> np.ndarray:
    """Return spike times from common SpikeAnalysis-style containers."""
    if hasattr(sa, "get_unit_spike_times"):
        return np.asarray(sa.get_unit_spike_times(unit_id), float).squeeze()
    if hasattr(sa, "units") and unit_id in getattr(sa, "units"):
        unit = sa.units[unit_id]
        for key in ("times", "spike_times", "spike_times_s", "t"):
            if isinstance(unit, dict) and key in unit:
                return np.asarray(unit[key], float).squeeze()
    return np.asarray([], float)


def find_region_channels(selected_names: List[str],
                         regions: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Infer a mapping {region: [channel_names]} based on channel name prefixes.
    Example: ["CA1_tet1","RTC_tet1","PFC_tet2"] -> {"CA1":[...], "RTC":[...], "PFC":[...]}
    """
    regions = regions or ["CA1", "RTC", "PFC"]
    out: Dict[str, List[str]] = {r: [] for r in regions}
    for r in regions:
        pat = re.compile(rf"^{re.escape(r)}", re.IGNORECASE)
        out[r] = [n for n in selected_names if pat.match(n)]
    return out


def build_region_lfp(loader,
                     region_channels: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Load traces from loader into per-region 2D arrays [n_chan, n_samples].
    """
    region_lfp: Dict[str, np.ndarray] = {}
    for region, chans in region_channels.items():
        if not chans:
            continue
        traces = [np.asarray(loader.get_selected_trace(ch), dtype=float) for ch in chans]
        region_lfp[region] = np.vstack(traces)
    return region_lfp


def interpolate_velocity_to_lfp(t_pos: np.ndarray,
                                v_pos: np.ndarray,
                                t_lfp: np.ndarray) -> np.ndarray:
    """
    Interpolate position-derived velocity to the LFP timebase.
    """
    t_pos = np.asarray(t_pos, float).squeeze()
    v_pos = np.asarray(v_pos, float).squeeze()
    t_lfp = np.asarray(t_lfp, float).squeeze()
    return np.interp(t_lfp, t_pos, v_pos)

def interpolate_signal_to_lfp(t_src: np.ndarray,
                              x_src: np.ndarray,
                              t_lfp: np.ndarray) -> np.ndarray:
    """Generic 1D interpolation helper for signals (e.g., firing rate, MUA) to LFP timebase."""
    t_src = np.asarray(t_src, float).squeeze()
    x_src = np.asarray(x_src, float).squeeze()
    t_lfp = np.asarray(t_lfp, float).squeeze()
    return np.interp(t_lfp, t_src, x_src)

def compute_immobility_mask(velocity: np.ndarray, v_thresh: float = 5.0) -> np.ndarray:
    """
    Compute immobility as velocity < v_thresh.
    """
    velocity = np.asarray(velocity, float).squeeze()
    return (velocity < v_thresh).astype(bool)


def detector_events_to_df(det: SWRDetector) -> pd.DataFrame:
    """
    Robustly extract SWR events as a DataFrame from various detector implementations.
    """
    if hasattr(det, "to_dataframe"):
        return det.to_dataframe()
    if hasattr(det, "events_df"):
        df = det.events_df
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    if hasattr(det, "events"):
        try:
            return pd.DataFrame(det.events)
        except Exception:
            pass
    if hasattr(det, "swr_events"):
        try:
            return pd.DataFrame(det.swr_events)
        except Exception:
            pass
    raise AttributeError("Could not extract events DataFrame from SWRDetector.")


def pick_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Pick a reasonable single time column for event indexing/overlay.
    """
    for c in ("t_peak", "peak_time", "center_time", "t_center"):
        if c in df.columns:
            return c
    return None


# def detect_swr_by_region(region_lfp: Dict[str, np.ndarray],
#                          fs: float,
#                          velocity: Optional[np.ndarray] = None,
#                          params: Optional[SWRParams] = None,
#                          immobility_mask: Optional[np.ndarray] = None,
#                          classify: bool = True,
#                          channels: Union[str, List[int]] = "all",
#                          average_mode: bool = False
#                          ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
#     """
#     Run SWR detection per region on multichannel LFP arrays.

#     - region_lfp: dict[region] -> array [n_chan, n_samples]
#     - fs: sampling frequency (Hz)
#     - velocity: optional velocity aligned to LFP timeline (not required)
#     - immobility_mask: optional boolean mask for filtering events after detection
#     - params: SWRParams (defaults to hippocampal preset)
#     """
#     params = params or SWRParams.PRESETS.get("hippocampal", SWRParams())
#     events_by_region: Dict[str, pd.DataFrame] = {}

#     for region, lfp in region_lfp.items():
#         lfp = np.asarray(lfp, float)
#         if lfp.ndim != 2:
#             raise ValueError(f"{region}: LFP must be 2D [n_chan, n_samples], got shape {lfp.shape}")

#         det = SWRDetector(lfp_data=lfp, fs=float(fs), params=params)
#         det.detect_events(channels=channels, average_mode=average_mode)

#         if classify and hasattr(det, "classify_events"):
#             det.classify_events()

#         df = detector_events_to_df(det).copy()
#         df["region"] = region

#         # Optional immobility filtering (after detection)
#         if immobility_mask is not None and len(df):
#             # pick a representative time per event
#             tcol = pick_time_column(df)
#             if tcol is None and {"t_start", "t_end"}.issubset(df.columns):
#                 t_center = 0.5 * (df["t_start"].to_numpy() + df["t_end"].to_numpy())
#             elif tcol is None and {"start_time", "end_time"}.issubset(df.columns):
#                 t_center = 0.5 * (df["start_time"].to_numpy() + df["end_time"].to_numpy())
#             else:
#                 t_center = df[tcol].to_numpy() if (tcol is not None) else None

#             if t_center is not None:
#                 idx = np.clip((np.asarray(t_center) * fs).astype(int), 0, len(immobility_mask) - 1)
#                 keep = np.asarray(immobility_mask, bool)[idx]
#                 df = df.loc[keep].reset_index(drop=True)

#         events_by_region[region] = df

#     events_all = pd.concat(list(events_by_region.values()), ignore_index=True) if events_by_region else pd.DataFrame()
#     return events_by_region, events_all
def detect_swr_by_region(region_lfp: Dict[str, np.ndarray],
                         fs: float,
                         velocity: Optional[np.ndarray] = None,
                         params=None,
                         immobility_mask: Optional[np.ndarray] = None,
                         mua_by_region: Optional[Dict[str, Optional[np.ndarray]]] = None,
                         classify: bool = True,
                         channels: Union[str, List[int]] = "all",
                         average_mode: bool = False,
                         # Optional return of detectors (backward compatible)
                         return_detectors: bool = False,
                         # Optional per-call overrides (fallback to params if None)
                         use_hmm_edge_detection: Optional[bool] = None,
                         adaptive_classification: Optional[bool] = None,
                         dbscan_eps: Optional[float] = None,
                         min_event_separation: Optional[float] = None,
                         merge_interval: Optional[float] = None,
                         detectors_out: Optional[Dict[str, SWRDetector]] = None
                         ) -> Union[
                             Tuple[Dict[str, pd.DataFrame], pd.DataFrame],
                             Tuple[Dict[str, pd.DataFrame], Dict[str, SWRDetector]]
                         ]:
    """
    Run SWR detection per region on multichannel LFP arrays.

    Inputs
    - region_lfp: {region: array [n_chan, n_samples]}
    - fs: sampling frequency (Hz)
    - velocity: optional 1D array aligned to LFP timebase; passed to detector as velocity_data
    - immobility_mask: optional boolean mask (same length as samples) to filter events post-hoc
    - mua_by_region: optional {region: mua array}. Accepts:
        * 1D [n_samples] firing rate or MUA envelope for that region
        * 2D [n_mua_chan, n_samples] multiunit channels (detector-dependent)
      Values can be None for regions without MUA.
    - params: SWRParams or None (uses hippocampal preset if available)

    Returns
    - events_by_region: {region: pd.DataFrame}
    - events_all: concatenated pd.DataFrame
    """
    # Effective params: copy to avoid mutating caller; default to hippocampal preset
    base_params = params or PRESETS.get("hippocampal", SWRParams())
    eff_params = copy.deepcopy(base_params)
    # Apply per-call overrides if provided and supported by the params object
    if use_hmm_edge_detection is not None and hasattr(eff_params, "use_hmm_edge_detection"):
        eff_params.use_hmm_edge_detection = bool(use_hmm_edge_detection)
    if adaptive_classification is not None and hasattr(eff_params, "adaptive_classification"):
        eff_params.adaptive_classification = bool(adaptive_classification)
    if dbscan_eps is not None and hasattr(eff_params, "dbscan_eps"):
        eff_params.dbscan_eps = float(dbscan_eps)
    if min_event_separation is not None and hasattr(eff_params, "min_event_separation"):
        eff_params.min_event_separation = float(min_event_separation)
    if merge_interval is not None and hasattr(eff_params, "merge_interval"):
        eff_params.merge_interval = float(merge_interval)

    # Delegate to the detector's static API
    if return_detectors or (detectors_out is not None):
        events_by_region, events_all, det_map = SWRDetector.detect_by_region(
            region_lfp=region_lfp,
            fs=float(fs),
            velocity=velocity,
            params=eff_params,
            immobility_mask=immobility_mask,
            mua_by_region=mua_by_region,
            classify=classify,
            channels=channels,
            average_mode=average_mode,
            return_detectors=True,
        )
        if detectors_out is not None:
            detectors_out.update(det_map)
        if return_detectors:
            # Return detectors instead of events_all when explicitly requested
            return events_by_region, det_map
        # Backward compatible: if caller didn't request detectors, return all events summary
        return events_by_region, events_all
    else:
        return SWRDetector.detect_by_region(
            region_lfp=region_lfp,
            fs=float(fs),
            velocity=velocity,
            params=eff_params,
            immobility_mask=immobility_mask,
            mua_by_region=mua_by_region,
            classify=classify,
            channels=channels,
            average_mode=average_mode,
            return_detectors=False,
        )


def quick_overlay_plot(loader,
                       region: str,
                       region_channels: Dict[str, List[str]],
                       events_df: pd.DataFrame,
                       start: float = 100.0,
                       dur: float = 5.0,
                       ch_index: int = 0) -> None:
    """
    Plot one channel from a region and overlay event markers in a short window.
    """
    import matplotlib.pyplot as plt

    if region not in region_channels or not region_channels[region]:
        print(f"No channels for region {region}.")
        return

    chs = region_channels[region]
    ch_index = int(np.clip(ch_index, 0, len(chs) - 1))
    ch_name = chs[ch_index]

    x = loader.get_selected_trace(ch_name, start_time=start, end_time=start + dur)
    t_win = loader.time_vector(start, start + dur)

    evt_times = None
    for c in ("t_peak", "peak_time", "center_time", "t_center"):
        if c in events_df.columns:
            evt_times = events_df[c].to_numpy()
            break
    if evt_times is None and {"t_start", "t_end"}.issubset(events_df.columns):
        evt_times = 0.5 * (events_df["t_start"].to_numpy() + events_df["t_end"].to_numpy())
    if evt_times is None and {"start_time", "end_time"}.issubset(events_df.columns):
        evt_times = 0.5 * (events_df["start_time"].to_numpy() + events_df["end_time"].to_numpy())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.plot(t_win, x, lw=0.6, label=ch_name)
    if evt_times is not None:
        mask = (evt_times >= start) & (evt_times <= start + dur)
        for tp in np.asarray(evt_times)[mask]:
            plt.axvline(tp, color="crimson", alpha=0.5, lw=1)
    plt.title(f"{region}: {ch_name} with SWR markers")
    plt.xlabel("Time (s)")
    plt.ylabel("LFP (a.u.)")
    plt.tight_layout()
    plt.show()


def create_region_event_browser(detectors_by_region: Dict[str, SWRDetector],
                                initial_region: Optional[str] = None) -> None:
    """Interactive widget to browse SWR events per region using SWRVisualizer."""
    import ipywidgets as widgets
    from IPython.display import display

    if not detectors_by_region:
        print("No detectors provided.")
        return

    # Lazy import avoids circular dependency at module load
    from pfr_neurofunctions.swr_detection.visualization import SWRVisualizer

    regions = sorted(detectors_by_region.keys())
    region_dropdown = widgets.Dropdown(options=regions,
                                       value=initial_region if initial_region in regions else regions[0],
                                       description='Region:',
                                       disabled=False)
    event_slider = widgets.IntSlider(value=0, min=0, max=0, step=1,
                                     description='Event #',
                                     continuous_update=False,
                                     disabled=True)
    prev_btn = widgets.Button(description='Previous', icon='arrow-left')
    next_btn = widgets.Button(description='Next', icon='arrow-right')
    out = widgets.Output()

    def _current_detector():
        det = detectors_by_region.get(region_dropdown.value)
        if det is None:
            print(f"No detector for region {region_dropdown.value}.")
        return det

    def _refresh_slider():
        det = _current_detector()
        if det is None or not det.swr_events:
            event_slider.disabled = True
            event_slider.max = 0
            event_slider.value = 0
            with out:
                out.clear_output()
                print(f"{region_dropdown.value}: no events to display.")
            return
        event_slider.disabled = False
        event_slider.max = len(det.swr_events) - 1
        event_slider.value = min(event_slider.value, event_slider.max)

    def _render_event(event_index: int) -> None:
        det = _current_detector()
        if det is None or not det.swr_events:
            return
        event_index = int(np.clip(event_index, 0, len(det.swr_events) - 1))
        event_id = det.swr_events[event_index]['event_id']
        viz = SWRVisualizer(det)
        with out:
            out.clear_output(wait=True)
            fig = viz.plot_event_traces(event_id)
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.show()
            event = det.swr_events[event_index]
            print(f"Region: {region_dropdown.value} | Event {event_id} | Channel {event['channel']}\n"
                  f"Start: {event['start_time']:.3f}s  Peak: {event['peak_time']:.3f}s  "
                  f"End: {event['end_time']:.3f}s  Duration: {event['duration']*1000:.1f} ms\n"
                  f"Type: {event['event_type']}  Peak power: {event['peak_power']:.3f}")

    def _on_region_change(change):
        if change['name'] == 'value':
            _refresh_slider()
            if not event_slider.disabled:
                _render_event(event_slider.value)

    def _on_slider_change(change):
        if change['name'] == 'value' and not event_slider.disabled:
            _render_event(change['new'])

    def _step(delta: int):
        if event_slider.disabled:
            return
        new_val = int(np.clip(event_slider.value + delta, event_slider.min, event_slider.max))
        event_slider.value = new_val

    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    region_dropdown.observe(_on_region_change, names='value')
    event_slider.observe(_on_slider_change, names='value')

    controls = widgets.HBox([region_dropdown, prev_btn, event_slider, next_btn])
    display(widgets.VBox([controls, out]))

    _refresh_slider()
    if not event_slider.disabled:
        _render_event(event_slider.value)


def build_spike_times_by_region(
    source,
    t_lfp: np.ndarray,
    region_mapping: Optional[Dict[Union[int, str], str]] = None,
    spike_sampling_rate: Optional[float] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Build a mapping {region: [spike_times_seconds per unit]} from either a SpikeAnalysis-like
    object or a processed units dict. Spike times are clipped to the span of t_lfp and converted
    to seconds if they appear to be in samples.

    Parameters
    - source: SpikeAnalysis-like object (with tetrode_mapping and optional region_mapping) or
              dict with a 'units' list of unit dicts.
    - t_lfp: 1D array of LFP timestamps (seconds) used to clip spike times to session span.
    - region_mapping: Optional explicit {tetrode -> region} mapping.
    - spike_sampling_rate: Sampling rate (Hz) for converting sample indices to seconds when needed.

    Returns
    - dict[str, list[np.ndarray]]
    """
    t_lfp = np.asarray(t_lfp, float).ravel()
    if t_lfp.size == 0:
        return {}
    t0, t1 = float(t_lfp[0]), float(t_lfp[-1])
    rec_span = (t1 - t0)
    sr = float(spike_sampling_rate) if spike_sampling_rate is not None else float(getattr(source, "sampling_rate", 30000.0))

    def _unit_times_any(unit, sa_obj=None) -> np.ndarray:
        arr = None
        if isinstance(unit, dict):
            for key in ("times_sec", "spike_times_sec", "t_sec", "times", "spike_times", "t", "ts",
                        "sample_times", "spike_samples", "times_samples"):
                if key in unit and unit[key] is not None:
                    try:
                        arr = np.asarray(unit[key], dtype=float).ravel()
                        break
                    except Exception:
                        pass
        if arr is None and isinstance(unit, (list, tuple)) and len(unit) == 2:
            try:
                arr = np.asarray(unit[1], dtype=float).ravel()
            except Exception:
                arr = None
        if arr is None and sa_obj is not None:
            for attr in ("get_unit_spike_times", "unit_spike_times", "get_spikes"):
                fn = getattr(sa_obj, attr, None)
                if callable(fn):
                    try:
                        arr = np.asarray(fn(unit), dtype=float).ravel()
                        break
                    except Exception:
                        pass
        return arr if isinstance(arr, np.ndarray) and arr.size else np.array([], dtype=float)

    def _get_tetrode_id(unit) -> Optional[int]:
        if isinstance(unit, dict):
            for tk in ("tetrode", "tet", "group", "tt", "shank", "channel_group"):
                if tk in unit:
                    try:
                        return int(unit[tk])
                    except Exception:
                        pass
        return None

    out: Dict[str, List[np.ndarray]] = {}
    reg_map = region_mapping if region_mapping is not None else getattr(source, "region_mapping", None)
    tet_map = getattr(source, "tetrode_mapping", None)

    # Case A: SpikeAnalysis-like
    if isinstance(tet_map, dict):
        for tet_key, units in tet_map.items():
            rname = None
            if reg_map is not None:
                try:
                    k_int = int(tet_key)
                except Exception:
                    k_int = None
                if k_int is not None and k_int in reg_map:
                    rname = reg_map[k_int]
                elif tet_key in reg_map:
                    rname = reg_map[tet_key]
                elif isinstance(tet_key, str) and tet_key.isdigit() and int(tet_key) in reg_map:
                    rname = reg_map[int(tet_key)]
            if not rname:
                continue
            out.setdefault(rname, [])
            if units is None:
                continue
            try:
                iter(units)
            except TypeError:
                units = [units]
            for u in units:
                st = _unit_times_any(u, sa_obj=source)
                if st.size == 0:
                    continue
                if np.nanmax(st) > rec_span * 1.5:
                    st = st / sr
                st = st[(st >= t0) & (st <= t1)]
                if st.size:
                    out[rname].append(np.sort(st))
        return out

    # Case B: processed units dict
    if isinstance(source, dict):
        for u in source.get("units", []):
            tet = _get_tetrode_id(u)
            if tet is None:
                continue
            rname = None
            if reg_map is not None and tet in reg_map:
                rname = reg_map[tet]
            elif reg_map is not None and str(tet) in reg_map:
                rname = reg_map[str(tet)]
            if not rname:
                continue
            st = _unit_times_any(u, sa_obj=None)
            if st.size == 0:
                continue
            if np.nanmax(st) > rec_span * 1.5:
                st = st / sr
            st = st[(st >= t0) & (st <= t1)]
            if st.size:
                out.setdefault(rname, []).append(np.sort(st))
        return out

    return out


def compute_region_mua_from_spikes(
    spike_times_by_region,
    t_lfp,
    sigma: float = 0.05,
    normalize: str = "sum",
    *,
    region_mapping: Optional[Dict[Union[int, str], str]] = None,
    spike_sampling_rate: Optional[float] = None,
):
    """
    Build per‑region MUA time series aligned to t_lfp by binning spikes to the
    nearest LFP sample and smoothing with a Gaussian kernel.

    This function accepts either:
      1) A pre-built mapping spike_times_by_region: dict[region -> list[np.ndarray of times (s)]]
      2) A SpikeAnalysis-like object with attributes such as 'tetrode_mapping', optionally
         'region_mapping' and 'sampling_rate'. In this case, spike times are extracted and
         grouped by region automatically.
      3) A processed units dictionary (e.g., from load_processed_spike_data) that contains
         a 'units' list where each unit dict has fields like 'tetrode' and 'times' (or similar).

    Parameters
    - spike_times_by_region: dict or object
        Either a dict as in (1) or a 'source' object/dict as in (2)-(3).
    - t_lfp: 1D array of timestamps (seconds) for the LFP timebase.
    - sigma: Gaussian smoothing sigma in seconds (default 0.05 s).
    - normalize: "sum" (sum across units) or "mean" (average across units).
    - region_mapping: Optional explicit mapping {tetrode_id -> region}. If not provided and
        a SpikeAnalysis-like object is passed, attempts to use its internal mapping.
    - spike_sampling_rate: Optional spike sampling rate (Hz) used when spike times appear to be
        in samples. Defaults to 30000 if an explicit rate cannot be resolved from the source.

    Returns
    - dict[str, np.ndarray] mapping region -> 1D vector of length len(t_lfp).
      Regions with no spikes return a zero vector (not None).

    Notes
    - The returned vectors always match len(t_lfp).
    - When given a non-dict source, spike times are converted to seconds if they
      appear to be in samples (heuristic: max(st) > recording_span_seconds * 1.5).
    """
    # -----------------------------
    # Helper: coerce to {region: [times(sec) per unit]}
    # -----------------------------
    def _unit_times_any(unit, sa_obj=None) -> np.ndarray:
        arr = None
        # Try dict-like payloads first
        if isinstance(unit, dict):
            for key in ("times_sec", "spike_times_sec", "t_sec", "times", "spike_times", "t", "ts",
                        "sample_times", "spike_samples", "times_samples"):
                if key in unit and unit[key] is not None:
                    try:
                        arr = np.asarray(unit[key], dtype=float).ravel()
                        break
                    except Exception:
                        pass
        # Tuple/list like (unit_id, times)
        if arr is None and isinstance(unit, (list, tuple)) and len(unit) == 2:
            try:
                arr = np.asarray(unit[1], dtype=float).ravel()
            except Exception:
                arr = None
        # SpikeAnalysis-like accessor fallbacks
        if arr is None and sa_obj is not None:
            for attr in ("get_unit_spike_times", "unit_spike_times", "get_spikes"):
                fn = getattr(sa_obj, attr, None)
                if callable(fn):
                    try:
                        arr = np.asarray(fn(unit), dtype=float).ravel()
                        break
                    except Exception:
                        pass
        return arr if isinstance(arr, np.ndarray) and arr.size else np.array([], dtype=float)

    def _get_tetrode_id(unit) -> Optional[int]:
        if isinstance(unit, dict):
            for tk in ("tetrode", "tet", "group", "tt", "shank", "channel_group"):
                if tk in unit:
                    try:
                        return int(unit[tk])
                    except Exception:
                        pass
        return None

    def _build_spike_dict_from_source(source, t_lfp_arr: np.ndarray) -> Dict[str, List[np.ndarray]]:
        t0, t1 = float(t_lfp_arr[0]), float(t_lfp_arr[-1])
        rec_span = (t1 - t0)
        # Resolve sampling rate for possible samples->seconds conversion
        sr = float(spike_sampling_rate) if spike_sampling_rate is not None else float(getattr(source, "sampling_rate", 30000.0))
        # Resolve tetrode->region mapping
        reg_map = region_mapping
        if reg_map is None:
            reg_map = getattr(source, "region_mapping", None)
        # Build
        out: Dict[str, List[np.ndarray]] = {}

        # Case A: SpikeAnalysis-like with tetrode_mapping
        tet_map = getattr(source, "tetrode_mapping", None)
        if isinstance(tet_map, dict):
            for tet_key, units in tet_map.items():
                # Map tetrode key (could be str or int) to region
                rname = None
                if reg_map is not None:
                    try:
                        k_int = int(tet_key)
                    except Exception:
                        k_int = None
                    if k_int is not None and k_int in reg_map:
                        rname = reg_map[k_int]
                    elif tet_key in reg_map:
                        rname = reg_map[tet_key]
                    elif isinstance(tet_key, str) and tet_key.isdigit() and int(tet_key) in reg_map:
                        rname = reg_map[int(tet_key)]
                # If no mapping known, skip
                if not rname:
                    continue
                out.setdefault(rname, [])
                if units is None:
                    continue
                try:
                    iter(units)
                except TypeError:
                    units = [units]
                for u in units:
                    st = _unit_times_any(u, sa_obj=source)
                    if st.size == 0:
                        continue
                    # Convert samples to seconds if it looks like samples
                    if np.nanmax(st) > rec_span * 1.5:
                        st = st / sr
                    st = st[(st >= t0) & (st <= t1)]
                    if st.size:
                        out[rname].append(np.sort(st))
            return out

        # Case B: processed units dict with 'units' key
        if isinstance(source, dict):
            units_list = source.get("units", [])
            for u in units_list:
                tet = _get_tetrode_id(u)
                if tet is None:
                    continue
                rname = None
                if region_mapping is not None and tet in region_mapping:
                    rname = region_mapping[tet]
                elif region_mapping is not None and str(tet) in region_mapping:
                    rname = region_mapping[str(tet)]
                if not rname:
                    continue
                st = _unit_times_any(u, sa_obj=None)
                if st.size == 0:
                    continue
                if np.nanmax(st) > rec_span * 1.5:
                    st = st / sr
                st = st[(st >= t0) & (st <= t1)]
                if st.size:
                    out.setdefault(rname, []).append(np.sort(st))
            return out

        # Fallback: return empty
        return {}

    # If caller passed a non-dict source, attempt to derive {region: [times]}
    spike_dict: Dict[str, List[np.ndarray]]
    if isinstance(spike_times_by_region, dict):
        spike_dict = spike_times_by_region
    elif spike_times_by_region is None:
        spike_dict = {}
    else:
        spike_dict = _build_spike_dict_from_source(spike_times_by_region, np.asarray(t_lfp, float).ravel())

    # Proceed with the original implementation using spike_dict
    t_lfp = np.asarray(t_lfp, dtype=float).ravel()
    if t_lfp.size < 2:
        raise ValueError("t_lfp must have at least two time points")
    t_lfp = np.asarray(t_lfp, dtype=float).ravel()
    if t_lfp.size < 2:
        raise ValueError("t_lfp must have at least two time points")

    # Sampling step and Gaussian kernel (±5σ)
    dt = float(np.median(np.diff(t_lfp)))
    half = max(1, int(ceil(5.0 * sigma / dt)))
    xk = np.arange(-half, half + 1, dtype=float) * dt
    ker = np.exp(-(xk ** 2) / (2.0 * sigma * sigma))
    ker /= ker.sum()

    n = t_lfp.size
    t0, t1 = t_lfp[0], t_lfp[-1]
    out = {}

    for region, unit_lists in (spike_dict or {}).items():
        unit_lists = unit_lists or []
        unit_traces = []

        for st in unit_lists:
            st = np.asarray(st, dtype=float).ravel()
            if st.size == 0:
                continue
            # Keep spikes within timebase and bin to nearest sample
            st = st[(st >= t0) & (st <= t1)]
            if st.size == 0:
                continue
            idx = np.searchsorted(t_lfp, st, side="left")
            idx = idx[(idx >= 0) & (idx < n)]

            vec = np.zeros(n, dtype=float)
            if idx.size:
                np.add.at(vec, idx, 1.0)
            vec = np.convolve(vec, ker, mode="same")
            unit_traces.append(vec)

        if not unit_traces:
            out[region] = np.zeros(n, dtype=float)
        else:
            stack = np.vstack(unit_traces)  # U x T
            out[region] = stack.mean(axis=0) if normalize == "mean" else stack.sum(axis=0)

    # Ensure caller’s expected keys remain present (zero-filled if missing)
    for r in getattr(spike_dict, "keys", lambda: [])():
        out.setdefault(r, np.zeros(n, dtype=float))

    return out


def summarize_region_df(df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize a single region's events DataFrame.

    Expects columns like: 't_start'/'start_time', 't_end'/'end_time', 'duration', 'peak_power', optional 'type'.
    Returns a dict with counts and basic stats; type counts are prefixed with 'type_count__'.
    """
    if df is None or len(df) == 0:
        return {
            "n_events": 0,
            "mean_duration": np.nan,
            "median_duration": np.nan,
            "mean_peak_power": np.nan,
        }

    out: Dict[str, float] = {"n_events": int(len(df))}

    # Duration stats
    dur_col = None
    for c in ("duration", "dur", "event_duration"):
        if c in df.columns:
            dur_col = c
            break
    if dur_col is not None:
        vals = pd.to_numeric(df[dur_col], errors="coerce")
        out["mean_duration"] = float(np.nanmean(vals))
        out["median_duration"] = float(np.nanmedian(vals))
    else:
        out["mean_duration"] = np.nan
        out["median_duration"] = np.nan

    # Peak power
    if "peak_power" in df.columns:
        vals = pd.to_numeric(df["peak_power"], errors="coerce")
        out["mean_peak_power"] = float(np.nanmean(vals))
    else:
        out["mean_peak_power"] = np.nan

    # Type counts if present
    if "type" in df.columns:
        for k, v in df["type"].value_counts(dropna=False).to_dict().items():
            out[f"type_count__{k}"] = int(v)

    return out


def analyze_events_by_region(events_by_region: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a tidy stats table with one row per region from events_by_region.
    """
    rows = []
    for region, df in (events_by_region or {}).items():
        s = summarize_region_df(df)
        s["region"] = region
        rows.append(s)
    if not rows:
        return pd.DataFrame(columns=["region", "n_events", "mean_duration", "median_duration", "mean_peak_power"])  # empty
    stats_df = pd.DataFrame(rows)
    # Column order: base then any type_count columns
    base = ["region", "n_events", "mean_duration", "median_duration", "mean_peak_power"]
    other = [c for c in stats_df.columns if c not in base]
    stats_df = stats_df[base + sorted(other)]
    return stats_df


def plot_events_by_region_basic(
    stats_df: pd.DataFrame,
    events_by_region: Optional[Dict[str, pd.DataFrame]] = None,
    session_duration: Optional[float] = None,
    figsize: Tuple[float, float] = (13, 8),
    count_ylim: Optional[Tuple[float, float]] = None,
    duration_ylim: Optional[Tuple[float, float]] = None,
    duration_bins: int = 40,
    power_bins: int = 40,
    time_bins: int = 100,
    rate_bin_sec: float = 60.0,
    showfliers: bool = False,
):
    """
    Duration-focused visualization by region:
      - Duration by region (box plot) [primary]
      - Duration distribution (histogram) per region
      - Peak power by region (box plot)
      - Inter‑event interval (IEI) distributions per region
      - Start‑time histogram over the session per region
      - Event rate (events/min) over time per region

    Returns the created matplotlib Figure.
    """
    if stats_df is None or len(stats_df) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    # Sort for consistent display in bars
    df = stats_df.sort_values("n_events", ascending=False).reset_index(drop=True)
    regions = df["region"].astype(str).tolist()

    # Palette
    if sns is not None:
        pal = {r: c for r, c in zip(regions, sns.color_palette("Set2", n_colors=max(3, len(regions))))}
    else:
        base_colors = plt.rcParams.get("axes.prop_cycle", None)
        base_colors = (base_colors.by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])) if base_colors else ["#1f77b4", "#ff7f0e", "#2ca02c"]
        pal = {r: base_colors[i % len(base_colors)] for i, r in enumerate(regions)}

    # Prepare figure: 2 rows x 3 cols
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
    # Top row: duration box, duration histogram, power box
    ax_durbox = fig.add_subplot(gs[0, 0])
    ax_durhist = fig.add_subplot(gs[0, 1])
    ax_powbox = fig.add_subplot(gs[0, 2])
    # Bottom row: IEI, time histogram, rate
    ax_iei = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])
    ax_rate = fig.add_subplot(gs[1, 2])

    # For distributions/time panels we need raw events
    if events_by_region is None or not isinstance(events_by_region, dict) or len(events_by_region) == 0:
        # annotate panels as N/A
        for ax in (ax_durbox, ax_durhist, ax_powbox, ax_iei, ax_time, ax_rate):
            ax.text(0.5, 0.5, "Provide events_by_region for distributions", ha="center", va="center")
            ax.axis("off")
        fig.suptitle("SWR events: distributions require events_by_region", y=0.99, fontsize=12)
        return fig

    # Helper to extract duration vector per region
    def _extract_duration(df_region: pd.DataFrame) -> np.ndarray:
        if df_region is None or len(df_region) == 0:
            return np.array([])
        for cname in ("duration", "dur", "event_duration"):
            if cname in df_region.columns:
                return pd.to_numeric(df_region[cname], errors="coerce").dropna().to_numpy()
        # Try compute from start/end
        if {"start_time", "end_time"}.issubset(df_region.columns):
            st = pd.to_numeric(df_region["start_time"], errors="coerce").to_numpy()
            et = pd.to_numeric(df_region["end_time"], errors="coerce").to_numpy()
            d = (et - st)
            return d[np.isfinite(d)]
        if {"t_start", "t_end"}.issubset(df_region.columns):
            st = pd.to_numeric(df_region["t_start"], errors="coerce").to_numpy()
            et = pd.to_numeric(df_region["t_end"], errors="coerce").to_numpy()
            d = (et - st)
            return d[np.isfinite(d)]
        return np.array([])

    # A) Duration (box) by region
    any_dur = False
    dur_data, dur_labels = [], []
    for r in regions:
        df_r = events_by_region.get(r)
        d = _extract_duration(df_r)
        if d.size == 0:
            continue
        any_dur = True
        dur_data.append(d)
        dur_labels.append(r)
    if any_dur:
        bp = ax_durbox.boxplot(dur_data, labels=dur_labels, patch_artist=True, showfliers=showfliers)
        for patch, r in zip(bp['boxes'], dur_labels):
            patch.set_facecolor(pal[r])
            patch.set_alpha(0.5)
        ax_durbox.set_title("Duration (box) by region")
        ax_durbox.set_xlabel("Region")
        ax_durbox.set_ylabel("Duration (s)")
        if duration_ylim is not None:
            ax_durbox.set_ylim(duration_ylim)
        ax_durbox.tick_params(axis='x', rotation=45)
    else:
        ax_durbox.text(0.5, 0.5, "No duration data", ha="center", va="center")
        ax_durbox.axis("off")

    # A2) Duration histogram per region
    any_dur_hist = False
    for r in regions:
        df_r = events_by_region.get(r)
        d = _extract_duration(df_r)
        if d.size == 0:
            continue
        any_dur_hist = True
        ax_durhist.hist(d, bins=int(max(5, duration_bins)), alpha=0.5, color=pal[r], label=r, density=True)
    ax_durhist.set_title("Duration distribution (hist)")
    ax_durhist.set_xlabel("Duration (s)")
    ax_durhist.set_ylabel("Density")
    if any_dur_hist:
        ax_durhist.legend(fontsize=8, frameon=False)
    else:
        ax_durhist.text(0.5, 0.5, "No duration data", ha="center", va="center")
        ax_durhist.axis("off")

    # B) Peak power (box) by region
    any_power = False
    pow_data, pow_labels = [], []
    for r in regions:
        df_r = events_by_region.get(r)
        if df_r is None or "peak_power" not in df_r.columns:
            continue
        x = pd.to_numeric(df_r["peak_power"], errors="coerce").dropna().values
        if x.size == 0:
            continue
        any_power = True
        pow_data.append(x)
        pow_labels.append(r)
    if any_power:
        bp2 = ax_powbox.boxplot(pow_data, labels=pow_labels, patch_artist=True, showfliers=showfliers)
        for patch, r in zip(bp2['boxes'], pow_labels):
            patch.set_facecolor(pal[r])
            patch.set_alpha(0.5)
        ax_powbox.set_title("Peak ripple power (box) by region")
        ax_powbox.set_xlabel("Region")
        ax_powbox.set_ylabel("Peak power (a.u.)")
        ax_powbox.tick_params(axis='x', rotation=45)
    else:
        ax_powbox.text(0.5, 0.5, "No peak_power data", ha="center", va="center")
        ax_powbox.axis("off")

    # D) Inter-event interval (IEI)
    any_iei = False
    for r in regions:
        df_r = events_by_region.get(r)
        if df_r is None or "start_time" not in df_r.columns:
            continue
        t = pd.to_numeric(df_r["start_time"], errors="coerce").dropna().values
        t = np.sort(t)
        if t.size < 2:
            continue
        iei = np.diff(t)
        if iei.size == 0:
            continue
        any_iei = True
        ax_iei.hist(iei, bins=40, alpha=0.5, color=pal[r], label=r, density=True)
    ax_iei.set_title("Inter-event intervals")
    ax_iei.set_xlabel("Interval (s)")
    ax_iei.set_ylabel("Density")
    if any_iei:
        ax_iei.legend(fontsize=8, frameon=False)
    else:
        ax_iei.text(0.5, 0.5, "Insufficient events for IEI", ha="center", va="center")

    # Session duration inference
    if session_duration is None:
        # Try infer from events
        max_end = 0.0
        for r in regions:
            df_r = events_by_region.get(r)
            if df_r is None:
                continue
            if "end_time" in df_r.columns:
                v = pd.to_numeric(df_r["end_time"], errors="coerce").dropna()
                if len(v):
                    max_end = max(max_end, float(v.max()))
            elif "start_time" in df_r.columns:
                v = pd.to_numeric(df_r["start_time"], errors="coerce").dropna()
                if len(v):
                    max_end = max(max_end, float(v.max()))
        session_duration = max_end

    # E) Start-time histogram over session
    any_time = False
    edges = np.linspace(0, max(session_duration or 0.0, 1e-6), int(time_bins) + 1)
    for r in regions:
        df_r = events_by_region.get(r)
        if df_r is None or "start_time" not in df_r.columns:
            continue
        t = pd.to_numeric(df_r["start_time"], errors="coerce").dropna().values
        if t.size == 0:
            continue
        h, e = np.histogram(t, bins=edges)
        centers = 0.5 * (e[:-1] + e[1:])
        any_time = True
        ax_time.plot(centers, h, color=pal[r], lw=1.2, label=r)
    ax_time.set_title("Event counts over session")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Count/bin")
    if any_time:
        ax_time.legend(fontsize=8, frameon=False)
    else:
        ax_time.text(0.5, 0.5, "No start_time data", ha="center", va="center")

    # F) Event rate (events/min)
    any_rate = False
    win = float(max(1.0, rate_bin_sec))
    edges_r = np.arange(0, max(session_duration or 0.0, 1e-6) + win, win)
    for r in regions:
        df_r = events_by_region.get(r)
        if df_r is None or "start_time" not in df_r.columns:
            continue
        t = pd.to_numeric(df_r["start_time"], errors="coerce").dropna().values
        if t.size == 0:
            continue
        h, e = np.histogram(t, bins=edges_r)
        rate_per_min = (h / win) * 60.0
        centers = 0.5 * (e[:-1] + e[1:])
        any_rate = True
        ax_rate.plot(centers, rate_per_min, color=pal[r], lw=1.4, label=r)
    ax_rate.set_title(f"Event rate (per {int(win)}s window)")
    ax_rate.set_xlabel("Time (s)")
    ax_rate.set_ylabel("Events/min")
    if any_rate:
        ax_rate.legend(fontsize=8, frameon=False)
    else:
        ax_rate.text(0.5, 0.5, "No start_time data", ha="center", va="center")

    fig.suptitle("SWR events: duration (box+hist), power (box), IEI, time hist, rate", y=0.99, fontsize=12)
    return fig
