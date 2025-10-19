import os
import json
import math
import numpy as np
from typing import Dict, List, Optional, Union
from scipy.signal import resample_poly


def _infer_duration_sec(dat_path: str, num_channels: int, fs_in: float, dtype) -> float:
    bytes_per_sample = np.dtype(dtype).itemsize
    file_bytes = os.path.getsize(dat_path)
    total_samples_allch = file_bytes // bytes_per_sample
    total_frames = total_samples_allch // num_channels
    return total_frames / fs_in


"""Fast LFP downsampling cache utilities."""


def _downsample_one_channel(
    dat_path: str,
    num_channels: int,
    channel_index: int,
    fs_in: float,
    fs_out: float,
    duration_sec: float,
    dtype,
    out_path: str,
    chunk_sec: float,
    overlap_sec: float,
) -> int:
    # Input memmap
    x_mm = np.memmap(dat_path, dtype=dtype, mode="r")

    # Output memmap (pre-size exactly)
    total_frames = int(round(duration_sec * fs_in))
    total_out = int(round(duration_sec * fs_out))
    # Write as a valid .npy using NumPy's open_memmap so np.load(mmap_mode='r') works
    y_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(total_out,))

    down = fs_in / fs_out
    up_i = 1
    down_i = int(round(down))
    if not math.isclose(down, down_i, rel_tol=0, abs_tol=1e-9):
        # Non-integer ratio; rational approx if needed
        up_i = 1000
        down_i = int(round(down * up_i))

    write_cursor = 0
    t0 = 0.0
    while t0 < duration_sec:
        t1 = min(t0 + chunk_sec, duration_sec)

        # Padding for FIR filter transient
        pad0 = max(0.0, t0 - overlap_sec)
        pad1 = min(duration_sec, t1 + overlap_sec)

        f0 = int(round(pad0 * fs_in))
        f1 = int(round(pad1 * fs_in))
        n_frames = f1 - f0

        # Read contiguous block across all channels, then pick the channel (avoid strided I/O)
        block_flat = x_mm[f0 * num_channels : f1 * num_channels]
        block = np.asarray(block_flat, dtype=dtype).reshape(n_frames, num_channels)
        chan = block[:, channel_index].astype(np.float32, copy=False)

        # Polyphase decimation
        ds = resample_poly(chan, up=up_i, down=down_i)

        # Trim padding in decimated domain
        cut0 = int(round((t0 - pad0) * fs_out))
        n_out = int(round((t1 - t0) * fs_out))
        seg = ds[cut0 : cut0 + n_out].astype(np.float32, copy=False)

        y_mm[write_cursor : write_cursor + seg.size] = seg
        write_cursor += seg.size
        t0 = t1

    y_mm.flush()
    return total_out


class CachedLFPLoader:
    """Lazy access to cached LFP channels stored as .npy float32 arrays."""

    def __init__(self, meta_path: str):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.meta = meta
        self.fs = float(meta["fs"])  # output fs
        self.duration = float(meta["duration_sec"])  # seconds
        self.channel_files: Dict[str, str] = meta["channel_files"]
        self._mmaps: Dict[str, np.memmap] = {}

    @property
    def sampling_frequency(self) -> float:
        return self.fs

    @property
    def selected_channels(self) -> Dict[str, str]:
        return self.channel_files

    def _ensure_mmap(self, name: str) -> np.memmap:
        if name not in self.channel_files:
            raise KeyError(f"Unknown channel name: {name}")
        if name not in self._mmaps:
            path = self.channel_files[name]
            try:
                # Preferred: load .npy with header
                self._mmaps[name] = np.load(path, mmap_mode="r")
            except Exception:
                # Fallback: legacy cache written as raw float32 binary without .npy header
                # Infer expected length from metadata
                expected_len = int(round(self.duration * self.fs))
                file_bytes = os.path.getsize(path)
                dtype = np.float32
                bytes_per_sample = np.dtype(dtype).itemsize
                if file_bytes // bytes_per_sample < expected_len:
                    raise ValueError(
                        f"Cached file {path} is too small or invalid. Delete the cache directory and rebuild."
                    )
                self._mmaps[name] = np.memmap(path, dtype=dtype, mode="r", shape=(expected_len,))
        return self._mmaps[name]

    def get_selected_trace(
        self,
        name_or_names: Union[str, List[str]],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        if isinstance(name_or_names, str):
            names = [name_or_names]
            single = True
        else:
            names = list(name_or_names)
            single = False

        fs = self.fs
        T = self.duration
        t0 = 0.0 if start_time is None else max(0.0, float(start_time))
        t1 = T if end_time is None else min(T, float(end_time))
        i0 = int(round(t0 * fs))
        i1 = int(round(t1 * fs))

        chans = []
        for nm in names:
            x = self._ensure_mmap(nm)
            chans.append(x[i0:i1])
        out = chans[0] if single else np.stack(chans, axis=0)
        return out

    def time_vector(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        fs = self.fs
        T = self.duration
        t0 = 0.0 if start_time is None else max(0.0, float(start_time))
        t1 = T if end_time is None else min(T, float(end_time))
        n = int(round((t1 - t0) * fs))
        return t0 + np.arange(n, dtype=np.float64) / fs


def build_cached_lfp(
    dat_path: str,
    num_channels: int,
    selected_channels: Dict[str, int],  # {'CA1_tet1': 17, ...}
    fs_in: float,
    fs_out: float = 1000.0,
    out_dir: Optional[str] = None,
    dtype=np.int16,
    chunk_sec: float = 120.0,
    overlap_sec: float = 0.25,
) -> str:
    """
    Downsample and cache selected channels to disk as .npy (float32) + metadata JSON.
    Returns the metadata file path.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
    os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 8))

    if out_dir is None:
        base, _ = os.path.splitext(dat_path)
        out_dir = base + f"_lfp_ds{int(fs_out)}"
    os.makedirs(out_dir, exist_ok=True)

    duration_sec = _infer_duration_sec(dat_path, num_channels, fs_in, dtype)
    channel_files: Dict[str, str] = {}

    for ch_name, ch_idx in selected_channels.items():
        out_path = os.path.join(out_dir, f"{ch_name}.npy")
        if not os.path.exists(out_path):
            _downsample_one_channel(
                dat_path=dat_path,
                num_channels=num_channels,
                channel_index=int(ch_idx),
                fs_in=fs_in,
                fs_out=fs_out,
                duration_sec=duration_sec,
                dtype=dtype,
                out_path=out_path,
                chunk_sec=chunk_sec,
                overlap_sec=overlap_sec,
            )
        channel_files[ch_name] = out_path

    meta = {
        "source_dat": os.path.abspath(dat_path),
        "num_channels": int(num_channels),
        "fs_in": float(fs_in),
        "fs": float(fs_out),
        "duration_sec": float(duration_sec),
        "dtype_source": np.dtype(dtype).name,
        "channel_files": channel_files,
    }
    meta_path = os.path.join(out_dir, "lfp_cache_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def open_cached_lfp(meta_path: str) -> CachedLFPLoader:
    return CachedLFPLoader(meta_path)


def fast_openephys_dat_lfp(
    filepath: str,
    num_channels: int,
    tetrode_groups: Dict[str, Dict[str, list]],
    selected_channels: Dict[str, int],
    sampling_frequency: float = 30000.0,
    target_sampling_frequency: float = 1000.0,
    cache_dir: Optional[str] = None,
    dtype: str = "int16",
    chunk_sec: float = 120.0,
    overlap_sec: float = 0.25,
    return_mode: str = "loader",  # 'loader' | 'stack' | 'dict'
    return_time: bool = False,
):
    """
    One-pass fast downsample of selected channels from .dat to target fs and cache to disk.
    Returns a CachedLFPLoader (default) or materialized arrays.
    """
    dtype_np = np.dtype(dtype)
    meta_path = build_cached_lfp(
        dat_path=filepath,
        num_channels=num_channels,
        selected_channels=selected_channels,
        fs_in=float(sampling_frequency),
        fs_out=float(target_sampling_frequency),
        out_dir=cache_dir,
        dtype=dtype_np,
        chunk_sec=float(chunk_sec),
        overlap_sec=float(overlap_sec),
    )
    loader = CachedLFPLoader(meta_path)

    if return_mode == "loader":
        return loader if not return_time else (loader, loader.time_vector(0, loader.duration))

    # Materialize arrays
    names = list(selected_channels.keys())
    if return_mode == "stack":
        traces = [loader.get_selected_trace(nm, 0, loader.duration) for nm in names]
        stack = np.stack(traces, axis=0)
        return (stack, loader.time_vector(0, loader.duration)) if return_time else stack
    elif return_mode == "dict":
        out = {nm: loader.get_selected_trace(nm, 0, loader.duration) for nm in names}
        return (out, loader.time_vector(0, loader.duration)) if return_time else out
    else:
        raise ValueError("return_mode must be 'loader', 'stack', or 'dict'")
