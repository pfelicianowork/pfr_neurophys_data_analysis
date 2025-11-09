import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional

def save_swr_events(events: List[Dict[str, Any]], base_path: str, include_traces: bool = True, include_spectrogram: bool = True):
    """
    Save SWR events to CSV and NPZ files.
    Args:
        events: List of event dicts.
        base_path: Path prefix for output files (no extension).
        include_traces: If True, save trace arrays.
        include_spectrogram: If True, save spectrogram arrays.
    """
    array_fields = [
        'raw_trace', 'ripple_trace', 'mua_trace', 'ripple_power', 'sharpwave_trace',
        'trace_timestamps'
    ]
    spectrogram_fields = ['spectrogram', 'spectrogram_freqs', 'spectrogram_times']
    scalar_events = []
    arrays = {}
    for i, event in enumerate(events):
        scalar = {k: v for k, v in event.items() if k not in array_fields + spectrogram_fields}
        scalar['event_idx'] = i
        scalar_events.append(scalar)
        if include_traces:
            for field in array_fields:
                if field in event and event[field] is not None:
                    arrays[f'{field}_{i}'] = np.array(event[field])
        if include_spectrogram:
            for field in spectrogram_fields:
                if field in event and event[field] is not None:
                    arrays[f'{field}_{i}'] = np.array(event[field])
    # Save scalar metadata
    pd.DataFrame(scalar_events).to_csv(base_path + '_metadata.csv', index=False)
    # Save arrays
    if arrays:
        np.savez_compressed(base_path + '_arrays.npz', **arrays)


def load_swr_events(base_path: str, include_traces: bool = True, include_spectrogram: bool = True) -> List[Dict[str, Any]]:
    """
    Load SWR events from CSV and NPZ files.
    Args:
        base_path: Path prefix for input files (no extension).
        include_traces: If True, load trace arrays.
        include_spectrogram: If True, load spectrogram arrays.
    Returns:
        List of event dicts.
    """
    array_fields = [
        'raw_trace', 'ripple_trace', 'mua_trace', 'ripple_power', 'sharpwave_trace',
        'trace_timestamps'
    ]
    spectrogram_fields = ['spectrogram', 'spectrogram_freqs', 'spectrogram_times']
    events_df = pd.read_csv(base_path + '_metadata.csv')
    events = [{str(k): v for k, v in event.items()} for event in events_df.to_dict('records')]
    arrays = None
    arrays_path = base_path + '_arrays.npz'
    if (include_traces or include_spectrogram) and os.path.exists(arrays_path):
        arrays = np.load(arrays_path, allow_pickle=True)
    for event in events:
        idx = event['event_idx']
        if arrays is not None:
            if include_traces:
                for field in array_fields:
                    key = f'{field}_{idx}'
                    if key in arrays:
                        event[field] = arrays[key]
            if include_spectrogram:
                for field in spectrogram_fields:
                    key = f'{field}_{idx}'
                    if key in arrays:
                        event[field] = arrays[key]
    return events
