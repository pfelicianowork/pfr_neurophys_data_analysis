import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

def get_cluster_average_waveform(npy_path: str, cluster_id: int) -> np.ndarray:
    """
    Returns the average waveform for a given cluster.

    Parameters:
    -----------
    npy_path : str
        Path to directory containing Kilosort output files
    cluster_id : int
        Cluster ID to extract waveform for

    Returns:
    --------
    avg_waveform : np.ndarray
        Average waveform for the cluster (channels x samples)
    """
    npy_dir = Path(npy_path)
    spike_clusters = np.load(npy_dir / 'spike_clusters.npy')
    spike_templates = np.load(npy_dir / 'spike_templates.npy')
    templates = np.load(npy_dir / 'templates.npy')  # shape: (n_templates, n_channels, n_samples)

    # Find spikes belonging to the cluster
    cluster_spike_indices = np.where(spike_clusters == cluster_id)[0]
    if len(cluster_spike_indices) == 0:
        logging.warning(f"No spikes found for cluster {cluster_id}")
        raise ValueError(f"No spikes found for cluster {cluster_id}")
    # Get template indices for these spikes
    cluster_template_indices = spike_templates[cluster_spike_indices]
    # Get unique template indices (usually one per cluster, but could be more)
    unique_templates, counts = np.unique(cluster_template_indices, return_counts=True)
    # Use the most common template for this cluster
    main_template_idx = unique_templates[np.argmax(counts)]
    avg_waveform = templates[main_template_idx]  # shape: (n_channels, n_samples)
    return avg_waveform

def _save_units_consistently(units_list, save_path: str) -> None:
    """
    Save units in a consistent, reloadable way:
    - units.npy: 1-D object array so reloading is predictable
    - units.pkl: Python list (optional convenience)
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure list
    if not isinstance(units_list, list):
        units_list = list(units_list)

    # Save as 1-D object array (avoids 0-D ndarray on load)
    arr = np.empty(len(units_list), dtype=object)
    arr[:] = units_list
    np.save(save_dir / 'units.npy', arr, allow_pickle=True)

    # Optional: pickle as direct Python list
    with open(save_dir / 'units.pkl', 'wb') as f:
        pickle.dump(units_list, f)


def load_processed_spike_data(units_file: str):
    """
    Robust loader that returns a Python list of unit dicts from units.npy or units.pkl.
    - For .npy: handles 0-D and 1-D object arrays.
    - For .pkl: returns the pickled list directly.
    """
    path = Path(units_file)
    if path.suffix.lower() == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)

    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:  # 0-D object saved previously
            obj = obj.item()
            return obj if isinstance(obj, list) else list(obj)
        # (N,) object array -> list of dicts
        if obj.dtype == object:
            return list(obj)
        return list(obj)
    # Fallback
    return list(obj)


def process_spike_data(npy_path: str, save_path: str, samp_freq: int = 30000) -> list:
    """
    Process spike sorting data from Kilosort output files.

    Parameters:
    -----------
    npy_path : str
        Path to directory containing Kilosort output files
    save_path : str
        Path where processed data should be saved
    samp_freq : int, optional
        Sampling frequency in Hz (default: 30000 for Open Ephys)

    Returns:
    --------
    processed_clusters : list
        List of dictionaries containing processed cluster data
    """
    samp_rate = 1 / samp_freq
    npy_dir = Path(npy_path)
    save_dir = Path(save_path)

    spike_times_path = npy_dir / 'spike_times.npy'
    spike_clusters_path = npy_dir / 'spike_clusters.npy'
    cluster_info_path = npy_dir / 'cluster_info.tsv'

    try:
        spike_times = np.load(spike_times_path)
        spike_clusters = np.load(spike_clusters_path)
        cluster_info = pd.read_csv(cluster_info_path, sep='\t')
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
        raise

    # Filter out noise clusters
    if 'group' not in cluster_info.columns:
        logging.warning("Column 'group' not found in cluster_info.tsv; assuming all clusters are good")
        noise_clusters = np.array([])
    else:
        noise_clusters = np.array(cluster_info[cluster_info['group'] == 'noise']['cluster_id'].values)

    good_spike_mask = ~np.isin(spike_clusters, noise_clusters)
    spike_times = spike_times[good_spike_mask]
    spike_clusters = spike_clusters[good_spike_mask]
    cluster_info = cluster_info[cluster_info['group'] != 'noise'] if 'group' in cluster_info.columns else cluster_info

    if len(cluster_info) == 0:
        logging.warning("No good clusters found after filtering noise")
        processed_clusters = []
    else:
        processed_clusters = []
        for cluster_id in cluster_info['cluster_id'].values:
            cluster_mask = spike_clusters == cluster_id
            if np.sum(cluster_mask) == 0:
                logging.warning(f"No spikes found for cluster {cluster_id} after filtering")
                continue
            try:
                cluster_data = {
                    'cluster_id': cluster_id,
                    'spk_samples': spike_times[cluster_mask],
                    'spk_time': spike_times[cluster_mask] * samp_rate,
                    'channel': cluster_info[cluster_info['cluster_id'] == cluster_id]['depth'].values[0],
                    'ch': cluster_info[cluster_info['cluster_id'] == cluster_id]['ch'].values[0],
                    'channel_group': cluster_info[cluster_info['cluster_id'] == cluster_id]['channel_group'].values[0]
                }
                processed_clusters.append(cluster_data)
            except (KeyError, IndexError) as e:
                logging.warning(f"Missing metadata for cluster {cluster_id}: {e}")
                continue

    # Save processed data consistently (both .npy object array and .pkl list)
    _save_units_consistently(processed_clusters, save_dir)
    logging.info(f"Processed data saved to {save_dir / 'units.npy'} and {save_dir / 'units.pkl'}")
    return processed_clusters