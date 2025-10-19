"""
utils.py - Utility functions for Open Ephys data handling

This module contains utility functions that support data loading and processing.
"""

import numpy as np
from pathlib import Path


def validate_directory(directory):
    """
    Validate that a directory exists and is accessible.
    
    Parameters
    ----------
    directory : str or Path
        Directory path to validate
        
    Returns
    -------
    directory : Path
        Validated directory as Path object
        
    Raises
    ------
    ValueError
        If directory doesn't exist or is not accessible
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    return directory


def format_time(seconds):
    """
    Convert seconds to a human-readable time format.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    time_str : str
        Formatted time string (e.g., "1h 23m 45.6s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"


def samples_to_time(samples, sample_rate):
    """
    Convert sample indices to time in seconds.
    
    Parameters
    ----------
    samples : int or numpy.ndarray
        Sample index or array of sample indices
    sample_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    time : float or numpy.ndarray
        Time in seconds
    """
    return samples / sample_rate


def time_to_samples(time, sample_rate):
    """
    Convert time in seconds to sample indices.
    
    Parameters
    ----------
    time : float or numpy.ndarray
        Time in seconds
    sample_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    samples : int or numpy.ndarray
        Sample index or array of sample indices
    """
    return np.round(time * sample_rate).astype(int)


def get_channel_info(data):
    """
    Extract channel information from loaded data.
    
    Parameters
    ----------
    data : dict or OpenEphysRecording
        Data loaded by load_open_ephys_data()
        
    Returns
    -------
    info : dict
        Dictionary containing:
        - 'num_channels': number of channels
        - 'channel_names': list of channel names
        - 'sample_rate': sampling rate
    """
    info = {}
    
    if isinstance(data, dict):
        # Old format
        info['num_channels'] = len(data)
        info['channel_names'] = list(data.keys())
        
        # Get sample rate from first channel
        first_channel = next(iter(data.values()))
        if 'header' in first_channel:
            info['sample_rate'] = float(first_channel['header'].get('sampleRate', 30000.0))
        else:
            info['sample_rate'] = 30000.0
    else:
        # New format - would need to load continuous first
        try:
            if not hasattr(data, '_continuous') or data._continuous is None:
                data.load_continuous()
            
            if data.continuous:
                stream = data.continuous[0]
                info['num_channels'] = stream.metadata['num_channels']
                info['channel_names'] = stream.metadata['channel_names']
                info['sample_rate'] = stream.metadata['sample_rate']
            else:
                info['num_channels'] = 0
                info['channel_names'] = []
                info['sample_rate'] = 30000.0
        except Exception as e:
            print(f"Warning: Could not extract channel info: {e}")
            info['num_channels'] = 0
            info['channel_names'] = []
            info['sample_rate'] = 30000.0
    
    return info


def create_time_vector(num_samples, sample_rate):
    """
    Create a time vector for data visualization.
    
    Parameters
    ----------
    num_samples : int
        Number of samples
    sample_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    time_vector : numpy.ndarray
        Time vector in seconds
    """
    return np.arange(num_samples) / sample_rate
