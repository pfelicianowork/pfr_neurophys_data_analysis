import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# Backward-compatible import for gaussian window function
try:
    from scipy.signal.windows import gaussian
except ImportError:
    from scipy.signal import gaussian

class SpikeAnalysis:
    """
    A class for analyzing spike data from neural recordings.

    This class provides methods to compute firing rates, analyze spike data across units or tetrodes,
    and visualize results. It also supports brain region assignments for regional statistics and plotting.

    Attributes:
        processed_data (list): List of unit dictionaries with spike data.
        sampling_rate (int): Sampling rate of the data (default: 30000 Hz).
        total_samples (int): Total number of samples (optional).
        duration (float): Duration of the recording in seconds (optional, required for some methods).
        unit_mapping (dict): Mapping of cluster_id to unit data.
        tetrode_mapping (dict): Mapping of channel_group to list of units.
        region_mapping (dict): Mapping of tetrode to brain region (set via assign_brain_regions).
        region_to_tetrodes (dict): Mapping of region to list of tetrodes (set via assign_brain_regions).
    """

    def __init__(self, processed_data, sampling_rate=30000, total_samples=None, duration=None):
        if not isinstance(processed_data, list) or not all(isinstance(u, dict) and 'cluster_id' in u and 'spk_time' in u and 'channel_group' in u for u in processed_data):
            raise ValueError("processed_data must be a list of dicts with 'cluster_id', 'spk_time', and 'channel_group' keys.")
        self.processed_data = processed_data
        self.sampling_rate = sampling_rate
        self.total_samples = total_samples
        self.duration = duration
        self.unit_mapping = {unit['cluster_id']: unit for unit in processed_data}
        self.tetrode_mapping = self._create_tetrode_mapping()

    def _create_tetrode_mapping(self):
        """Create a mapping from channel_group (tetrode) to list of units."""
        mapping = defaultdict(list)
        for unit in self.processed_data:
            mapping[unit['channel_group']].append(unit)
        return mapping

    def _get_time_bins(self, analysis_sampling_rate=1000, time_range=None):
        """Generate time bins for analysis based on duration and sampling rate."""
        if self.duration is None:
            raise ValueError("Duration must be set for time bin calculation.")
        start_time = 0 if time_range is None else max(0, time_range[0])
        end_time = self.duration if time_range is None else min(self.duration, time_range[1])
        dt = 1 / analysis_sampling_rate
        num_bins = int(np.ceil((end_time - start_time) * analysis_sampling_rate))
        bins = np.linspace(start_time, end_time, num_bins + 1)
        times = (bins[:-1] + bins[1:]) / 2
        return times, bins

    def get_firing_rate(self, spike_times, bins, kernel_width=0.05, sampling_rate=1000):
        """Compute smoothed firing rates from spike times using Gaussian kernel."""
        hist, _ = np.histogram(spike_times, bins=bins)
        kernel_bins = int(kernel_width * sampling_rate)
        if kernel_bins % 2 == 0:
            kernel_bins += 1
        sigma = kernel_bins / (2 * np.sqrt(2 * np.log(2)))
        rates = gaussian_filter1d(hist.astype(float), sigma)
        rates = rates * sampling_rate
        return rates

    def analyze_data(self, identifiers, id_type='unit', kernel_width=0.05, sampling_rate=1000,
                    time_range=None, full_recording=False):
        """Analyze firing rates for specified units or tetrodes."""
        if not isinstance(identifiers, (list, tuple, np.ndarray)):
            identifiers = [identifiers]
        results = {
            'window': {'individual': {}, 'average': None, 'std': None, 'times': None},
            'full': {'individual': {}, 'average': None, 'std': None, 'times': None}
        }
        units_to_analyze = []
        if id_type == 'unit':
            units_to_analyze = [self.unit_mapping[id] for id in identifiers if id in self.unit_mapping]
        elif id_type in ['tetrode', 'multi_tetrode']:
            for tet in identifiers:
                if tet in self.tetrode_mapping:
                    units_to_analyze.extend(self.tetrode_mapping[tet])
        if not units_to_analyze:
            return results
        time_ranges = [time_range]
        if full_recording:
            time_ranges.append((0, self.duration))
        for t_range in time_ranges:
            key = 'window' if t_range == time_range else 'full'
            times, bins = self._get_time_bins(sampling_rate, t_range)
            rates_list = []
            for unit in units_to_analyze:
                rates = self.get_firing_rate(unit['spk_time'], bins, kernel_width, sampling_rate)
                rates_list.append(rates)
            if rates_list:
                rates = np.array(rates_list)
                avg_rate = np.mean(rates, axis=0)
                std_rate = np.std(rates, axis=0)
                for i, unit in enumerate(units_to_analyze):
                    results[key]['individual'][unit['cluster_id']] = {
                        'times': times,
                        'rates': rates[i],
                        'spikes': unit['spk_time'],
                        'channel_group': unit['channel_group']
                    }
                results[key]['average'] = avg_rate
                results[key]['std'] = std_rate
                results[key]['times'] = times
        return results

    def plot_analysis(self, results, plot_type='both', figsize=(15, 5)):
        """Plot spike rasters and firing rates for analysis results."""
        if plot_type not in ['window', 'full', 'both']:
            raise ValueError("plot_type must be 'window', 'full', or 'both'")
        n_plots = 2 if plot_type == 'both' else 1
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, n_plots, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        def plot_results(data, col, title_suffix):
            if not data['individual']:
                return
            times = data['times']
            start_time, end_time = times[0], times[-1]
            ax_raster = fig.add_subplot(gs[0, col])
            unit_count = 0
            tetrode_boundaries = []
            tetrode_labels = []
            current_tetrode = None
            sorted_units = sorted(data['individual'].items(), key=lambda x: x[1]['channel_group'])
            tetrode_start = 0
            for cluster_id, unit_data in sorted_units:
                spikes = unit_data['spikes']
                spikes_in_window = spikes[(spikes >= start_time) & (spikes <= end_time)]
                tetrode = unit_data['channel_group']
                if current_tetrode != tetrode:
                    if current_tetrode is not None:
                        tetrode_boundaries.append((tetrode_start + unit_count - 1) / 2)
                        tetrode_labels.append(f'T{current_tetrode}')
                    current_tetrode = tetrode
                    tetrode_start = unit_count
                ax_raster.plot(spikes_in_window, np.ones_like(spikes_in_window) * unit_count, '|',
                               markersize=4, label=f'Unit {cluster_id}')
                unit_count += 1
            if current_tetrode is not None:
                tetrode_boundaries.append((tetrode_start + unit_count - 1) / 2)
                tetrode_labels.append(f'T{current_tetrode}')
            ax_raster.set_yticks(tetrode_boundaries)
            ax_raster.set_yticklabels(tetrode_labels)
            ax_raster.set_title(f'Spike Raster Plot {title_suffix}')
            ax_raster.set_xlabel('Time (s)')
            ax_raster.set_xlim(start_time, end_time)
            ax_rates = fig.add_subplot(gs[1, col])
            for unit_data in data['individual'].values():
                ax_rates.plot(unit_data['times'], unit_data['rates'], alpha=0.2)
            if data['average'] is not None:
                ax_rates.plot(times, data['average'], color='black', linewidth=2, label='Average')
                ax_rates.fill_between(times, data['average'] - data['std'], data['average'] + data['std'],
                                     color='gray', alpha=0.2)
            ax_rates.set_xlabel('Time (s)')
            ax_rates.set_ylabel('Firing Rate (Hz)')
            ax_rates.set_title(f'Firing Rates {title_suffix}')
            ax_rates.legend()
            ax_rates.grid(True, alpha=0.3)
            ax_rates.set_xlim(start_time, end_time)
        if plot_type in ['window', 'both']:
            plot_results(results['window'], 0, '(Window)')
        if plot_type in ['full', 'both']:
            plot_results(results['full'], 1 if plot_type == 'both' else 0, '(Full Recording)')
        plt.show()

    def assign_brain_regions(self, region_mapping):
        """Assign brain regions to tetrodes for regional analysis."""
        self.region_mapping = region_mapping
        self.region_to_tetrodes = defaultdict(list)
        for tetrode, region in region_mapping.items():
            self.region_to_tetrodes[region].append(tetrode)

    def get_region_statistics(self, region):
        """Get statistics for units in a specific brain region."""
        if not hasattr(self, 'region_mapping'):
            return {'error': "Brain regions haven't been assigned. Use assign_brain_regions first.", 'region': region}
        tetrodes = self.region_to_tetrodes.get(region, [])
        if not tetrodes:
            return {'error': f'No tetrodes found in region {region}', 'region': region}
        units = []
        for tetrode in tetrodes:
            units.extend(self.tetrode_mapping[tetrode])
        if not units:
            return {'error': f'No units found in region {region}', 'region': region}
        n_units = len(units)
        total_spikes = sum(len(unit['spk_time']) for unit in units)
        avg_firing_rates = []
        for unit in units:
            spk_times = unit['spk_time']
            if len(spk_times) < 2:
                continue  # Skip units with insufficient data
            duration = spk_times[-1] - spk_times[0]
            if duration > 0:
                avg_firing_rates.append(len(spk_times) / duration)
        if not avg_firing_rates:
            return {'error': f'No valid units in region {region}', 'region': region}
        stats = {
            'region': region,
            'n_tetrodes': len(tetrodes),
            'n_units': n_units,
            'total_spikes': total_spikes,
            'mean_firing_rate': np.mean(avg_firing_rates),
            'std_firing_rate': np.std(avg_firing_rates),
            'tetrodes': tetrodes
        }
        return stats

    def plot_multiple_brain_regions(
        self, regions, time_window=None, figsize=(12, 6), region_colors=None,
        markersize=4, linewidth=1, show_grid=True, legend_fontsize=10, legend_loc='upper left', y_label_format="Unit Index"
    ):
        """Plot spike activity across multiple brain regions."""
        if not hasattr(self, 'region_mapping'):
            raise ValueError("Brain regions haven't been assigned. Use assign_brain_regions first.")
        if region_colors is None:
            region_colors = {'CA1': 'blue', 'PFC': 'red', 'DG': 'green', 'Thalamus': 'purple'}
        all_units = []
        unit_regions = []
        for region in regions:
            tetrodes = self.region_to_tetrodes.get(region, [])
            for tetrode in tetrodes:
                units = self.tetrode_mapping[tetrode]
                all_units.extend(units)
                unit_regions.extend([region] * len(units))
        if not all_units:
            print(f"No units found in regions {regions}")
            return
        fig, ax = plt.subplots(figsize=figsize)
        y_offset = 0
        y_labels = []
        for i, (unit, region) in enumerate(zip(all_units, unit_regions)):
            spike_times = unit['spk_time']
            if time_window is not None:
                mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
                spike_times = spike_times[mask]
            ax.plot(spike_times, np.ones_like(spike_times) * y_offset, '|', markersize=markersize,
                    linewidth=linewidth, color=region_colors.get(region, 'gray'),
                    label=f"{region} Unit {unit['cluster_id']}")
            if y_label_format == "Region_Unit":
                y_labels.append(f"{region}_{unit['cluster_id']}")
            else:
                y_labels.append(str(y_offset))
            y_offset += 1
        ax.set_title(f"Spike Activity Across Regions: {', '.join(regions)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unit")
        if show_grid:
            ax.grid(visible=True, alpha=0.3)
        else:
            ax.grid(visible=False)
        ax.set_yticks(np.arange(0, y_offset, 1))
        ax.set_yticklabels(y_labels)
        handles = [
            Line2D([0], [0], color=color, marker='|', linestyle='None', markersize=10, label=region)
            for region, color in region_colors.items()
        ]
        ax.legend(handles=handles, fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc=legend_loc)
        plt.tight_layout()
        plt.show()

    def get_spike_times_by_region(self, region):
        """
        Extract spike times for all units in a brain region.
        
        Parameters
        ----------
        region : str
            Brain region name (e.g., 'CA1', 'PFC', 'RTC')
            
        Returns
        -------
        list of np.ndarray
            List of spike time arrays (in seconds) for each unit in the region
        """
        if not hasattr(self, 'region_mapping'):
            raise ValueError("Brain regions haven't been assigned. Use assign_brain_regions first.")
        
        spike_times_list = []
        
        # Find all tetrodes mapped to this region
        for tetrode, reg in self.region_mapping.items():
            if reg == region:
                # Get all units from this tetrode
                unit_dicts = self.tetrode_mapping.get(tetrode, [])
                
                # Extract spike times for each unit
                for unit_data in unit_dicts:
                    spike_times_list.append(unit_data['spk_time'])
        
        return spike_times_list

    def get_units_by_region(self, region):
        """
        Get all unit IDs assigned to a brain region.
        
        Parameters
        ----------
        region : str
            Brain region name
            
        Returns
        -------
        list
            List of unit cluster IDs in this region
        """
        if not hasattr(self, 'region_mapping'):
            raise ValueError("Brain regions haven't been assigned. Use assign_brain_regions first.")
        
        units = []
        for tetrode, reg in self.region_mapping.items():
            if reg == region:
                unit_dicts = self.tetrode_mapping.get(tetrode, [])
                units.extend([u['cluster_id'] for u in unit_dicts])
        return units

    def compute_mua(self, region, t_lfp, kernel_width=0.02):
        """
        Compute Multi-Unit Activity (MUA) for a brain region.
        
        This method bins spike times from all units in a region and convolves
        with a Gaussian kernel to produce a smoothed population activity signal
        aligned with an LFP timeline.
        
        Parameters
        ----------
        region : str
            Brain region name (e.g., 'CA1', 'PFC', 'RTC')
        t_lfp : np.ndarray
            LFP time vector (in seconds) to align MUA with
        kernel_width : float, optional
            Standard deviation of Gaussian smoothing kernel in seconds (default: 0.02)
            
        Returns
        -------
        np.ndarray
            MUA vector with shape matching t_lfp, representing firing rate (Hz)
            
        Examples
        --------
        >>> mua_ca1 = spike_analysis.compute_mua('CA1', t_lfp, kernel_width=0.02)
        >>> print(f"MUA shape: {mua_ca1.shape}, LFP shape: {t_lfp.shape}")
        """
        spike_times_list = self.get_spike_times_by_region(region)
        
        if len(spike_times_list) == 0:
            # Return zeros if no units in region
            return np.zeros(len(t_lfp))
        
        # Concatenate all spike times
        all_spikes = np.concatenate(spike_times_list)
        
        # Create time bins aligned with LFP
        dt = np.mean(np.diff(t_lfp))
        bins = np.concatenate([t_lfp - dt/2, [t_lfp[-1] + dt/2]])
        
        # Bin spikes
        spike_counts, _ = np.histogram(all_spikes, bins=bins)
        
        # Convert to firing rate (Hz)
        spike_rate = spike_counts / dt
        
        # Smooth with Gaussian kernel
        if kernel_width > 0:
            sigma_bins = kernel_width / dt
            kernel_size = int(6 * sigma_bins)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = gaussian(kernel_size, sigma_bins)
            kernel = kernel / kernel.sum()
            
            # Convolve with edge handling
            mua = np.convolve(spike_rate, kernel, mode='same')
        else:
            mua = spike_rate
        
        return mua

    def compute_mua_all_regions(self, t_lfp, kernel_width=0.02, regions=None):
        """
        Compute MUA for multiple brain regions.
        
        Parameters
        ----------
        t_lfp : np.ndarray
            LFP time vector (in seconds)
        kernel_width : float, optional
            Gaussian kernel width in seconds (default: 0.02)
        regions : list of str, optional
            List of regions to compute MUA for. If None, uses all assigned regions.
            
        Returns
        -------
        dict
            Dictionary mapping region names to MUA vectors
            
        Examples
        --------
        >>> mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp)
        >>> for region, mua in mua_by_region.items():
        ...     print(f"{region}: {len(mua)} samples")
        """
        if not hasattr(self, 'region_mapping'):
            raise ValueError("Brain regions haven't been assigned. Use assign_brain_regions first.")
        
        if regions is None:
            regions = list(self.region_to_tetrodes.keys())
        
        mua_by_region = {}
        for region in regions:
            mua_by_region[region] = self.compute_mua(region, t_lfp, kernel_width)
        
        return mua_by_region

    def visualize_mua_segments(self, mua_by_region, t_lfp, windows=None, 
                               figsize=None, show_spikes=True):
        """
        Visualize MUA segments with overlaid spike times for validation.
        
        Parameters
        ----------
        mua_by_region : dict
            Dictionary of region -> MUA vector (from compute_mua_all_regions)
        t_lfp : np.ndarray
            LFP time vector
        windows : list of tuples, optional
            Time windows to visualize as [(start, end), ...] in seconds.
            Default: [(1, 3), (5, 7)]
        figsize : tuple, optional
            Figure size (width, height). Auto-calculated if None.
        show_spikes : bool, optional
            Whether to overlay individual unit spike times (default: True)
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object
            
        Examples
        --------
        >>> mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp)
        >>> fig = spike_analysis.visualize_mua_segments(
        ...     mua_by_region, t_lfp, windows=[(10, 12), (20, 22)]
        ... )
        """
        if windows is None:
            windows = [(1, 3), (5, 7)]
        
        n_regions = len(mua_by_region)
        n_windows = len(windows)
        
        if figsize is None:
            figsize = (7 * n_windows, 3 * n_regions)
        
        fig, axes = plt.subplots(n_regions, n_windows, 
                                 figsize=figsize,
                                 squeeze=False)
        
        for col, (start, end) in enumerate(windows):
            mask = (t_lfp >= start) & (t_lfp <= end)
            t_seg = t_lfp[mask]
            
            for row, (region, mua_vec) in enumerate(mua_by_region.items()):
                ax = axes[row, col]
                mua_seg = mua_vec[mask]
                
                # Plot MUA
                ax.plot(t_seg, mua_seg, 'k-', lw=1.5, label='MUA')
                ax.fill_between(t_seg, 0, mua_seg, alpha=0.2, color='blue')
                
                # Overlay spike times if requested
                if show_spikes and hasattr(self, 'region_mapping'):
                    spike_times_list = self.get_spike_times_by_region(region)
                    
                    for unit_idx, spike_times in enumerate(spike_times_list):
                        unit_spikes = spike_times[(spike_times >= start) & (spike_times <= end)]
                        if len(unit_spikes) > 0:
                            y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.max(mua_seg)
                            ax.vlines(unit_spikes, 0, y_max * 0.95, 
                                     colors=f'C{unit_idx % 10}', alpha=0.5, 
                                     linewidths=1.5, label=f'Unit {unit_idx+1}' if unit_idx < 3 else '')
                
                ax.set_ylabel(f'{region}\nMUA (Hz)', fontsize=11)
                ax.set_title(f'{start:.1f}-{end:.1f}s', fontsize=10)
                ax.grid(alpha=0.3)
                
                if col == n_windows - 1 and show_spikes and len(spike_times_list) > 0:
                    # Only show legend for first few units to avoid clutter
                    handles, labels = ax.get_legend_handles_labels()
                    if len(handles) > 0:
                        ax.legend(handles[:min(4, len(handles))], 
                                 labels[:min(4, len(labels))],
                                 loc='upper right', fontsize=8)
        
        for ax in axes[-1, :]:
            ax.set_xlabel('Time (s)', fontsize=11)
        
        plt.suptitle('MUA Estimation Validation', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig