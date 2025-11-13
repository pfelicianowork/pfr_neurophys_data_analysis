"""
IMPROVED: Generate spectrograms and extract biological features in one pass.
This combines spectrogram generation with feature extraction.
"""

import sys
import os
import numpy as np
import pickle
from PIL import Image

# Add workspace root to Python path for imports
workspace_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, workspace_root)

# Import your existing modules
from open_ephys_loader import fast_openephys_dat_lfp
from spike_analysis import SpikeAnalysis, loader, process_spike_data, load_processed_spike_data
from swr_detection.swr_hmm_detection import SWRHMMParams, SWRHMMDetector
from swr_detection.pipeline import find_region_channels, build_region_lfp
from swr_detection.swr_spectral_features import batch_compute_spectral_features

# Import new feature extraction module
from feature_extraction import batch_extract_features, validate_biological_features


def generate_spectrograms_and_features():
    """
    Performs SWR detection, computes spectrograms, AND extracts biological features.
    This creates a richer dataset for clustering.
    """
    print("="*80)
    print("IMPROVED SWR DETECTION WITH COMPREHENSIVE FEATURE EXTRACTION")
    print("="*80)

    # --- Configuration ---
    dat_path = r"D:\Spikeinterface_practice\s4_rec\ephys.dat"
    num_channels = 43
    selected_channels = {
        'CA1_tet1': 17, 'CA1_tet2': 21, 'RTC_tet1': 14, 'PFC_tet1': 0, 'PFC_tet2': 5
    }
    fs_in = 30000.0
    fs_out = 1000.0
    output_dir = "all_spectrograms"

    # --- Load LFP and Spike Data ---
    print("\n--- Loading Data ---")
    try:
        loader = fast_openephys_dat_lfp(
            filepath=dat_path,
            num_channels=num_channels,
            tetrode_groups={},
            selected_channels=selected_channels,
            sampling_frequency=fs_in,
            target_sampling_frequency=fs_out,
            return_mode="loader",
        )
        fs = float(loader.sampling_frequency)
        t_lfp = loader.time_vector()
        print(f"✓ LFP duration: {loader.duration:.2f}s at {fs:.1f} Hz")

        # Load spike data
        npy_path = r'D:\Spikeinterface_practice\s4_rec\phyMS5'
        save_path = r'D:\Spikeinterface_practice\s4_rec'
        if not os.path.exists(os.path.join(save_path, 'units.npy')):
            print("Processing spike data...")
            process_spike_data(npy_path, save_path, samp_freq=30000)

        units_file = os.path.join(save_path, 'units.npy')
        processed_spike_data = load_processed_spike_data(units_file)

        spike_analysis = SpikeAnalysis(
            processed_data=processed_spike_data,
            sampling_rate=30000,
            duration=loader.duration
        )

        region_mapping = {7: 'CA1', 8: 'CA1', 6: 'RTC', 2: 'PFC', 3: 'PFC'}
        spike_analysis.assign_brain_regions(region_mapping)

        mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp=t_lfp, kernel_width=0.01)
        mua_vec = mua_by_region['CA1']

        region_channels = find_region_channels(list(loader.selected_channels.keys()))
        region_lfp = build_region_lfp(loader, region_channels)
        lfp_array = region_lfp['CA1']
        
        print(f"✓ LFP shape: {lfp_array.shape}")
        print(f"✓ MUA vector length: {len(mua_vec)}")

    except FileNotFoundError as e:
        print(f"\nERROR: Data files not found: {e}")
        print(f"Attempted to load LFP from: {dat_path}")
        print("Cannot proceed without data. Exiting.")
        return

    # --- SWR Detection ---
    print("\n--- Detecting SWRs ---")
    ripple_th = 2.75
    params = SWRHMMParams(
        ripple_band=(125, 250),
        threshold_multiplier=ripple_th,
        use_smoothing=True,
        smoothing_sigma=0.01,
        normalization_method='zscore',
        min_duration=0.025,
        max_duration=0.4,
        min_event_separation=0.07,
        merge_interval=0.07,
        trace_window=1.0,
        adaptive_classification=True,
        dbscan_eps=0.15,
        mua_threshold_multiplier=2.5,
        mua_min_duration=0.03,
        enable_mua=True,
        use_hmm_edge_detection=False,
        hmm_margin=0.1,
        use_global_hmm=False,
        global_hmm_fraction=0.1,
        hmm_states_ripple=2,
        hmm_states_mua=2,
        use_hysteresis=True,
        hysteresis_low_multiplier=0.75,
        hysteresis_confirmation_window=0.07
    )

    detector = SWRHMMDetector(
        lfp_data=lfp_array,
        fs=fs,
        mua_data=mua_vec,
        params=params
    )

    detector.detect_events(channels=[0], average_mode=False)
    detector.classify_events_improved()

    print(f"✓ Found {len(detector.swr_events)} events")

    # --- Compute Spectrograms ---
    print("\n--- Computing Spectrograms ---")
    for event in detector.swr_events:
        event['spec_method'] = 'cwt'

    lfp_channel = region_lfp['CA1'][0]
    n_computed = batch_compute_spectral_features(
        detector, 
        lfp_channel, 
        fs,
        use_optimized_cwt=True,
        n_workers=20,
        verbose=True,
        target_freq_bins=150,
        n_bins=100,
        smoothing_sigma=1.0,
        pre_ms=250,
        post_ms=250
    )
    print(f"✓ Successfully computed {n_computed} spectrograms")

    # --- Extract Biological Features ---
    print("\n--- Extracting Biological Features ---")
    feature_matrix, feature_names = batch_extract_features(
        detector.swr_events,
        lfp_channel,
        mua_vec,
        fs,
        region_lfp=region_lfp,
        verbose=True
    )
    
    print(f"✓ Extracted {feature_matrix.shape[1]} features from {feature_matrix.shape[0]} events")
    print(f"\nFeature names: {feature_names}")
    
    # Validate features
    validate_biological_features(feature_matrix, feature_names)

    # --- Save Spectrograms as Images ---
    print("\n--- Saving Spectrograms ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_saved = 0
    for i, event in enumerate(detector.swr_events):
        spectrogram = event.get('spectrogram')
        if spectrogram is not None and isinstance(spectrogram, np.ndarray):
            try:
                spec_min = spectrogram.min()
                spec_max = spectrogram.max()
                if spec_max > spec_min:
                    spec_norm = (spectrogram - spec_min) / (spec_max - spec_min) * 255
                else:
                    spec_norm = np.zeros_like(spectrogram)
                
                img = Image.fromarray(spec_norm.astype(np.uint8), 'L')
                img_path = os.path.join(output_dir, f"event_{i:05d}.png")
                img.save(img_path)
                n_saved += 1
            except Exception as e:
                if n_saved < 5:
                    print(f"Could not save image for event {i}: {e}")

    print(f"✓ Successfully saved {n_saved} images to '{output_dir}/'")

    # --- Save Data ---
    print("\n--- Saving Data ---")
    
    # Save events
    events_path = "detected_events.pkl"
    with open(events_path, "wb") as f:
        pickle.dump(detector.swr_events, f)
    print(f"✓ Saved events to '{events_path}'")
    
    # Save biological features
    bio_features_path = "biological_features.pkl"
    with open(bio_features_path, "wb") as f:
        pickle.dump({
            'feature_matrix': feature_matrix,
            'feature_names': feature_names
        }, f)
    print(f"✓ Saved biological features to '{bio_features_path}'")

    # --- Summary Statistics ---
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total events detected: {len(detector.swr_events)}")
    print(f"Spectrograms saved: {n_saved}")
    print(f"Biological features extracted: {feature_matrix.shape[1]}")
    print(f"Total feature dimensions: {feature_matrix.shape}")
    print("\nNext steps:")
    print("  1. Train autoencoder: python train_autoencoder_improved.py")
    print("  2. Cluster with combined features: python cluster_events_improved.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_spectrograms_and_features()
