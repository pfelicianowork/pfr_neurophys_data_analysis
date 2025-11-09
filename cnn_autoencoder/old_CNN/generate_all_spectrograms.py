
import sys
import os
import numpy as np
import pickle
from PIL import Image

# Add workspace root to Python path for imports
workspace_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, workspace_root)

# --- Replicate SWR Detection Setup ---
from open_ephys_loader import fast_openephys_dat_lfp
from spike_analysis import SpikeAnalysis, loader, process_spike_data, load_processed_spike_data
# from swr_detection import SWRParams, SWRHMMDetector
from swr_detection.swr_hmm_detection import SWRHMMParams, SWRHMMDetector
from swr_detection.pipeline import find_region_channels, build_region_lfp
from swr_detection.mua_high_activity_refinement import refine_all_events_with_global_hmm
from swr_detection.swr_spectral_features import batch_compute_spectral_features

def generate_spectrogram_images():
    """
    This script performs SWR detection, computes spectrograms for each event using CWT,
    and saves each spectrogram as a PNG image in a new directory.
    """
    print("Replicating SWR detection setup...")

    # --- Configuration (from phase_one_classification.py) ---
    dat_path = r"D:\Spikeinterface_practice\s4_rec\ephys.dat"
    num_channels = 43
    # tetrode_groups = {
    # 'CA1': {'tet1': [17, 18, 19, 20], 'tet2': [21, 22, 23, 24]},
    # 'RTC': {'tet1': [14, 15, 16]},
    # 'PFC': {'tet1': [0, 1, 2, 3], 'tet2': [4, 5, 6, 7]},
    # }

    selected_channels = {
        'CA1_tet1': 17, 'CA1_tet2': 21, 'RTC_tet1': 14, 'PFC_tet1': 0, 'PFC_tet2': 5
    }
    fs_in = 30000.0
    fs_out = 1000.0
    output_dir = "all_spectrograms"

    # --- LFP and Spike Data Loading ---
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
        print(f"LFP duration: {loader.duration:.2f}s at {fs:.1f} Hz")

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

    except FileNotFoundError:
        print("\nERROR: Data files not found. Please ensure the paths in this script are correct.")
        print(f"Attempted to load LFP from: {dat_path}")
        print(f"Attempted to load spike data from: {npy_path}")
        print("Cannot proceed without data. Exiting.")
        return

    # --- SWR Detection ---
    ripple_th = 2.75  # High threshold in SD units
    params = SWRHMMParams(
        # Ripple detection
        
        ripple_band=(125, 250),
        threshold_multiplier=ripple_th,  # Not used when hysteresis enabled
        # Enable Karlsson-style preprocessing
        use_smoothing=True,
        smoothing_sigma=0.01,  # 25ms Gaussian smoothing
        normalization_method='zscore',  # or 'median_mad' for robust
        min_duration=0.025,
        max_duration=0.4,
        min_event_separation=0.07,
        merge_interval=0.07,
        trace_window=1.0,
        adaptive_classification=True,
        dbscan_eps=0.15,
        
        # MUA detection
        mua_threshold_multiplier=2.5,
        mua_min_duration=0.03,
        enable_mua=True,
        
        # HMM parameters
        use_hmm_edge_detection=False,
        hmm_margin=0.1,
        use_global_hmm=False,
        global_hmm_fraction=0.1,
        hmm_states_ripple=2,
        hmm_states_mua=2,
        
        # Hysteresis parameters (NEW!)
        use_hysteresis=True,
        # hysteresis_high_multiplier=ripple_th,
        hysteresis_low_multiplier=0.75,
        hysteresis_confirmation_window=0.07
    )

    # --- Initialize detector ---
    detector = SWRHMMDetector(
        lfp_data=lfp_array,   # shape: (n_channels, n_samples)
        fs=fs,
        mua_data=mua_vec,     # shape: (n_samples,)
        params=params
    )

    # --- Detect events (global HMM will be trained automatically if enabled) ---
    # detector.detect_events(channels='all', average_mode=True) # 
    detector.detect_events(channels=[0], average_mode=False)
    detector.classify_events_improved()
    # refine_all_events_with_global_hmm(detector.swr_events, mua_vec, fs)

    print(f"SWR detection complete. Found {len(detector.swr_events)} events.")

    # --- Step 1: Compute Spectrograms for all events ---
    print("Computing spectrograms for all events using CWT...")
    
    # Set the desired spectral method for each event
    for event in detector.swr_events:
        event['spec_method'] = 'cwt'

    # Run the batch computation
    lfp_array = region_lfp['CA1'][0]  # Use channel 0, or [1] for channel 1
    n_computed = batch_compute_spectral_features(
        detector, 
        lfp_array, 
        fs,
        use_optimized_cwt=True,
        n_workers=20,
        verbose=True, # Set to True to see progress
        target_freq_bins=150,
        n_bins=100,
        smoothing_sigma=1.0,
        pre_ms=250,
        post_ms=250
    )
    print(f"Successfully computed {n_computed} spectrograms.")

    # --- DEBUG: Print errors from the first few failed events ---
    print("\n--- Spectrogram Computation Debugging ---")
    errors_found = 0
    for i, event in enumerate(detector.swr_events):
        if 'spectrogram_error' in event and event['spectrogram_error'] is not None:
            print(f"Error for event {i}: {event['spectrogram_error']}")
            errors_found += 1
            if errors_found >= 5:
                break
    if errors_found == 0 and n_computed == 0 and len(detector.swr_events) > 0:
        print("No specific errors were stored, but computation still failed. There might be a silent issue.")
    print("--- End Debugging ---\n")

    # --- Step 2: Save Spectrograms as Images ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("Saving spectrograms as PNG images...")
    n_saved = 0
    for i, event in enumerate(detector.swr_events):
        spectrogram = event.get('spectrogram')
        if spectrogram is not None and isinstance(spectrogram, np.ndarray):
            try:
                # Normalize the spectrogram to 0-255 for image conversion
                spec_min = spectrogram.min()
                spec_max = spectrogram.max()
                if spec_max > spec_min:
                    spec_norm = (spectrogram - spec_min) / (spec_max - spec_min) * 255
                else:
                    spec_norm = np.zeros_like(spectrogram)
                
                # Convert to an 8-bit grayscale image
                img = Image.fromarray(spec_norm.astype(np.uint8), 'L')
                
                # Save the image
                img_path = os.path.join(output_dir, f"event_{i:05d}.png")
                img.save(img_path)
                n_saved += 1
            except Exception as e:
                print(f"Could not save image for event {i}: {e}")

        print(f"Successfully saved {n_saved} images to the '{output_dir}' directory.")

    

        # --- Step 3: Save the events list for the next step ---

        events_path = "detected_events.pkl"

        with open(events_path, "wb") as f:

            pickle.dump(detector.swr_events, f)

        print(f"Detected events list saved to '{events_path}'.")

    

        print("\nNext step: Train an autoencoder on these images.")

if __name__ == "__main__":
    generate_spectrogram_images()
