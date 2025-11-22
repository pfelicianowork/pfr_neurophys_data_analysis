import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SWR CNN Autoencoder & Classification Pipeline

    This notebook runs the complete SWR classification pipeline by importing and executing the main functions from the scripts in the `../cnn_autoencoder` directory.

    **Workflow:**
    1.  **Setup Paths:** Configure paths and set the main recording directory for inputs/outputs.
    2.  **Step 1:** Generate spectrograms and extract biological features.
    3.  **Step 2:** Train the CNN autoencoder (ResNet or VAE) on the spectrograms.
    4.  **Step 3:** Cluster events using a combination of autoencoder and biological features.
    5.  **Step 4:** Evaluate and visualize the final clusters.
    """)
    return


@app.cell
def _():
    import torch

    # 1. Check if CUDA is available to PyTorch
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 2. Check the CUDA version PyTorch was compiled with
    print(f"PyTorch CUDA Version: {torch.version.cuda}")

    # 3. Check the cuDNN version PyTorch is using
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")

    # 4. Get the name of your GPU
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    print("DEBUG: CuPy/CUDA status at script startup")
    try:
        import cupy
        print("CuPy version:", cupy.__version__)
        print("CUDA runtime version:", cupy.cuda.runtime.runtimeGetVersion())
        print("CUDA driver version:", cupy.cuda.runtime.driverGetVersion())
        # print("GPU name:", cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())
    except Exception as e:
        print("CuPy/CUDA ERROR:", e)
    print("="*60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What it means: The CuPy library is looking for a system-wide setting called CUDA_PATH. This path tells programs where the full NVIDIA CUDA Toolkit (which includes the compiler nvcc and other development tools) is installed.

    Is it a problem? Probably not. Many Python packages, like PyTorch and CuPy, now ship with the essential CUDA libraries they need to run code. This warning only becomes an issue if you try to do something that requires compiling new CUDA code from source (like installing a complex package or writing your own custom kernels).
    """)
    return


@app.cell
def _():
    import os
    import sys

    # --- 1. DEFINE PATHS & PARAMETERS ---

    # !! IMPORTANT: This is the user-specified path for all outputs (and inputs)
    recording_path = r"D:\Spikeinterface_practice"

    # This is the root of the project (pfr_neurophys_data_analysis)
    # Assumes this notebook is in pfr_neurophys_data_analysis/notebooks/
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..')) # root detection mechanism that automatically 
    #determines the parent directory path regardless of where the script is executed from.

    # Add the project root to the Python path to allow imports from cnn_autoencoder
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Change the current working directory to the recording path
    # All generated files (models, plots, .pkl) will be saved here.
    try:
        os.chdir(recording_path)
        print(f"Changed working directory to: {os.getcwd()}")
    except FileNotFoundError:
        print(f"ERROR: Recording path not found: {recording_path}")
        print("Please update the 'recording_path' variable in this cell.")

    print(f"Project root added to sys.path: {project_root}")

    # --- 2. DEFINE PIPELINE PARAMETERS ---
    # These can be adjusted as needed
    ARCH = 'resnet'      # 'resnet', 'vae', or 'attention'
    LATENT_DIM = 128     # Latent dimension for the autoencoder
    EPOCHS = 15          # Number of epochs for training
    return ARCH, EPOCHS, LATENT_DIM, os, project_root, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 1: Generate Spectrograms and Extract Features

    This step runs `generate_spectrograms.py`.
    - Detects SWR events from the raw data.
    - Generates spectrogram images for each event.
    - Extracts biological features (duration, frequency, power, etc.).

    **Output files (saved to recording_path):**
    - `all_spectrograms/` (directory)
    """)
    return


@app.cell
def _(os, sys):
    import numpy as np
    from open_ephys_loader import fast_openephys_dat_lfp
    from spike_analysis import SpikeAnalysis, loader, process_spike_data, load_processed_spike_data
    from swr_detection.swr_hmm_detection import SWRHMMParams, SWRHMMDetector
    from swr_detection.pipeline import find_region_channels, build_region_lfp
    from swr_detection.swr_spectral_features import batch_compute_spectral_features
    from cnn_autoencoder.feature_extraction import batch_extract_features, validate_biological_features
    '\nPerforms SWR detection, computes spectrograms, AND extracts biological features.\nThis creates a richer dataset for clustering.\n'
    print('=' * 80)
    print('IMPROVED SWR DETECTION WITH COMPREHENSIVE FEATURE EXTRACTION')
    print('=' * 80)
    dat_path = 'D:\\Spikeinterface_practice\\s4_rec\\ephys.dat'
    num_channels = 43
    selected_channels = {'CA1_tet1': 17, 'CA1_tet2': 21, 'RTC_tet1': 14, 'PFC_tet1': 0, 'PFC_tet2': 5}
    fs_in = 30000.0
    fs_out = 1000.0
    output_dir = 'D:\\Spikeinterface_practice\\all_spectrograms'
    print('\n--- Loading Data ---')
    try:
        loader = fast_openephys_dat_lfp(filepath=dat_path, num_channels=num_channels, tetrode_groups={}, selected_channels=selected_channels, sampling_frequency=fs_in, target_sampling_frequency=fs_out, return_mode='loader')
        fs = float(loader.sampling_frequency)
        t_lfp = loader.time_vector()
        print(f'✓ LFP duration: {loader.duration:.2f}s at {fs:.1f} Hz')
        npy_path = 'D:\\Spikeinterface_practice\\s4_rec\\phyMS5'
        save_path = 'D:\\Spikeinterface_practice\\s4_rec'
        if not os.path.exists(os.path.join(save_path, 'units.npy')):
            print('Processing spike data...')
            process_spike_data(npy_path, save_path, samp_freq=30000)
        units_file = os.path.join(save_path, 'units.npy')
        processed_spike_data = load_processed_spike_data(units_file)
        spike_analysis = SpikeAnalysis(processed_data=processed_spike_data, sampling_rate=30000, duration=loader.duration)
        region_mapping = {7: 'CA1', 8: 'CA1', 6: 'RTC', 2: 'PFC', 3: 'PFC'}
        spike_analysis.assign_brain_regions(region_mapping)
        mua_by_region = spike_analysis.compute_mua_all_regions(t_lfp=t_lfp, kernel_width=0.01)
        mua_vec = mua_by_region['CA1']
        region_channels = find_region_channels(list(loader.selected_channels.keys()))
        region_lfp = build_region_lfp(loader, region_channels)
        lfp_array = region_lfp['CA1']
        print(f'✓ LFP shape: {lfp_array.shape}')
        print(f'✓ MUA vector length: {len(mua_vec)}')
    except FileNotFoundError as e:
        print(f'\nERROR: Data files not found: {e}')
        print(f'Attempted to load LFP from: {dat_path}')
        print('Cannot proceed without data. Exiting.')
        sys.exit(1)
    print('\n--- Detecting SWRs ---')
    ripple_th = 2.75
    params = SWRHMMParams(ripple_band=(120, 250), threshold_multiplier=ripple_th, use_smoothing=True, smoothing_sigma=0.01, normalization_method='zscore', min_duration=0.025, max_duration=0.4, min_event_separation=0.07, merge_interval=0.07, trace_window=1.0, adaptive_classification=True, dbscan_eps=0.15, mua_threshold_multiplier=2.5, mua_min_duration=0.03, enable_mua=True, use_hmm_edge_detection=False, hmm_margin=0.1, use_global_hmm=False, global_hmm_fraction=0.1, hmm_states_ripple=2, hmm_states_mua=2, use_hysteresis=False, hysteresis_low_multiplier=0.75, hysteresis_confirmation_window=0.07)
    detector = SWRHMMDetector(lfp_data=lfp_array, fs=fs, mua_data=mua_vec, params=params)
    detector.detect_events(channels=[0], average_mode=False)
    detector.classify_events_improved()
    print(f'✓ Found {len(detector.swr_events)} events')
    print('\n--- Computing Spectrograms ---')
    pre_ms = 100
    post_ms = 100
    for _event in detector.swr_events:
        _event['spec_method'] = 'cwt'
        _event['spec_pre_ms'] = pre_ms
        _event['spec_post_ms'] = post_ms
    lfp_channel = region_lfp['CA1'][0]
    n_computed = batch_compute_spectral_features(detector, lfp_channel, fs, use_gpu='auto', gpu_batch_size='auto', use_optimized_cwt=True, n_workers=20, verbose=True, target_freq_bins=150, n_bins=100, smoothing_sigma=1.0, pre_ms=pre_ms, post_ms=post_ms)
    print(f'✓ Successfully computed {n_computed} spectrograms')
    peak_freqs = []
    for e in detector.swr_events:
        spec = e.get('spectrogram')
        freqs = e.get('spectrogram_freqs')
        if spec is not None and freqs is not None and (spec.ndim == 2):
            power_avg = np.mean(spec, axis=1)
            peak_idx = np.argmax(power_avg)
            peak_freqs.append(freqs[peak_idx])
    if peak_freqs:
        peak_freqs = np.array(peak_freqs)
        print(f'\n📊 Peak Frequency Statistics:')
        print(f'   Mean: {np.mean(peak_freqs):.1f} Hz')
        print(f'   Median: {np.median(peak_freqs):.1f} Hz')
        print(f'   Std: {np.std(peak_freqs):.1f} Hz')
        print(f'   Range: [{np.min(peak_freqs):.1f}, {np.max(peak_freqs):.1f}] Hz')
        if np.mean(peak_freqs) < 125:
            print(f'   ⚠️ WARNING: Mean peak frequency is LOW ({np.mean(peak_freqs):.1f} Hz)')
            print(f'              Expected ripple range: 125-250 Hz')
        elif np.mean(peak_freqs) > 200:
            print(f'   ⚠️ WARNING: Mean peak frequency is HIGH ({np.mean(peak_freqs):.1f} Hz)')
        else:
            print(f'   ✅ Peak frequencies in expected ripple range (125-250 Hz)')
    print('\n' + '=' * 60)
    return (
        batch_extract_features,
        detector,
        fs,
        lfp_channel,
        mua_vec,
        np,
        output_dir,
        region_lfp,
        validate_biological_features,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 2 Extracting Features

    **Output files (saved to recording_path):**
    - `detected_events.pkl`
    - `biological_features.pkl`
    """)
    return


@app.cell
def _(
    batch_extract_features,
    detector,
    fs,
    lfp_channel,
    mua_vec,
    np,
    os,
    output_dir,
    region_lfp,
    validate_biological_features,
):
    import pickle
    from PIL import Image
    import importlib
    import cnn_autoencoder.feature_extraction
    importlib.reload(_cnn_autoencoder.feature_extraction)
    print('\n--- Extracting Biological Features ---')
    extension_ms = 100
    feature_matrix, feature_names = batch_extract_features(detector.swr_events, lfp_channel, mua_vec, fs, region_lfp=region_lfp, verbose=True, normalize_lfp=True, extension_ms=extension_ms)
    print(f'✓ Extracted {feature_matrix.shape[1]} features from {feature_matrix.shape[0]} events')
    print(f'\nFeature names: {feature_names}')
    validate_biological_features(feature_matrix, feature_names)
    print('\n--- Saving Spectrograms ---')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    n_saved = 0
    for _i, _event in enumerate(detector.swr_events):
        spectrogram = _event.get('spectrogram')
        if spectrogram is not None and isinstance(spectrogram, np.ndarray):
            try:
                spec_min = spectrogram.min()
                spec_max = spectrogram.max()
                if spec_max > spec_min:
                    spec_norm = (spectrogram - spec_min) / (spec_max - spec_min) * 255
                else:
                    spec_norm = np.zeros_like(spectrogram)
                img = Image.fromarray(spec_norm.astype(np.uint8), 'L')
                img_path = os.path.join(output_dir, f'event_{_i:05d}.png')
                img.save(img_path)
                n_saved = n_saved + 1
            except Exception as e:
                if n_saved < 5:
                    print(f'Could not save image for event {_i}: {e}')
    print(f"✓ Successfully saved {n_saved} images to '{output_dir}/'")
    print('\n--- Saving Data ---')
    events_path = 'detected_events.pkl'
    with open(events_path, 'wb') as _f:
        pickle.dump(detector.swr_events, _f)
    print(f"✓ Saved events to '{os.path.join(os.getcwd(), events_path)}'")
    bio_features_path = 'biological_features.pkl'
    with open(bio_features_path, 'wb') as _f:
        pickle.dump({'feature_matrix': feature_matrix, 'feature_names': feature_names}, _f)
    print(f"✓ Saved biological features to '{os.path.join(os.getcwd(), bio_features_path)}'")
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total events detected: {len(detector.swr_events)}')
    print(f'Spectrograms saved: {n_saved}')
    print(f'Biological features extracted: {feature_matrix.shape[1]}')
    print(f'Total feature dimensions: {feature_matrix.shape}')
    print('\nNext steps:')
    print('  1. Train autoencoder: python train_autoencoder_improved.py')
    print('  2. Cluster with combined features: python cluster_events_improved.py')
    print('=' * 80 + '\n')
    return feature_matrix, feature_names, importlib, pickle


@app.cell
def _(lfp_channel, np, region_lfp):
    print("lfp_channel shape:", np.asarray(lfp_channel).shape)
    print("region_lfp['CA1'] shape:", np.asarray(region_lfp['CA1']).shape)
    return


@app.cell
def _(detector, feature_matrix, feature_names, importlib):
    import cnn_autoencoder.feature_viz as fv
    importlib.reload(fv)
    from cnn_autoencoder.feature_viz import visualize_features
    fv.visualize_features(feature_matrix, feature_names, events=detector.swr_events)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Train Autoencoder

    This step runs `train_autoencoder.py`.
    - Loads the spectrograms from `all_spectrograms/`.
    - Trains the specified CNN autoencoder (`resnet` or `vae`).

    **Output files (saved to recording_path):**
    - `full_model_resnet.pkl` (or similar, based on arch)
    - `encoder_model_resnet.pkl`
    - `training_history_resnet.png`
    """)
    return


@app.cell
def _(ARCH, EPOCHS, LATENT_DIM, importlib, os, output_dir, project_root):
    print('\n' + '=' * 80)
    print('STEP 2: TRAINING AUTOENCODER...')
    print('=' * 80)
    try:
        import cnn_autoencoder.train_autoencoder
        importlib.reload(_cnn_autoencoder.train_autoencoder)
        from cnn_autoencoder.train_autoencoder import train_autoencoder
        data_dir = os.path.dirname(output_dir)
        cnn_files_dir = 'F:\\Spikeinterface_practice\\cnn_files'
        print(f"Looking for spectrograms in: {os.path.join(data_dir, 'all_spectrograms')}")
        train_autoencoder(arch=ARCH, latent_dim=LATENT_DIM, epochs=EPOCHS, lr=None, beta=1.0, data_dir=data_dir, cnn_files_dir=cnn_files_dir)
        print('\n' + '-' * 80)
        print('STEP 2 COMPLETE')
        print('-' * 80)
    except ImportError as e:
        print(f"ERROR: Could not import 'train_autoencoder'.")
        print(f"Ensure 'train_autoencoder.py' is in {os.path.join(project_root, 'cnn_autoencoder')}")
        print(f'Import error: {e}')
    except Exception as e:
        print(f'An error occurred during Step 2: {e}')
        import traceback
        traceback.print_exc()
    return cnn_files_dir, data_dir, traceback


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 3: Cluster Events (general)

    This step runs `cluster_events.py`.
    - Loads the trained encoder (`encoder_model_...pkl`).
    - Loads the biological features (`biological_features.pkl`).
    - Generates latent features from the autoencoder.
    - Combines features and performs clustering to find optimal k.

    **Output files (saved to recording_path):**
    - `events_with_clusters_combined.pkl`
    - `clustering_info_combined.pkl`
    - `clustering_metrics_plot.png`
    - `dendrogram.png`
    """)
    return


@app.cell
def _(
    ARCH,
    LATENT_DIM,
    cnn_files_dir,
    data_dir,
    feature_matrix,
    importlib,
    np,
    os,
):
    import cnn_autoencoder.train_autoencoder
    importlib.reload(_cnn_autoencoder.train_autoencoder)
    from cnn_autoencoder.train_autoencoder import load_encoder, encode_spectrograms
    encoder = load_encoder(os.path.join(cnn_files_dir, 'encoder_model_resnet.pkl'), arch=ARCH, latent_dim=LATENT_DIM)
    _latents = encode_spectrograms(encoder, image_dir=data_dir, n_events=feature_matrix.shape[0])
    np.save(os.path.join(cnn_files_dir, 'latent_matrix.npy'), _latents)
    return


@app.cell
def _():
    # print("\n" + "="*80)
    # print("STEP 3: CLUSTERING EVENTS...")
    # print("="*80)

    # try:
    #     # Force reload to pick up changes to cluster_events.py
    #     import importlib
    #     import cnn_autoencoder.cluster_events
    #     importlib.reload(cnn_autoencoder.cluster_events)
    #     from cnn_autoencoder.cluster_events import cluster_events_improved

    #     cluster_events_improved(
    #         arch=ARCH,
    #         latent_dim=LATENT_DIM,
    #         k_range=range(2, 13),  # Test k from 2 to 12
    #         ae_weight=0.7,
    #         bio_weight=0.3,
    #         use_combined=True    # <-- FIXED HERE
    #     )

    #     print("\n" + "-"*80)
    #     print("STEP 3 COMPLETE")
    #     print("-"*80)
    # except ImportError as e:
    #     print(f"ERROR: Could not import 'cluster_events_improved'.")
    #     print(f"Ensure 'cluster_events.py' is in {os.path.join(project_root, 'cnn_autoencoder')}")
    #     print(f"Import error: {e}")
    # except Exception as e:
    #     print(f"An error occurred during Step 3: {e}")
    #     import traceback
    #     traceback.print_exc()
    return


@app.cell
def _(cnn_files_dir, feature_matrix, importlib, np, os, pickle):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import cnn_autoencoder.advanced_clustering as adv_clust
    importlib.reload(adv_clust)
    print('Feature matrix shape:', getattr(globals().get('feature_matrix'), 'shape', None))
    latent_path = os.path.join(cnn_files_dir, 'latent_matrix.npy')
    if os.path.exists(latent_path):
        _latents = np.load(latent_path)
        print('Loaded latent matrix:', _latents.shape)
    else:
        raise RuntimeError(f"latent_matrix.npy not found; compute latents or save them to '{latent_path}'")
    ae_weight = 0.7
    bio_weight = 0.3
    scaler_ae = StandardScaler().fit(_latents)
    scaler_bio = StandardScaler().fit(feature_matrix)
    latents_s = scaler_ae.transform(_latents) * ae_weight
    bio_s = scaler_bio.transform(feature_matrix) * bio_weight
    pca_dim = 50
    if latents_s.shape[1] + bio_s.shape[1] > 200:
        pca = PCA(n_components=min(pca_dim, latents_s.shape[1] + bio_s.shape[1] - 1), random_state=0)
        X = pca.fit_transform(np.hstack([latents_s, bio_s]))
        print('Reduced to PCA dims:', X.shape)
    else:
        X = np.hstack([latents_s, bio_s])
        print('Combined feature matrix shape:', X.shape)
    k_range = range(2, 13)
    methods = ['kmeans', 'gmm', 'agglomerative']
    best_config, results_df = adv_clust.find_optimal_clusters(X, k_range=k_range, methods=methods)
    print('Best clustering:', best_config)
    ari_scores = adv_clust.cross_validate_clustering(X, n_clusters=int(best_config['k']), method=best_config['method'])
    print('Stability ARI (median):', np.median(ari_scores))
    with open('advanced_clustering_results.pkl', 'wb') as _f:
        pickle.dump({'best': best_config, 'results_df': results_df, 'ari_scores': ari_scores}, _f)
    print('Saved advanced clustering results -> advanced_clustering_results.pkl')
    try:
        adv_clust.plot_clustering_metrics(results_df, outpath='clustering_metrics_advanced.png')
        print('Saved clustering metrics plot.')
    except Exception:
        pass
    return X, best_config, results_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Evaluate and Validate Clusters

    This step runs `evaluate_clusters.py`.
    - Loads the clustered events (`events_with_clusters_combined.pkl`).
    - Performs detailed biological validation.
    - Generates summary plots and visualizations.

    **Output files (saved to recording_path):**
    - `feature_space_combined.png`
    - `cluster_validation_report_combined.txt`
    - `cluster_..._summary.png` (and other plots)
    """)
    return


@app.cell
def _(importlib, os, project_root, traceback):
    import cnn_autoencoder.evaluate_clusters
    importlib.reload(_cnn_autoencoder.evaluate_clusters)
    from cnn_autoencoder.evaluate_clusters import evaluate_clusters_improved
    print('\n' + '=' * 80)
    print('STEP 4: EVALUATING CLUSTERS...')
    print('=' * 80)
    try:
        import cnn_autoencoder.evaluate_clusters
        importlib.reload(_cnn_autoencoder.evaluate_clusters)
        evaluate_clusters_improved(feature_type='combined')
        print('\n' + '=' * 80)
        print('PIPELINE FINISHED!')
        print(f'All outputs saved to: {os.getcwd()}')
        print('=' * 80)
    except ImportError as e:
        print("ERROR: Could not import 'evaluate_clusters_improved'.")
        print(f"Ensure 'evaluate_clusters.py' is in {os.path.join(project_root, 'cnn_autoencoder')}")
        print(f'Import error: {e}')
    except Exception as e:
        print(f'An error occurred during Step 4: {e}')
        traceback.print_exc()
    return


@app.cell
def _(X, best_config, detector, np, pickle, results_df):
    if 'results_df' in globals() and 'labels' in results_df.columns:
        labels = results_df['labels'].astype(int).values
    elif isinstance(best_config, dict) and 'labels' in best_config:
        labels = np.asarray(best_config['labels']).astype(int)
    else:
        model = best_config.get('model') if isinstance(best_config, dict) else None
        if model is not None and hasattr(model, 'predict'):
            labels = model.predict(X).astype(int)
        else:
            raise RuntimeError('Could not locate labels. Inspect results_df, best_config or model.')
    if len(labels) != len(detector.swr_events):
        raise RuntimeError(f'Label count ({len(labels)}) != number of events ({len(detector.swr_events)}). Ensure same ordering.')
    for _i, (evt, lab) in enumerate(zip(detector.swr_events, labels)):
        evt['cluster_advanced'] = int(lab)
        evt['cluster_method'] = best_config.get('method') if isinstance(best_config, dict) else None
        evt['cluster_k'] = int(best_config.get('k')) if isinstance(best_config, dict) and best_config.get('k') is not None else None
    model = best_config.get('model') if isinstance(best_config, dict) else None
    if model is not None and hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        for _i, evt in enumerate(detector.swr_events):
            evt['cluster_probs'] = probs[_i].tolist()
    out_path = 'events_with_clusters_advanced.pkl'
    with open(out_path, 'wb') as _f:
        pickle.dump(detector.swr_events, _f)
    print(f'Saved events with cluster labels -> {out_path}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
