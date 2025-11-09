import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from ipywidgets import IntSlider, IntText, Button, HBox, VBox, Output
from IPython.display import display

# --- State expansion utility ---
def expand_state_edges(states, target_state=1, expand_bins=2):
    """
    Expand target_state edges by a given number of bins and merge short interruptions.
    Returns a new state sequence.
    """
    expanded = states.copy()
    # Merge short interruptions
    for i in range(1, len(states) - 1):
        if states[i] != target_state and (states[i-1] == target_state or states[i+1] == target_state):
            expanded[i] = target_state
    # Expand edges
    for i in range(len(states)):
        if states[i] == target_state:
            for j in range(1, expand_bins+1):
                if i-j >= 0 and expanded[i-j] != target_state:
                    expanded[i-j] = target_state
                if i+j < len(states) and expanded[i+j] != target_state:
                    expanded[i+j] = target_state
    return expanded


def bin_mua(mua_vec, bin_size, fs):
    """
    Bin the MUA vector into counts per bin_size seconds.
    """
    bin_samples = int(bin_size * fs)
    n_bins = len(mua_vec) // bin_samples
    mua_binned = np.array([
        mua_vec[i*bin_samples:(i+1)*bin_samples].sum()
        for i in range(n_bins)
    ])
    return mua_binned


def train_hmm_on_mua(mua_binned, n_states=2, params=None):
    """
    Train a 2-state HMM on binned MUA (high/low activity states).
    """
    mua_binned = mua_binned.reshape(-1, 1)
    
    # Check if we have enough data
    if len(mua_binned) < 5:
        # Return a simple model for very short sequences
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=10, random_state=42)
        # Set reasonable defaults
        model.startprob_ = np.ones(n_states) / n_states
        model.transmat_ = np.eye(n_states) * 0.9 + (1 - np.eye(n_states)) * 0.1
        model.means_ = np.percentile(mua_binned, np.linspace(0, 100, n_states)).reshape(-1, 1)
        model.covars_ = np.ones((n_states, 1)) * np.var(mua_binned)
        return model
    
    # Fit the model
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(mua_binned)
    
    # IMPORTANT: Ensure transition matrix is valid after fitting
    # Check for zero rows and fix
    row_sums = model.transmat_.sum(axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] == 0 or np.isnan(row_sums[i]):
            # Replace with uniform distribution
            model.transmat_[i, :] = 1.0 / n_states
    
    # Normalize each row to sum to 1
    transmat_sum = model.transmat_.sum(axis=1, keepdims=True)
    model.transmat_ = model.transmat_ / transmat_sum
    
    # Ensure start probabilities sum to 1
    startprob_sum = model.startprob_.sum()
    if startprob_sum == 0 or np.isnan(startprob_sum):
        model.startprob_ = np.ones(n_states) / n_states
    else:
        model.startprob_ = model.startprob_ / startprob_sum
    
    return model


def extract_high_activity_periods(states, bin_size, fs):
    """
    Extract contiguous high activity periods from HMM state sequence.
    Returns list of (start_time, end_time) tuples in seconds.
    """
    # Rank states by mean MUA to find which is high activity
    unique_states = np.unique(states)
    state_means = [np.mean(states == s) for s in unique_states]
    high_state = unique_states[np.argmax(state_means)]
    periods = []
    in_high = False
    start_idx = 0
    for i, s in enumerate(states):
        if s == high_state and not in_high:
            start_idx = i
            in_high = True
        elif s != high_state and in_high:
            end_idx = i
            in_high = False
            periods.append((start_idx * bin_size, end_idx * bin_size))
    # Handle case where high activity continues to end
    if in_high:
        periods.append((start_idx * bin_size, len(states) * bin_size))
    return periods


def refine_event_with_mua_states(event, mua_binned, fs, *, bin_size=0.01, params=None):
    """
    Refine event boundaries using high activity state transitions in binned MUA.
    Returns (refined_start_time, refined_end_time) in seconds.
    """
    # Extract event window
    start_time = event.get('start_time', None)
    end_time = event.get('end_time', None)
    if start_time is None or end_time is None:
        return None, None
    start_bin = int(start_time / bin_size)
    end_bin = int(end_time / bin_size)
    mua_event = mua_binned[start_bin:end_bin]
    if len(mua_event) < 5:
        return start_time, end_time
    # Train HMM on event MUA
    model = train_hmm_on_mua(mua_event, n_states=2, params=params)
    states = model.predict(mua_event.reshape(-1, 1))
    # Find high activity periods
    periods = extract_high_activity_periods(states, bin_size, fs)
    # Find the largest high activity period within the event
    if not periods:
        return start_time, end_time
    largest = max(periods, key=lambda p: p[1] - p[0])
    refined_start, refined_end = largest
    # Convert to absolute time
    refined_start += start_time
    refined_end += start_time
    return refined_start, refined_end


def refine_event_with_global_states(event, mua_binned, global_states, fs, bin_size=0.005, search_margin=0.2, expand_to_state_edges=False):
    """
    Refine event boundaries using pre-computed global HMM states.
    
    IMPROVED: Searches in an EXTENDED window around the ripple event to capture full MUA activity.
    This solves the problem where ripple boundaries are narrow but MUA activity extends beyond them.
    
    Args:
        event: Event dictionary with 'start_time', 'end_time'
        mua_binned: Full binned MUA array
        global_states: Pre-computed state sequence for entire recording
        fs: Sampling frequency
        bin_size: Bin size in seconds
        search_margin: How far to extend search window beyond ripple boundaries (seconds, default: 0.2)
        
    Returns:
        (refined_start_time, refined_end_time) in seconds
    """
    # Get ripple boundaries
    ripple_start = event.get('combined_start_time', event.get('start_time'))
    ripple_end = event.get('combined_end_time', event.get('end_time'))
    
    if ripple_start is None or ripple_end is None:
        return None, None
    
    # EXTENDED SEARCH WINDOW (e.g., ±200ms around ripple)
    search_start = max(0, ripple_start - search_margin)
    search_end = min((len(mua_binned) * bin_size), ripple_end + search_margin)
    
    # Convert to bin indices
    search_start_bin = int(search_start / bin_size)
    search_end_bin = int(search_end / bin_size)
    ripple_peak_bin = int(event.get('peak_time', (ripple_start + ripple_end) / 2) / bin_size)
    
    # Identify high-activity state using GLOBAL statistics (more robust than per-event)
    n_states = int(np.max(global_states) + 1) if global_states.size > 0 else 1
    global_state_means = [mua_binned[global_states == s].mean() if (global_states == s).any() else 0 
                          for s in range(n_states)]
    high_state = int(np.argmax(global_state_means))

    # If user wants to expand to the full contiguous high-state edges, choose
    # a representative high-state bin (prefer one inside the search window) and
    # expand outward over the entire global_states vector.
    if expand_to_state_edges:
        # All high-state bins in recording
        high_bins_all = np.where(global_states == high_state)[0]
        if high_bins_all.size == 0:
            return ripple_start, ripple_end

        # Prefer high-state bins inside the search window
        in_search = high_bins_all[(high_bins_all >= search_start_bin) & (high_bins_all <= search_end_bin)]
        if in_search.size > 0:
            center_bin = int(in_search[np.argmin(np.abs(in_search - ripple_peak_bin))])
        else:
            center_bin = int(high_bins_all[np.argmin(np.abs(high_bins_all - ripple_peak_bin))])

        # Expand outward to contiguous high-state boundaries on the full state vector
        start_bin = center_bin
        while start_bin > 0 and global_states[start_bin - 1] == high_state:
            start_bin -= 1
        end_bin = center_bin
        while end_bin < len(global_states) - 1 and global_states[end_bin + 1] == high_state:
            end_bin += 1

        refined_start = start_bin * bin_size
        refined_end = (end_bin + 1) * bin_size
        return refined_start, refined_end

    # --- original behavior (search-window based) ---
    # Extract states in EXTENDED window
    search_states = global_states[search_start_bin:search_end_bin]
    

    if len(search_states) == 0:
        # Always return ripple boundaries if no search window
        return ripple_start, ripple_end

    # Find ALL high-activity bins in extended window (relative indices)
    high_indices = np.where(search_states == high_state)[0]

    if len(high_indices) == 0:
        # Always set to ripple boundaries for display, even if no MUA high-activity state
        return ripple_start, ripple_end

    # Find the high-activity segment that CONTAINS the ripple peak
    ripple_peak_relative = ripple_peak_bin - search_start_bin

    # Label connected components of high-activity state
    diff = np.diff(high_indices)
    breaks = np.where(diff > 1)[0] + 1
    segments = np.split(high_indices, breaks)

    # Find which segment contains the ripple peak
    containing_segment = None
    for seg in segments:
        if len(seg) > 0 and seg[0] <= ripple_peak_relative <= seg[-1]:
            containing_segment = seg
            break

    if containing_segment is None:
        # Fallback: find closest segment to peak
        distances = [np.abs(seg.mean() - ripple_peak_relative) for seg in segments if len(seg) > 0]
        if len(distances) > 0:
            containing_segment = segments[np.argmin(distances)]
        else:
            return ripple_start, ripple_end

    # Convert back to time (search window offset + relative indices)
    refined_start = (search_start_bin + containing_segment[0]) * bin_size
    refined_end = (search_start_bin + containing_segment[-1] + 1) * bin_size

    return refined_start, refined_end


def refine_all_events_with_mua_states(events, mua_vec, fs, bin_size=0.01, params=None):
    """
    Batch refinement for all events.
    Updates each event dict with 'mua_high_start' and 'mua_high_end'.
    """
    mua_binned = bin_mua(mua_vec, bin_size, fs)
    for event in events:
        refined_start, refined_end = refine_event_with_mua_states(event, mua_binned, fs, bin_size=bin_size, params=params)
        event['mua_high_start'] = refined_start
        event['mua_high_end'] = refined_end
    return events


def refine_all_events_with_global_hmm(events, mua_vec, fs, bin_size=0.005, n_states=2, params=None, 
                                      search_margin=0.2, expand_bins=0, expand_to_state_edges=False,
                                      save_to=None):
    """
    Batch refinement using a single global HMM trained on the entire recording.
    This is more robust than training per-event HMMs.
    
    IMPROVED: Now uses extended search window to capture full MUA extent beyond narrow ripple boundaries.
    
    Args:
        events: List of event dictionaries
        mua_vec: Full MUA vector (continuous, at LFP sampling rate)
        fs: Sampling frequency
        bin_size: Bin size for MUA (in seconds, default: 0.005 = 5ms)
        n_states: Number of HMM states (default: 2)
        params: Optional parameters
        search_margin: Extended search window around ripple (seconds, default: 0.2 = 200ms)
        expand_bins: Additional bins to expand high-activity state edges (default: 0)
        
    Returns:
        events: Updated with 'mua_high_start' and 'mua_high_end'
    """
    # Bin MUA once
    mua_binned = bin_mua(mua_vec, bin_size, fs)
    
    # Train global HMM once
    global_model = train_hmm_on_mua(mua_binned, n_states=n_states, params=params)
    global_states = global_model.predict(mua_binned.reshape(-1, 1))
    
    # Optional: expand state edges
    if expand_bins > 0:
        state_means = [mua_binned[global_states == s].mean() if (global_states == s).any() else 0 
                       for s in range(n_states)]
        high_state = int(np.argmax(state_means))
        global_states = expand_state_edges(global_states, target_state=high_state, expand_bins=expand_bins)

    # Optionally save computed objects to the provided container (e.g. detector)
    if save_to is not None:
        try:
            setattr(save_to, 'mua_binned', mua_binned)
            setattr(save_to, 'global_states', global_states)
            setattr(save_to, 'mua_bin_size', bin_size)
        except Exception:
            # non-fatal if assignment fails
            pass
    
    # Refine all events using global states with extended search
    for event in events:
        refined_start, refined_end = refine_event_with_global_states(
            event, mua_binned, global_states, fs, bin_size=bin_size, search_margin=search_margin,
            expand_to_state_edges=expand_to_state_edges
        )
        event['mua_high_start'] = refined_start
        event['mua_high_end'] = refined_end
    
    return events


def interactive_mua_state_browser(mua_binned, states, bin_size=0.01, window_sec=10):
    """
    Interactive widget to browse MUA and high/low activity states across the recording,
    with Next/Previous buttons and direct window selection.
    """
    n_bins = len(mua_binned)
    total_sec = n_bins * bin_size
    min_window = 1
    max_window = int(total_sec // 2)
    window_slider = IntSlider(value=window_sec, min=min_window, max=max_window, step=1, description='Window (s)')
    out = Output()
    idx = IntText(value=0, min=0, max=int(total_sec - window_sec), description='Start (s)')
    prev_btn = Button(description='Previous')
    next_btn = Button(description='Next')

    def plot_window(start_sec, window_sec, expand_target=None, expand_bins=0):
        max_start = int(total_sec - window_sec)
        start_sec = max(0, min(start_sec, max_start))
        start_bin = int(start_sec / bin_size)
        end_bin = int((start_sec + window_sec) / bin_size)
        t = np.arange(start_bin, end_bin) * bin_size
        mb = mua_binned[start_bin:end_bin]
        st = states[start_bin:end_bin]
        unique_states = np.unique(states)
        palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'gray']

        # Optionally expand a target state
        if expand_target is not None and expand_bins > 0:
            st = expand_state_edges(st, target_state=expand_target, expand_bins=expand_bins)

        plt.figure(figsize=(10, 3))
        plt.plot(t, mb, label='Binned MUA', color='black', lw=1)
        for i, s in enumerate(unique_states):
            plt.fill_between(
                t, 0, mb, where=(st == s),
                color=palette[i % len(palette)], alpha=0.3,
                label=f'State {s}'
            )
        plt.xlabel('Time (s)')
        plt.ylabel('MUA (binned count)')
        plt.title(f'MUA with Activity States: {start_sec:.1f}-{start_sec+window_sec:.1f}s')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def on_prev(b):
        if idx.value - window_slider.value >= 0:
            idx.value -= window_slider.value

    def on_next(b):
        if idx.value + window_slider.value <= int(total_sec - window_slider.value):
            idx.value += window_slider.value

    # Add controls for edge expansion
    expand_state = IntText(value=None, description='Expand State (int)')
    expand_bins = IntText(value=0, description='Expand Bins')

    def on_idx_change(change):
        with out:
            out.clear_output(wait=True)
            plot_window(change['new'], window_slider.value, expand_state.value, expand_bins.value)

    def on_window_change(change):
        idx.max = int(total_sec - window_slider.value)
        with out:
            out.clear_output(wait=True)
            plot_window(idx.value, change['new'], expand_state.value, expand_bins.value)

    def on_expand_change(change):
        with out:
            out.clear_output(wait=True)
            plot_window(idx.value, window_slider.value, expand_state.value, expand_bins.value)

    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    idx.observe(on_idx_change, names='value')
    window_slider.observe(on_window_change, names='value')
    expand_state.observe(on_expand_change, names='value')
    expand_bins.observe(on_expand_change, names='value')

    controls = HBox([prev_btn, next_btn, idx, window_slider, expand_state, expand_bins])
    display(VBox([controls, out]))
    plot_window(idx.value, window_slider.value, expand_state.value, expand_bins.value)


def visualize_event_refinement_comparison(detector, mua_vec, event_idx=0, margin=0.5):
    """
    Visualize how MUA-based refinement extends SWR event boundaries.
    
    Shows side-by-side comparison of:
    - Ripple power HMM boundaries (blue)
    - MUA high-activity boundaries (red)
    
    Args:
        detector: SWRHMMDetector with detected events
        mua_vec: Full MUA vector (continuous, at LFP sampling rate)
        event_idx: Which event to visualize (default: 0)
        margin: Time margin around event to display (seconds, default: 0.5)
    """
    if event_idx >= len(detector.swr_events):
        print(f"Event index {event_idx} out of range. Only {len(detector.swr_events)} events detected.")
        return
    
    event = detector.swr_events[event_idx]
    
    # Get event times
    ripple_start = event.get('combined_start_time', event.get('start_time'))
    ripple_end = event.get('combined_end_time', event.get('end_time'))
    mua_start = event.get('mua_high_start')
    mua_end = event.get('mua_high_end')
    peak_time = event.get('peak_time', (ripple_start + ripple_end) / 2)
    
    # Get LFP data (handle both dict and array cases)
    if isinstance(detector.lfp_data, dict):
        # Multi-region case - use first region or mean
        lfp_full = list(detector.lfp_data.values())[0]
    else:
        # Single array case
        lfp_full = detector.lfp_data if detector.lfp_data.ndim == 1 else np.mean(detector.lfp_data, axis=0)
    
    # Define window
    window_start = max(0, ripple_start - margin)
    window_end = min(len(lfp_full) / detector.fs, ripple_end + margin)
    
    # Extract signals
    start_idx = int(window_start * detector.fs)
    end_idx = int(window_end * detector.fs)
    t = np.arange(start_idx, end_idx) / detector.fs
    lfp = lfp_full[start_idx:end_idx]
    mua = mua_vec[start_idx:end_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot LFP
    ax1.plot(t, lfp, 'k', lw=0.5, label='Filtered LFP')
    ax1.axvspan(ripple_start, ripple_end, color='blue', alpha=0.2, label='Ripple HMM')
    if mua_start is not None and mua_end is not None:
        ax1.axvspan(mua_start, mua_end, color='red', alpha=0.2, label='MUA Refined')
    ax1.axvline(peak_time, color='orange', ls='--', lw=1, label='Peak')
    ax1.set_ylabel('LFP (μV)')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Event {event_idx}: Ripple vs MUA Boundaries')
    
    # Plot MUA
    ax2.plot(t, mua, 'k', lw=0.5, label='MUA')
    ax2.axvspan(ripple_start, ripple_end, color='blue', alpha=0.2, label='Ripple HMM')
    if mua_start is not None and mua_end is not None:
        ax2.axvspan(mua_start, mua_end, color='red', alpha=0.2, label='MUA Refined')
    ax2.axvline(peak_time, color='orange', ls='--', lw=1, label='Peak')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('MUA (a.u.)')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    ripple_duration = ripple_end - ripple_start
    if mua_start is not None and mua_end is not None:
        mua_duration = mua_end - mua_start
        extension = mua_duration - ripple_duration
        print(f"\nEvent {event_idx} Statistics:")
        print(f"  Ripple HMM:  {ripple_start:.3f}s - {ripple_end:.3f}s  (duration: {ripple_duration*1000:.1f}ms)")
        print(f"  MUA Refined: {mua_start:.3f}s - {mua_end:.3f}s  (duration: {mua_duration*1000:.1f}ms)")
        print(f"  Extension: {extension*1000:.1f}ms ({extension/ripple_duration*100:.1f}% longer)")
    else:
        print(f"\nEvent {event_idx}: No MUA refinement available")


def interactive_event_refinement_browser(detector, mua_vec, margin=0.5):
    """
    Interactive widget to browse through events and compare ripple vs MUA boundaries.
    
    Args:
        detector: SWRHMMDetector with detected events
        mua_vec: Full MUA vector
        margin: Time margin around event (seconds)
    """
    from ipywidgets import IntSlider, Button, HBox, VBox, Output
    from IPython.display import display
    
    n_events = len(detector.swr_events)
    if n_events == 0:
        print("No events to display")
        return
    
    event_slider = IntSlider(value=0, min=0, max=n_events-1, description='Event:')
    prev_btn = Button(description='Previous')
    next_btn = Button(description='Next')
    out = Output()
    
    def plot_event(event_idx):
        with out:
            out.clear_output(wait=True)
            visualize_event_refinement_comparison(detector, mua_vec, event_idx, margin)
    
    def on_prev(b):
        if event_slider.value > 0:
            event_slider.value -= 1
    
    def on_next(b):
        if event_slider.value < n_events - 1:
            event_slider.value += 1
    
    def on_slider_change(change):
        plot_event(change['new'])
    
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    event_slider.observe(on_slider_change, names='value')
    
    controls = HBox([prev_btn, next_btn, event_slider])
    display(VBox([controls, out]))
    plot_event(0)


def classify_event(event):
    """Get the event classification from the original detection."""
    # Use the event_type that was already determined during original detection
    # If not present, infer from available fields
    et = event.get('event_type', None)
    if et is not None:
        return et
    # Fallback: infer from MUA and ripple fields
    mua_present = event.get('mua_high_start') is not None and event.get('mua_high_end') is not None
    ripple_present = event.get('combined_start_time', event.get('start_time')) is not None and event.get('combined_end_time', event.get('end_time')) is not None
    if mua_present and ripple_present:
        return 'ripple_mua'
    elif mua_present:
        return 'mua_only'
    elif ripple_present:
        return 'ripple_only'
    return 'unknown'


def interactive_event_refinement_with_spikes(detector, mua_vec, spike_times_by_region, 
                                             region_colors=None, margin=0.5,
                                             global_states=None, bin_size=None, show_global_states=False):
    """
    Interactive widget to browse events with spike raster plots showing MUA-refined boundaries.
    
    Args:
        detector: SWRHMMDetector with detected events (must have mua_high_start/mua_high_end)
        mua_vec: Full MUA vector
        spike_times_by_region: Dict of {region: {unit_id: spike_times_array}}
        region_colors: Dict of {region: color} (optional)
        margin: Time margin around event (seconds)
    """
    from ipywidgets import IntSlider, FloatSlider, Button, HBox, VBox, Output
    from IPython.display import display
    
    # Post-process events to prevent overlaps after MUA refinement
    def resolve_event_overlaps(events):
        """Prevent MUA-refined boundaries from overlapping with next event's basic ripple start."""
        for i in range(len(events) - 1):
            curr = events[i]
            next_ev = events[i+1]
            
            # Get current event's MUA end (if present)
            curr_mua_end = curr.get('mua_high_end')
            
            # Get next event's basic ripple start (NOT MUA start, to avoid cascade)
            next_basic_start = next_ev.get('basic_start_time', next_ev.get('combined_start_time', next_ev.get('start_time')))
            
            # Only process if current event has MUA refinement
            if curr_mua_end is not None and next_basic_start is not None:
                # Check if MUA end overlaps with next event's ripple
                if curr_mua_end > next_basic_start:
                    # Truncate current event's MUA end to next event's basic start
                    curr['mua_high_end'] = next_basic_start
                    
                    # Check if truncation resulted in invalid MUA duration
                    mua_start = curr.get('mua_high_start')
                    if mua_start is not None:
                        if curr['mua_high_end'] <= mua_start:
                            # Invalid duration - clear MUA refinement for this event
                            curr['mua_high_start'] = None
                            curr['mua_high_end'] = None
                            print(f"  Warning: Event {curr.get('event_id', i)} MUA refinement removed due to overlap with next event's ripple")
        return events

    # Overlap resolution removed: each event's MUA-refined duration is now estimated independently.
    n_events = len(detector.swr_events)
    if n_events == 0:
        print("No events to display")
        return

    # Add a slider for margin (window expansion)
    margin_slider = FloatSlider(value=margin, min=0.1, max=5.0, step=0.05, description='Window (s):', continuous_update=False)
    
    # Default colors
    if region_colors is None:
        region_colors = {'CA1': '#FF6B6B', 'RTC': '#4ECDC4', 'PFC': '#45B7D1'}
    
    # Prefer explicit global_states/bin_size; otherwise try detector-stored values
    if global_states is None and hasattr(detector, 'global_states'):
        global_states = getattr(detector, 'global_states')
    if bin_size is None and hasattr(detector, 'mua_bin_size'):
        bin_size = getattr(detector, 'mua_bin_size')

    # Prepare binned MUA and high-state info if requested
    mua_binned_full = None
    high_state = None
    bin_samples = None
    if show_global_states and global_states is not None:
        try:
            if hasattr(detector, 'mua_binned'):
                mua_binned_full = getattr(detector, 'mua_binned')
            else:
                # require bin_size to compute binned MUA
                if bin_size is not None:
                    mua_binned_full = bin_mua(mua_vec, bin_size, detector.fs)
            bin_samples = int(bin_size * detector.fs) if bin_size is not None else None
            if mua_binned_full is not None:
                n_states = int(np.max(global_states) + 1) if global_states.size > 0 else 1
                state_means = [mua_binned_full[global_states == s].mean() if (global_states == s).any() else 0
                               for s in range(n_states)]
                high_state = int(np.argmax(state_means))
        except Exception:
            mua_binned_full = None
            high_state = None

    event_slider = IntSlider(value=0, min=0, max=n_events-1, description='Event:')
    prev_btn = Button(description='Previous')
    next_btn = Button(description='Next')
    out = Output()
    

    def plot_event(event_idx, margin_val=None):
        # Use the current value of the margin slider if not provided
        m = margin_val if margin_val is not None else margin_slider.value
        # Plot LFP, ripple power, MUA, and spike raster with peak & thresholds
        event = detector.swr_events[event_idx]

        # Classify the event
        event_type = classify_event(event)
        print(f"Event type: {event_type}")

        # Get event times (try several possible keys)
        basic_start = event.get('basic_start_time', event.get('start_time'))
        basic_end = event.get('basic_end_time', event.get('end_time'))
        ripple_start = event.get('combined_start_time', basic_start)
        ripple_end = event.get('combined_end_time', basic_end)
        mua_start = event.get('mua_high_start')
        mua_end = event.get('mua_high_end')
        peak_time = event.get('peak_time', (ripple_start + ripple_end) / 2 if (ripple_start is not None and ripple_end is not None) else 0)

        # Get LFP data (support dict or array)
        if isinstance(detector.lfp_data, dict):
            lfp_full = list(detector.lfp_data.values())[0]
        else:
            lfp_full = detector.lfp_data if detector.lfp_data.ndim == 1 else np.mean(detector.lfp_data, axis=0)

        # Window around event
        window_start = max(0, (basic_start if basic_start is not None else peak_time) - m)
        window_end = min(len(lfp_full) / detector.fs, (basic_end if basic_end is not None else peak_time) + m)
        start_idx = int(window_start * detector.fs)
        end_idx = int(window_end * detector.fs)
        if end_idx <= start_idx:
            # safe fallback
            start_idx = max(0, int((peak_time - m) * detector.fs))
            end_idx = min(len(lfp_full), int((peak_time + m) * detector.fs))

        t_trace = np.arange(start_idx, end_idx) / detector.fs
        lfp = lfp_full[start_idx:end_idx]
        mua = mua_vec[start_idx:end_idx] if mua_vec is not None else np.zeros_like(lfp)

        # Ripple power: use detector.ripple_power if available, else compute locally
        ripple_power = None
        used_global_ripple = False
        if hasattr(detector, 'ripple_power') and detector.ripple_power is not None:
            try:
                rp_full = detector.ripple_power
                ripple_power = rp_full[start_idx:end_idx]
                used_global_ripple = True
            except Exception:
                ripple_power = None

        if ripple_power is None:
            # compute bandpassed envelope**2
            try:
                from scipy.signal import butter, filtfilt, hilbert
                b, a = butter(4, [detector.params.ripple_band[0] / (detector.fs / 2),
                                  detector.params.ripple_band[1] / (detector.fs / 2)], btype='band')
                filtered = filtfilt(b, a, lfp)
                env = np.abs(hilbert(filtered))
                ripple_power = env ** 2
            except Exception:
                # fallback: smooth absolute LFP as a proxy
                ripple_power = np.abs(lfp)

        # Thresholds for ripple power (use detector.params when available)
        mean_rp = np.nanmean(detector.ripple_power) if (hasattr(detector, 'ripple_power') and detector.ripple_power is not None) else np.nanmean(ripple_power)
        std_rp = np.nanstd(detector.ripple_power) if (hasattr(detector, 'ripple_power') and detector.ripple_power is not None) else np.nanstd(ripple_power)
        th_mult = getattr(detector.params, 'threshold_multiplier', None) if hasattr(detector, 'params') else None
        if th_mult is None:
            try:
                th_mult = detector.threshold_multiplier
            except Exception:
                th_mult = 3.0
        ripple_high_th = mean_rp + float(th_mult) * std_rp if (not np.isnan(mean_rp) and not np.isnan(std_rp)) else None
        ripple_low_th = None
        if hasattr(detector, 'params') and getattr(detector.params, 'use_hysteresis', False):
            low_mult = getattr(detector.params, 'hysteresis_low_multiplier', None)
            if low_mult is not None:
                ripple_low_th = mean_rp + float(low_mult) * std_rp

        # Get hysteresis times once for later use
        hyst_start = event.get('hysteresis_start_time')
        hyst_end = event.get('hysteresis_end_time')

        # Spike raster preparation
        unit_counter = 0
        yticks_pos = []
        yticks_labels = []
        spikes_plot_data = []
        for region, units_dict in spike_times_by_region.items():
            color = region_colors.get(region, 'gray')
            for unit_id, spike_times in units_dict.items():
                mask = (spike_times >= window_start) & (spike_times <= window_end)
                spikes_in_window = spike_times[mask]
                spikes_plot_data.append((unit_counter, spikes_in_window, color, f'{region}_{unit_id}'))
                yticks_pos.append(unit_counter)
                yticks_labels.append(f'{region}_{unit_id}')
                unit_counter += 1

        with out:
            out.clear_output(wait=True)
            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

            # Panel 1: Raw LFP with BASIC ripple duration and peak
            axs[0].plot(t_trace, lfp, color='k', lw=0.8, label='LFP')
            if basic_start is not None and basic_end is not None:
                axs[0].axvspan(basic_start, basic_end, color='cyan', alpha=0.25)
                # Calculate and display basic duration in legend
                basic_duration_ms = (basic_end - basic_start) * 1000
                basic_legend = f'Basic ripple ({basic_duration_ms:.1f} ms)'
            else:
                # fallback shading
                axs[0].axvspan(ripple_start, ripple_end, color='cyan', alpha=0.15)
                basic_legend = 'Basic ripple'
            
            # Only show peak line for non-mua_only events
            if event_type != 'mua_only':
                axs[0].axvline(peak_time, color='orange', ls='--', lw=1.2, label='Peak')
            
            axs[0].set_ylabel('LFP (µV)', fontweight='bold')
            
            # Update title to include event classification
            title = f'Event {event_idx+1}/{n_events} | {event_type.upper()} | Peak: {peak_time:.3f}s'
            axs[0].set_title(title, fontweight='bold')
            
            # Add legend manually to ensure both entries are shown
            legend_elements_1 = [
                mlines.Line2D([0], [0], color='k', lw=0.8, label='LFP'),
                mpatches.Rectangle((0,0),1,1, facecolor='cyan', alpha=0.25, label=basic_legend),
            ]
            if event_type != 'mua_only':
                legend_elements_1.append(mlines.Line2D([0], [0], color='orange', ls='--', lw=1.2, label='Peak'))
            
            axs[0].legend(handles=legend_elements_1, loc='upper right', fontsize=8)
            axs[0].spines['top'].set_visible(False); axs[0].spines['right'].set_visible(False)

            # Panel 2: Ripple power with hysteresis and PEAK (always shows all details if present)
            axs[1].plot(t_trace, ripple_power, color='purple', lw=0.9, label='Ripple power')

            # Always show thresholds if present
            if ripple_high_th is not None:
                axs[1].axhline(ripple_high_th, color='red', ls='--', lw=1, label=f'High thresh ({th_mult}×SD)')
            if ripple_low_th is not None:
                axs[1].axhline(ripple_low_th, color='blue', ls='--', lw=1, label='Low thresh (hysteresis)')

            # Always show hysteresis shading if present
            if hyst_start is not None and hyst_end is not None:
                axs[1].axvspan(hyst_start, hyst_end, color='blue', alpha=0.15, label=f'Hysteresis ({(hyst_end-hyst_start)*1000:.1f} ms)')

            # Always show stored peak power if present
            stored_peak_power = event.get('peak_power')
            if stored_peak_power is not None:
                axs[1].plot(peak_time, stored_peak_power, marker='o', color='red', markersize=7, 
                            label=f'Stored peak power: {stored_peak_power:.3f}')
            # If no stored peak, fallback to window peak
            elif len(ripple_power) > 0:
                rel_peak_idx = int(np.nanargmax(ripple_power))
                rel_peak_time = t_trace[rel_peak_idx]
                rel_peak_val = ripple_power[rel_peak_idx]
                axs[1].plot(rel_peak_time, rel_peak_val, marker='o', color='red', markersize=7, label='Ripple peak (window)')

            axs[1].set_ylabel('Ripple power', fontweight='bold')
            axs[1].legend(loc='upper right', fontsize=8)
            axs[1].spines['top'].set_visible(False); axs[1].spines['right'].set_visible(False)

            # Panel 3: MUA with MUA refined shading and peak (conditional based on event type)
            axs[2].plot(t_trace, mua, color='green', lw=0.8, label='MUA')
            

            # Always show MUA threshold if available
            mua_th = None
            try:
                mua_mult = getattr(detector.params, 'mua_threshold_multiplier', None)
            except Exception:
                mua_mult = None
            if mua_mult is None:
                try:
                    mua_mult = detector.mua_threshold_multiplier
                except Exception:
                    mua_mult = None
            if mua is not None and len(mua) > 0 and mua_mult is not None:
                mua_th = np.nanmean(mua_vec) + float(mua_mult) * np.nanstd(mua_vec)
                axs[2].axhline(mua_th, color='orange', ls='--', lw=1, label='MUA thresh')
            if mua_start is not None and mua_end is not None:
                axs[2].axvspan(mua_start, mua_end, color='red', alpha=0.2, label='MUA refined')
            # Use stored MUA peak time and amplitude if available
            mua_peak_time = event.get('mua_peak_time', None)
            mua_peak_amplitude = event.get('mua_peak_amplitude', None)
            if mua_peak_time is not None and mua_peak_amplitude is not None:
                axs[2].plot(mua_peak_time, mua_peak_amplitude, marker='D', color='magenta', markersize=7, label='MUA peak (stored)')
            elif len(mua) > 0:
                mua_rel_peak_idx = int(np.nanargmax(mua))
                mua_rel_peak_time = t_trace[mua_rel_peak_idx]
                mua_rel_peak_val = mua[mua_rel_peak_idx]
                axs[2].plot(mua_rel_peak_time, mua_rel_peak_val, marker='D', color='magenta', markersize=7, label='MUA peak (window)')

            # Always overlay global HMM high-state as a dashed step-line when requested
            if show_global_states and (global_states is not None) and (mua_binned_full is not None) and (high_state is not None) and (bin_samples is not None):
                try:
                    # Map window sample indices to bin indices
                    start_bin = int(start_idx // bin_samples)
                    end_bin = int(np.ceil(end_idx / bin_samples))
                    # Clip range
                    start_bin = max(0, start_bin)
                    end_bin = min(len(global_states), end_bin)
                    if end_bin > start_bin:
                        bin_idx = np.arange(start_bin, end_bin)
                        # use bin centers for plotting
                        bs = float(bin_size) if bin_size is not None else 0.005
                        times = bin_idx * bs + 0.5 * bs
                        is_high = (global_states[start_bin:end_bin] == high_state).astype(float)
                        # scale the state line to near the top of the MUA panel
                        if np.all(np.isnan(mua)) or mua.size == 0:
                            y_vals = is_high
                        else:
                            y_max = np.nanmax(mua)
                            y_min = np.nanmin(mua)
                            span = y_max - y_min if (y_max - y_min) != 0 else 1.0
                            y_offset = y_min + 0.9 * span
                            y_vals = is_high * y_offset
                        axs[2].step(times, y_vals, where='mid', color='k', linestyle='--', linewidth=1.2, label='Global HMM state')
                except Exception:
                    pass

            axs[2].set_ylabel('MUA (a.u.)', fontweight='bold')
            axs[2].legend(loc='upper right', fontsize=8)
            axs[2].spines['top'].set_visible(False); axs[2].spines['right'].set_visible(False)

            # Panel 4: Spike raster with appropriate shading + peak
            for unit_idx, spk_times, color, label in spikes_plot_data:
                if spk_times.size > 0:
                    axs[3].vlines(spk_times, unit_idx - 0.4, unit_idx + 0.4, color=color, linewidth=1.2)
            
            # Add appropriate shading and legends based on event type
            legend_elements_4 = []
            
            if event_type == 'mua_only' and mua_start is not None and mua_end is not None:
                mua_duration_ms = (mua_end - mua_start) * 1000
                mua_legend = f'MUA only ({mua_duration_ms:.1f} ms)'
                legend_elements_4.append(mpatches.Rectangle((0,0),1,1, facecolor='red', alpha=0.15, label=mua_legend))
                # Only show MUA peak for mua_only events
                legend_elements_4.append(mlines.Line2D([0], [0], color='orange', ls='--', lw=1, label='MUA peak'))
                
            elif event_type == 'ripple_only' and ripple_start is not None and ripple_end is not None:
                ripple_duration_ms = (ripple_end - ripple_start) * 1000
                ripple_legend = f'Ripple only ({ripple_duration_ms:.1f} ms)'
                legend_elements_4.append(mpatches.Rectangle((0,0),1,1, facecolor='blue', alpha=0.15, label=ripple_legend))
                # Only show ripple peak for ripple_only events
                legend_elements_4.append(mlines.Line2D([0], [0], color='orange', ls='--', lw=1, label='Ripple peak'))
                
            elif event_type == 'ripple_mua':
                if mua_start is not None and mua_end is not None:
                    mua_duration_ms = (mua_end - mua_start) * 1000
                    legend_elements_4.append(mpatches.Rectangle((0,0),1,1, facecolor='red', alpha=0.15, label=f'Ripple+MUA ({mua_duration_ms:.1f} ms)'))
                if ripple_start is not None and ripple_end is not None:
                    ripple_duration_ms = (ripple_end - ripple_start) * 1000
                    legend_elements_4.append(mpatches.Rectangle((0,0),1,1, facecolor='blue', alpha=0.15, label=f'Ripple+MUA ({ripple_duration_ms:.1f} ms)'))
                # Show both peaks for co-occur events
                legend_elements_4.append(mlines.Line2D([0], [0], color='orange', ls='--', lw=1, label='Peak'))
            
            if legend_elements_4:
                axs[3].legend(handles=legend_elements_4, loc='upper right', fontsize=8)
            else:
                axs[3].axvline(peak_time, color='orange', ls='--', lw=1, label='Peak')
                
            axs[3].set_yticks(yticks_pos)
            axs[3].set_yticklabels(yticks_labels, fontsize=7)
            axs[3].set_ylabel('Units', fontweight='bold')
            axs[3].set_xlabel('Time (s)', fontweight='bold')
            axs[3].spines['top'].set_visible(False); axs[3].spines['right'].set_visible(False)
            axs[3].set_ylim(-0.5, unit_counter - 0.5)

            plt.tight_layout()
            plt.show()


            # Print stats
            ripple_duration = None
            if ripple_start is not None and ripple_end is not None:
                ripple_duration = (ripple_end - ripple_start) * 1000.0
            mua_duration = None
            if mua_start is not None and mua_end is not None:
                mua_duration = (mua_end - mua_start) * 1000.0

            print(f"\nEvent {event_idx+1} stats ({event_type}):")
            if ripple_duration is not None:
                print(f"  Ripple (basic/combined) duration: {ripple_duration:.1f} ms")
            # Always show hysteresis bounds if present, even for mua_only
            if hyst_start is not None and hyst_end is not None:
                print(f"  Hysteresis bounds: {hyst_start:.3f}s - {hyst_end:.3f}s")
            if mua_duration is not None:
                print(f"  MUA refined duration: {mua_duration:.1f} ms")
            print(f"  Peak time: {peak_time:.3f}s")
            
            # Only show relevant peak information based on event type
            if event_type == 'mua_only':
                mua_peak_time = event.get('mua_peak_time', None)
                mua_peak_amplitude = event.get('mua_peak_amplitude', None)
                if mua_peak_time is not None and mua_peak_amplitude is not None:
                    print(f"  MUA peak amplitude (stored): {mua_peak_amplitude:.3f} at {mua_peak_time:.3f}s")
                elif len(mua) > 0:
                    mua_peak_val = np.nanmax(mua)
                    mua_peak_idx = int(np.nanargmax(mua))
                    mua_peak_time_window = t_trace[mua_peak_idx]
                    print(f"  MUA peak amplitude (window): {mua_peak_val:.3f} at {mua_peak_time_window:.3f}s")
            elif event_type == 'ripple_only':
                stored_peak_power = event.get('peak_power')
                if stored_peak_power is not None:
                    print(f"  Stored peak power: {stored_peak_power:.3f}")
            else:  # ripple_mua
                stored_peak_power = event.get('peak_power')
                if stored_peak_power is not None:
                    print(f"  Stored peak power: {stored_peak_power:.3f}")
                mua_peak_time = event.get('mua_peak_time', None)
                mua_peak_amplitude = event.get('mua_peak_amplitude', None)
                if mua_peak_time is not None and mua_peak_amplitude is not None:
                    print(f"  MUA peak amplitude (stored): {mua_peak_amplitude:.3f} at {mua_peak_time:.3f}s")
                elif len(mua) > 0:
                    mua_peak_val = np.nanmax(mua)
                    mua_peak_idx = int(np.nanargmax(mua))
                    mua_peak_time_window = t_trace[mua_peak_idx]
                    print(f"  MUA peak amplitude (window): {mua_peak_val:.3f} at {mua_peak_time_window:.3f}s")
    
    def on_prev(b):
        if event_slider.value > 0:
            event_slider.value -= 1
    
    def on_next(b):
        if event_slider.value < n_events - 1:
            event_slider.value += 1
    
    def on_slider_change(change):
        plot_event(change['new'])

    def on_margin_change(change):
        plot_event(event_slider.value, margin_val=change['new'])
    
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    event_slider.observe(on_slider_change, names='value')
    margin_slider.observe(on_margin_change, names='value')

    controls = HBox([prev_btn, next_btn, event_slider, margin_slider])
    display(VBox([controls, out]))
    plot_event(0)
