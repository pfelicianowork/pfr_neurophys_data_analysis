"""
feature_viz.py

Interactive visualization utilities for biological features extracted from
SWR events. Designed to work directly with the output of
`cnn_autoencoder.feature_extraction.batch_extract_features()` which returns
`feature_matrix` (n_events x n_features) and `feature_names` (list of strings).

Main entrypoint:
    visualize_features(feature_matrix, feature_names, events=None)

The widget shows:
 - Scatter plot for any selected X/Y feature with regression line and Pearson r
 - Correlation heatmap for all features
 - Single-feature boxplot + histogram with optional KDE

Requires: numpy, pandas, matplotlib, seaborn, ipywidgets
"""
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except Exception:
    widgets = None  # latent fallback; we'll raise if user tries to use interactive UI


def _ensure_df(feature_matrix, feature_names):
    """Return a pandas DataFrame from feature_matrix and feature_names.

    Accepts either a numpy ndarray or a pandas DataFrame. If ndarray,
    `feature_names` must be provided and match the column count.
    """
    if isinstance(feature_matrix, pd.DataFrame):
        return feature_matrix.copy()
    if isinstance(feature_matrix, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names required when feature_matrix is ndarray")
        if len(feature_names) != feature_matrix.shape[1]:
            raise ValueError("feature_names length must match number of columns in feature_matrix")
        return pd.DataFrame(feature_matrix, columns=list(feature_names))
    raise ValueError("feature_matrix must be a numpy.ndarray or pandas.DataFrame")


def _autodetect_label_keys(events):
    """Return candidate label/cluster keys found in events list of dicts."""
    if not events:
        return []
    keys = set().union(*(e.keys() for e in events))
    # prioritize keys containing 'cluster' or 'label'
    return [k for k in keys if 'cluster' in k.lower() or 'label' in k.lower()]


def visualize_features(feature_matrix, feature_names=None, events: Optional[list] = None, figsize=(8, 4)):
    """Launch an interactive widget for exploring extracted features.

    Parameters
    ----------
    feature_matrix : np.ndarray or pd.DataFrame
        shape (n_events, n_features)
    feature_names : list or None
        list of feature name strings (required if feature_matrix is ndarray)
    events : list of dict, optional
        If provided, used to subset events by label/cluster keys present in the dicts.
    figsize : tuple
        Matplotlib figure size

    Returns
    -------
    out : ipywidgets.Output
        Output widget containing the plots (also displayed inline).
    """
    if widgets is None:
        raise RuntimeError("ipywidgets not available. Install ipywidgets to use interactive visualization.")

    df = _ensure_df(feature_matrix, feature_names)
    cols = list(df.columns)

    # Widgets
    view_w = widgets.ToggleButtons(options=['scatter', 'heatmap', 'single'], description='View:')
    x_w = widgets.Dropdown(options=cols, description='X:')
    y_w = widgets.Dropdown(options=cols, description='Y:')
    single_w = widgets.Dropdown(options=cols, description='Feature:')
    bins_w = widgets.IntSlider(value=30, min=5, max=200, step=1, description='Bins')
    kde_w = widgets.Checkbox(value=False, description='KDE')
    clip_w = widgets.FloatSlider(value=0.0, min=0.0, max=10.0, step=0.1, description='Clip %')
    logx_w = widgets.Checkbox(value=False, description='log X')
    logy_w = widgets.Checkbox(value=False, description='log Y')
    label_keys = _autodetect_label_keys(events)
    label_key_w = widgets.Dropdown(options=[''] + label_keys, description='Label key:')
    label_val_w = widgets.Text(description='Label val:')
    export_btn = widgets.Button(description='Export CSV')
    out = widgets.Output()


    def _clip_series(s, pct):
        if pct <= 0:
            return s
        lo = np.nanpercentile(s, pct)
        hi = np.nanpercentile(s, 100.0 - pct)
        return s.clip(lo, hi)


    def _subset_df():
        data = df
        if events and label_key_w.value:
            key = label_key_w.value
            val = label_val_w.value
            # try numeric cast
            vnum = None
            if val != '':
                try:
                    vnum = float(val)
                except Exception:
                    vnum = None
            idx = [i for i, e in enumerate(events) if (e.get(key) == vnum if vnum is not None else e.get(key) == val)]
            if len(idx) == 0:
                return df.iloc[[]]
            return df.iloc[idx]
        return df


    def _on_export(_):
        sel = _subset_df()
        sel.to_csv('selected_features_export.csv', index=False)
        with out:
            print("Exported selected_features_export.csv")


    def _plot(_=None):
        with out:
            clear_output(wait=True)
            view = view_w.value
            data = _subset_df()
            pct = clip_w.value
            if view == 'scatter':
                x = _clip_series(data[x_w.value], pct).dropna()
                y = _clip_series(data[y_w.value], pct).dropna()
                # align indices
                common_idx = x.index.intersection(y.index)
                x = x.loc[common_idx]
                y = y.loc[common_idx]
                if logx_w.value:
                    x = np.log1p(x - x.min() + 1e-9)
                if logy_w.value:
                    y = np.log1p(y - y.min() + 1e-9)

                # Marginal histograms layout using GridSpec (compact)
                # Use bins and KDE controls from widgets
                bins = bins_w.value

                # compute explicit bin edges aligned with data min/max so histograms
                # always reflect the same range shown in the scatter plot
                x_min, x_max = np.nanmin(x), np.nanmax(x)
                y_min, y_max = np.nanmin(y), np.nanmax(y)
                # guard against constant values
                if x_min == x_max:
                    x_min -= 0.5
                    x_max += 0.5
                if y_min == y_max:
                    y_min -= 0.5
                    y_max += 0.5

                # compute a 10% padding around data ranges so points and bars
                # near the edges are fully visible
                x_range = x_max - x_min
                y_range = y_max - y_min
                xpad = 0.1 * x_range
                ypad = 0.1 * y_range
                # in case of extremely small ranges, ensure non-zero pad
                if xpad == 0:
                    xpad = 0.1
                if ypad == 0:
                    ypad = 0.1

                # bins computed from raw data extents (so hist counts reflect data),
                # axis limits will be extended by padding for visibility
                xbins = np.linspace(x_min, x_max, bins + 1)
                ybins = np.linspace(y_min, y_max, bins + 1)

                x_min_pad = x_min - xpad
                x_max_pad = x_max + xpad
                y_min_pad = y_min - ypad
                y_max_pad = y_max + ypad

                fig = plt.figure(figsize=figsize)
                # width_ratios: main scatter wide, right histogram narrow
                # height_ratios: top histogram short, main scatter tall
                gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                      wspace=0.05, hspace=0.05)
                ax_scatter = fig.add_subplot(gs[1, 0])
                ax_xhist = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
                ax_yhist = fig.add_subplot(gs[1, 1])

                # Scatter + regression
                sns.scatterplot(x=x, y=y, ax=ax_scatter, s=24, alpha=0.75)
                try:
                    r = np.corrcoef(x.fillna(0), y.fillna(0))[0, 1]
                except Exception:
                    r = np.nan
                sns.regplot(x=x, y=y, scatter=False, ax=ax_scatter,
                            line_kws={'color': 'r', 'alpha': 0.7})
                ax_scatter.set_xlabel(x_w.value)
                ax_scatter.set_ylabel(y_w.value)
                ax_scatter.set_title(f"{x_w.value} vs {y_w.value} (r={np.nan_to_num(r):.3f})")

                # enforce same axis limits so marginals map to scatter extents
                # use padded limits so points/bars at the edges are fully visible
                ax_scatter.set_xlim(x_min_pad, x_max_pad)
                ax_scatter.set_ylim(y_min_pad, y_max_pad)


                # Top histogram (X) - compute counts explicitly and draw bars so
                # axis alignment and extents exactly match the scatter plot
                x_counts, _ = np.histogram(x, bins=xbins)
                # bar centers and width
                x_centers = 0.5 * (xbins[:-1] + xbins[1:])
                x_width = xbins[1] - xbins[0]
                ax_xhist.bar(x_centers, x_counts, width=x_width, color='gray', align='center')
                # match x histogram to padded scatter x-limits so bars at edges are visible
                ax_xhist.set_xlim(x_min_pad, x_max_pad)
                ax_xhist.set_ylabel('')
                ax_xhist.set_xlabel('')
                ax_xhist.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                ax_xhist.tick_params(axis='y', which='both', left=False, labelleft=False)

                # Right histogram (Y) - compute counts and draw horizontal bars
                y_counts, _ = np.histogram(y, bins=ybins)
                y_centers = 0.5 * (ybins[:-1] + ybins[1:])
                y_height = ybins[1] - ybins[0]
                ax_yhist.barh(y_centers, y_counts, height=y_height, color='gray', align='center')
                # match y histogram to padded scatter y-limits so bars at edges are visible
                ax_yhist.set_ylim(y_min_pad, y_max_pad)
                # limit x to show full counts with small padding
                if len(y_counts) > 0:
                    ax_yhist.set_xlim(0, np.max(y_counts) * 1.05)
                ax_yhist.set_xlabel('')
                ax_yhist.set_ylabel('')
                ax_yhist.tick_params(axis='y', which='both', left=False, labelleft=False)

                # Improve layout: remove overlapping tick labels on histograms
                plt.setp(ax_xhist.get_xticklabels(), visible=False)
                plt.setp(ax_yhist.get_yticklabels(), visible=False)

                plt.show()
            elif view == 'heatmap':
                corr = data.corr(method='pearson')
                fig, ax = plt.subplots(figsize=(min(12, 0.25 * len(cols) + 3), min(12, 0.25 * len(cols) + 3)))
                sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0)
                ax.set_title('Feature Pearson Correlation')
                plt.show()
            elif view == 'single':
                s = _clip_series(data[single_w.value], pct).dropna()
                fig, axs = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] / 2))
                sns.boxplot(x=s, ax=axs[0])
                axs[0].set_title(f"Boxplot: {single_w.value}")
                sns.histplot(s, bins=bins_w.value, kde=kde_w.value, ax=axs[1])
                axs[1].set_title(f"Histogram: {single_w.value}")
                plt.show()
            else:
                print("Unknown view")


    # Observers
    for w in (view_w, x_w, y_w, single_w, bins_w, kde_w, clip_w, logx_w, logy_w, label_key_w, label_val_w):
        w.observe(_plot, names='value')
    export_btn.on_click(_on_export)

    controls_top = widgets.HBox([view_w, x_w, y_w, single_w])
    controls_mid = widgets.HBox([bins_w, kde_w, clip_w, logx_w, logy_w])
    controls_bot = widgets.HBox([label_key_w, label_val_w, export_btn])
    display(widgets.VBox([controls_top, controls_mid, controls_bot, out]))
    # initial plot
    _plot()
    return out
