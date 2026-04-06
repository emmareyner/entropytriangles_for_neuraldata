"""General helpers used across the project.

This file groups together:
- data loading helpers
- virtual worm construction
- context similarity and regime detection
- small plotting helpers for regime summaries

The similarity and regime-detection functions are re-exported from
``similarities.py`` so that previous imports from ``utils`` keep working.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.patches import Patch

from similarities import (
    vec01,
    jaccard,
    activation_density,
    mean_pairwise_similarity,
    mean_cross_similarity,
    jaccard_triangle_context,
    mean_cross_Delta,
    delta_confidence,
    boundary_score_at_split,
    boundary_score_distribution,
    baseline_threshold,
    centroid_profile,
    best_centroid_in_interval,
    evaluate_centroid_candidates,
    best_split_between,
    compute_best_splits_for_centroids,
    add_baseline_to_splits,
    medoid_phi_in_regime,
    build_perception_regimes,
    dice_similarity_flat,
    dice_context,
    boundary_score_at_split_dice,
    boundary_score_distribution_dice,
    best_split_between_dice,
    absolute_threshold_dice,
    best_split_with_significance_dice,
    recursive_splits_dice,
    absolute_threshold_jaccard,
    best_split_with_significance_jaccard,
    recursive_splits_jaccard
)


def load_context_from_file(filename, CONTEXT_DIR):
    filename = str(filename)
    path = CONTEXT_DIR / filename
    match = re.search('_([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\.csv$', filename)
    if not match:
        raise ValueError(f"Could not extract phi from file name: '{filename}'")
    phi = float(match.group(1))
    df = pd.read_csv(path, sep=';', header=0)
    if df.columns[0] == '':
        df = df.drop(columns=df.columns[0])
    if df.columns[-1] == '':
        df = df.drop(columns=df.columns[-1])
    df = df.set_index(df.columns[0])
    odorants = df.index.tolist()
    neurons = df.columns.tolist()
    C_bin = df.to_numpy(dtype=int)
    return (C_bin, odorants, neurons, phi)

def load_compiled_data(mat_path):
    mat_path = Path(mat_path)
    mat = sio.loadmat(mat_path)
    conc = mat_path.stem.split('_')[-1]
    if conc == '4':
        ap_time = mat['ap_time'][0]
        compiled_LR_ordered = mat['compiled_LR_ordered'][0]
        neuron_pairs_raw = mat['neuron_pairs'][0]
        odor_list_raw = mat['odor_list'][:, 0]
        Ns = mat['Ns']
    elif conc == '5':
        ap_time = mat['ap_time'][0]
        compiled_LR_ordered = mat['compiled_LR_5_ordered'][0]
        neuron_pairs_raw = mat['neuron_pairs'][0]
        odor_list_raw = mat['odor_list'][:, 0]
        Ns = mat['Ns_5']
    elif conc == '6':
        ap_time = mat['ap_time'][0]
        compiled_LR_ordered = mat['compiled_LR_6_ordered'][0]
        neuron_pairs_raw = mat['neuron_pairs'][0]
        odor_list_raw = mat['odor_list'][:, 0]
        Ns = mat['Ns_6']
    neuron_names = [str(neuron_pairs_raw[j][0]) for j in range(len(neuron_pairs_raw))]
    odor_names = [str(odor_list_raw[i][0]) for i in range(len(odor_list_raw))]
    min_samples = int(Ns[Ns > 0].min())
    return {'ap_time': ap_time, 'compiled_LR_ordered': compiled_LR_ordered, 'neuron_names': neuron_names, 'odor_names': odor_names, 'Ns': Ns, 'min_samples': min_samples}

def parse_phi_from_filename(path: str) -> float:
    name, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() != '.csv':
        raise ValueError(f'Unexpected file extension: {path}')
    phi_str = name.rsplit('_', 1)[-1]
    return float(phi_str)

def read_context_matrix_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', index_col=0)
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df = (df.astype(float) > 0).astype(np.uint8)
    return df

def load_contexts(folder: str, pattern='*.csv') -> dict[float, pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f'No CSV files were found in {folder} with pattern={pattern}')
    contexts = {}
    for p in paths:
        phi = parse_phi_from_filename(p)
        contexts[phi] = read_context_matrix_csv(p)
    return dict(sorted(contexts.items(), key=lambda kv: kv[0]))

def align_contexts(contexts: dict[float, pd.DataFrame]) -> dict[float, pd.DataFrame]:
    mats = list(contexts.values())
    all_rows = sorted(set().union(*[set(m.index) for m in mats]))
    all_cols = sorted(set().union(*[set(m.columns) for m in mats]))
    out = {}
    for phi, m in contexts.items():
        out[phi] = m.reindex(index=all_rows, columns=all_cols, fill_value=0).astype(np.uint8)
    return out

def build_odor_neuron_structs(compiled_LR_ordered, neuron_names):
    num_odors = len(compiled_LR_ordered)
    neuron_name_to_idx = {name: j for j, name in enumerate(neuron_names)}
    odor_neuron_structs = []
    for oi in range(num_odors):
        arr = compiled_LR_ordered[oi]
        mapping = {}
        for s in arr[0]:
            nm = str(s['neuronpair'][0])
            if nm in neuron_name_to_idx:
                mapping[nm] = s
        if len(mapping) != len(neuron_names):
            raise ValueError(f'Odor index {oi} does not contain the expected {len(neuron_names)} neuron pairs.')
        odor_neuron_structs.append(mapping)
    return (odor_neuron_structs, neuron_name_to_idx)

def compute_window_radius(ap_time, window_seconds=2.0):
    dt = float(np.median(np.diff(ap_time)))
    window_radius = int(round(window_seconds / dt))
    return (window_radius, dt)

def compute_peak_value(trace, window_radius):
    trace = np.asarray(trace).reshape(-1)
    if np.isnan(trace).all():
        return np.nan
    idx_peak = int(np.nanargmax(np.abs(trace)))
    start = max(0, idx_peak - window_radius)
    end = min(trace.shape[0], idx_peak + window_radius + 1)
    window = trace[start:end]
    if np.isnan(window).all():
        return np.nan
    return float(np.nanmean(window))

def build_virtual_worm_traces(odor_neuron_structs, neuron_name_to_idx, ap_time, min_samples):
    num_odors = len(odor_neuron_structs)
    num_neurons = len(neuron_name_to_idx)
    n_timepoints = ap_time.shape[0]
    virtual_traces = np.zeros((min_samples, num_odors, num_neurons, n_timepoints), dtype=float)
    for w in range(min_samples):
        for oi in range(num_odors):
            mapping = odor_neuron_structs[oi]
            for neuron_name, j in neuron_name_to_idx.items():
                s = mapping[neuron_name]
                traces = s['all_traces']
                if traces.shape[0] < min_samples:
                    raise ValueError(f'Odor {oi}, neuron {neuron_name}: {traces.shape[0]} repetitions (< {min_samples}).')
                virtual_traces[w, oi, j, :] = traces[w, :]
    return virtual_traces

def build_random_virtual_worms(odor_neuron_structs, neuron_name_to_idx, ap_time, n_worms, seed=None, store_indices=False):
    rng = np.random.default_rng(seed)
    n_odors = len(odor_neuron_structs)
    n_neurons = len(neuron_name_to_idx)
    n_timepoints = ap_time.shape[0]
    virtual_traces = np.zeros((n_worms, n_odors, n_neurons, n_timepoints))
    index_map = np.zeros((n_worms, n_odors, n_neurons), dtype=int)
    for w in range(n_worms):
        for oi in range(n_odors):
            mapping = odor_neuron_structs[oi]
            for neuron_name, j in neuron_name_to_idx.items():
                traces = mapping[neuron_name]['all_traces']
                N_ij = traces.shape[0]
                k = rng.integers(0, N_ij)
                virtual_traces[w, oi, j, :] = traces[k, :]
                index_map[w, oi, j] = k
    return (virtual_traces, index_map) if store_indices else virtual_traces

def compute_formal_contexts_from_traces(virtual_traces, ap_time, window_seconds=2.0):
    n_worms, n_odors, n_neurons, _ = virtual_traces.shape
    window_radius, _ = compute_window_radius(ap_time, window_seconds)
    contexts = np.zeros((n_worms, n_odors, n_neurons), dtype=float)
    for w in range(n_worms):
        for oi in range(n_odors):
            for j in range(n_neurons):
                trace = virtual_traces[w, oi, j, :]
                contexts[w, oi, j] = compute_peak_value(trace, window_radius)
    return contexts

def binarize_contexts(contexts, phi=0.256, mode='>='):
    if mode == '>=':
        return (contexts >= phi).astype(int)
    elif mode == '<=':
        return (contexts <= phi).astype(int)
    else:
        raise ValueError("mode debe ser '>=' o '<='.")

def phi_to_percent(phi, phi_min, phi_max):
    return 100.0 * (phi - phi_min) / (phi_max - phi_min)

def phi_to_percent_log(phi, phi_min, phi_max, base=np.e):
    phi = np.asarray(phi, dtype=float)
    phi = np.maximum(phi, phi_min)
    if base == 10:
        num = np.log10(phi / phi_min)
        den = np.log10(phi_max / phi_min)
    else:
        num = np.log(phi / phi_min)
        den = np.log(phi_max / phi_min)
    return 100.0 * (num / den)

def add_conc(df, conc_label):
    out = df.copy()
    out['conc'] = conc_label
    return out

def normalize_conc_label(x):
    s = str(x).strip().lower().replace(' ', '')
    mapping = {'1e-4': '1e-4', '10^-4': '1e-4', '10^{-4}': '1e-4', '104': '1e-4', '4': '1e-4', '1e-5': '1e-5', '10^-5': '1e-5', '10^{-5}': '1e-5', '105': '1e-5', '5': '1e-5', '1e-6': '1e-6', '10^-6': '1e-6', '10^{-6}': '1e-6', '106': '1e-6', '6': '1e-6'}
    return mapping.get(s, s)

def plot_regime_bars(df_all, conc_order=None, xlim=(0, 100), title=None, scale="log", ax=None):
    if scale == "log":
        start_col = "pL_start"
        end_col = "pL_end"
        width_col = "pL_width"
        medoid_col = "pL_medoid"
        xlabel = "ϕ progression (Log %)"
    elif scale == "linear":
        start_col = "p_start"
        end_col = "p_end"
        width_col = "p_width"
        medoid_col = "p_medoid"
        xlabel = "ϕ progression (%)"
    else:
        raise ValueError("scale must be either 'log' or 'linear'")
    
    df = df_all.copy()
    df[end_col] = df[end_col].clip(upper=xlim[1])
    df[start_col] = df[start_col].clip(lower=xlim[0])
    df[width_col] = df[end_col] - df[start_col]
    if conc_order is None:
        conc_order = sorted(df['conc'].unique())
    ypos = np.arange(len(conc_order))
    df['conc'], conc_order = (df['conc'].replace({'1e-4': '$10^{-4}$', '1e-5': '$10^{-5}$', '1e-6': '$10^{-6}$'}), ['$10^{-4}$' if str(c).strip() == '1e-4' else '$10^{-5}$' if str(c).strip() == '1e-5' else '$10^{-6}$' if str(c).strip() == '1e-6' else c for c in conc_order])
    df = df.sort_values(['conc', start_col, 'regime']).reset_index(drop=True)
    for conc, idx in df.groupby('conc').groups.items():
        rows = df.loc[idx].sort_values(start_col).index.tolist()
        prev_end = float(xlim[0])
        for r in rows:
            df.loc[r, start_col] = prev_end
            df.loc[r, width_col] = df.loc[r, end_col] - df.loc[r, start_col]
            prev_end = float(df.loc[r, end_col])
    regimes = sorted(df['regime'].unique())
    cmap = plt.get_cmap('tab10')
    color_map = {reg: cmap((reg - 1) % 10) for reg in regimes}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, conc in enumerate(conc_order):
        d = df[df['conc'] == conc].sort_values(start_col)
        for _, row in d.iterrows():
            ax.barh(y=i, width=row[width_col], left=row[start_col], height=0.55, color=color_map[row['regime']], edgecolor='white', linewidth=1.0)
            y = i
            ax.plot([row[start_col], row[start_col]], [y - 0.32, y + 0.32], linewidth=1.2)
            ax.plot([row[end_col], row[end_col]], [y - 0.32, y + 0.32], linewidth=1.2)
            ax.scatter([row[medoid_col]], [y], marker='D', facecolors='white', edgecolors='black', s=28, zorder=5)
            reg = int(row['regime'])
            if row[start_col] > xlim[0] + 1e-09:
                ax.text(row[start_col], y + 0.38, f"{row[start_col]:.1f}", ha='center', va='bottom', fontsize=9, zorder=20)
            if row[end_col] < xlim[1] - 1e-09:
                ax.text(row[end_col], y + 0.38, f"{row[end_col]:.1f}", ha='center', va='bottom', fontsize=9, zorder=20)
            ax.annotate(f"{row[medoid_col]:.1f}", xy=(row[medoid_col], y), xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', fontsize=8, zorder=20)
    ax.set_xlim(*xlim)
    ax.set_yticks(ypos, conc_order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Concentration')
    ax.set_title(title or 'Perception regimes along continuous ϕ (stacked by regime)', pad=25)
    ax.grid(axis='x', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_handles = [Patch(facecolor=color_map[r], edgecolor='none', label=f'Regime {r}') for r in regimes]
    ax.legend(handles=legend_handles, ncol=1, frameon=True, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
