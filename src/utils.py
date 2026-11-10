"""General helpers used across the project.

This file groups together:
- data loading helpers (Lin et al. 2023 compiled .mat files, K-FCA context CSVs)
- the graded odorant-neuron matrix R(c) used in the K-FCA analysis
- virtual worm construction (single-presentation traces, sampled without replacement)
- thresholding of graded contexts for the two K-FCA branches
- small plotting/summary helpers
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
    """Load one ``compiled_data_conc_X.mat`` file of Lin et al. (2023).

    The variable names inside the file depend on the concentration
    (``compiled_LR_ordered`` / ``Ns`` for 10^-4, ``compiled_LR_5_ordered`` / ``Ns_5``
    for 10^-5, ...); they are detected automatically.
    """
    mat_path = Path(mat_path)
    mat = sio.loadmat(mat_path)
    conc = mat_path.stem.split('_')[-1]
    suffix = '' if conc == '4' else f'_{conc}'
    lr_key = f'compiled_LR{suffix}_ordered'
    ns_key = f'Ns{suffix}'
    if lr_key not in mat or ns_key not in mat:
        raise KeyError(f"{mat_path.name}: expected variables '{lr_key}' and '{ns_key}'.")
    ap_time = mat['ap_time'][0]
    compiled_LR_ordered = mat[lr_key][0]
    neuron_pairs_raw = mat['neuron_pairs'][0]
    odor_list_raw = mat['odor_list'][:, 0]
    Ns = mat[ns_key]
    neuron_names = [str(neuron_pairs_raw[j][0]) for j in range(len(neuron_pairs_raw))]
    odor_names = [str(odor_list_raw[i][0]) for i in range(len(odor_list_raw))]
    min_samples = int(Ns[Ns > 0].min())
    return {'ap_time': ap_time, 'compiled_LR_ordered': compiled_LR_ordered, 'neuron_names': neuron_names, 'odor_names': odor_names, 'Ns': Ns, 'min_samples': min_samples, 'concentration': int(conc)}

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
    """Assemble ``n_worms`` virtual worms from single-presentation traces.

    For every odorant-neuron pair, ``n_worms`` *distinct* traces are drawn at once
    (sampling without replacement) and trace k is assigned to virtual worm k, so no
    trace is used in more than one virtual worm. With ``n_worms`` equal to the
    smallest number of traces of any pair (``min_samples``) every pair can be served.
    A virtual worm is not a biological individual: different entries of the same
    matrix generally come from different presentations and animals.
    """
    rng = np.random.default_rng(seed)
    n_odors = len(odor_neuron_structs)
    n_neurons = len(neuron_name_to_idx)
    n_timepoints = ap_time.shape[0]
    virtual_traces = np.zeros((n_worms, n_odors, n_neurons, n_timepoints))
    index_map = np.zeros((n_worms, n_odors, n_neurons), dtype=int)
    for oi in range(n_odors):
        mapping = odor_neuron_structs[oi]
        for neuron_name, j in neuron_name_to_idx.items():
            traces = mapping[neuron_name]['all_traces']
            N_ij = traces.shape[0]
            if N_ij < n_worms:
                raise ValueError(f'Odor {oi}, neuron {neuron_name}: {N_ij} traces < {n_worms} worms.')
            ks = rng.choice(N_ij, n_worms, replace=False)
            virtual_traces[:, oi, j, :] = traces[ks, :]
            index_map[:, oi, j] = ks
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
        raise ValueError("mode must be '>=' or '<='.")

BRANCHES = {
    # K-FCA branch -> (binarization mode, part of the phi axis analysed)
    'minplus': ('>=', lambda phi: phi >= 0),   # activation: R(g, m) >= phi
    'maxplus': ('<=', lambda phi: phi < 0),    # inhibition: R(g, m) <= phi
}


def binarize_branch(contexts, phi, branch):
    """Binarize with the rule of the K-FCA branch (``'minplus'`` or ``'maxplus'``)."""
    return binarize_contexts(contexts, phi=phi, mode=BRANCHES[branch][0])


def compute_peak_matrix(odor_neuron_structs, neuron_name_to_idx, window_seconds=2.0):
    """Graded matrix R(c) of the K-FCA analysis (odorants x neuron pairs).

    For each odorant-neuron pair, the time point with the largest absolute deflection
    of the *mean* trace is located and the mean trace is averaged over a symmetric
    window of +/- round(window_seconds * fps) samples around it, keeping the sign.
    This is the same computation as the K-FCA live script of KFCA_for_neuraldata.
    """
    n_odors = len(odor_neuron_structs)
    R = np.zeros((n_odors, len(neuron_name_to_idx)))
    for oi, mapping in enumerate(odor_neuron_structs):
        for name, j in neuron_name_to_idx.items():
            s = mapping[name]
            fps = float(np.ravel(s['fps'])[0])
            R[oi, j] = compute_peak_value(np.ravel(s['mean_trace']), int(round(window_seconds * fps)))
    return R


def build_thesis_worms(mat_path, seed=42, window_seconds=2.0):
    """Averaged worm + virtual worms used in the population-level CLasIC analysis.

    Returns a dict with ``contexts`` (n_worms + 1, n_odors, n_neurons): index 0 is the
    averaged worm (entry-wise mean of the virtual worms) and indices 1..K the K virtual
    worms, K = min number of traces of any odorant-neuron pair (18 at 10^-4, 12 at 10^-5
    and 10^-6), together with the data dictionary, the sampled trace indices and K.
    """
    data = load_compiled_data(mat_path)
    structs, n2i = build_odor_neuron_structs(data['compiled_LR_ordered'], data['neuron_names'])
    K = data['min_samples']
    traces, idx = build_random_virtual_worms(structs, n2i, data['ap_time'], K, seed=seed, store_indices=True)
    worms = compute_formal_contexts_from_traces(traces, data['ap_time'], window_seconds=window_seconds)
    contexts = np.concatenate([worms.mean(axis=0)[None], worms], axis=0)
    return {'data': data, 'contexts': contexts, 'K': K, 'trace_indices': idx}


def phi_grid_from_context_dir(folder):
    """Thresholds of existence of the K-FCA contexts in ``folder`` (read from the CSV names)."""
    return sorted(parse_phi_from_filename(p) for p in glob.glob(os.path.join(str(folder), '*.csv')))


def phi_grid(branch, R=None, context_dir=None):
    """Thresholds analysed for a K-FCA branch.

    If ``context_dir`` exists (the contexts exported by the K-FCA live script of
    KFCA_for_neuraldata), its thresholds are used, which reproduces the published
    results exactly. Otherwise every distinct entry of the graded matrix R(c) is used:
    a superset of the structural thresholds, giving the same trajectories at the shared
    values of phi.
    """
    if context_dir is not None and Path(context_dir).is_dir() and glob.glob(os.path.join(str(context_dir), '*.csv')):
        grid = phi_grid_from_context_dir(context_dir)
    elif R is not None:
        grid = sorted(np.unique(np.asarray(R)).tolist())
    else:
        raise ValueError('Provide the graded matrix R or an existing context_dir.')
    keep = BRANCHES[branch][1]
    return [float(p) for p in grid if keep(p)]


def normalize_conc_label(x):
    s = str(x).strip().lower().replace(' ', '')
    mapping = {'1e-4': '1e-4', '10^-4': '1e-4', '10^{-4}': '1e-4', '104': '1e-4', '4': '1e-4', '1e-5': '1e-5', '10^-5': '1e-5', '10^{-5}': '1e-5', '105': '1e-5', '5': '1e-5', '1e-6': '1e-6', '10^-6': '1e-6', '10^{-6}': '1e-6', '106': '1e-6', '6': '1e-6'}
    return mapping.get(s, s)


def _conc_label(exp: int) -> str:
    return rf"$10^{{-{exp}}}$"


def _conc_exp(label: str) -> int:
    m = re.search(r"-\{?(\d+)\}?", str(label))
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", str(label))
    if nums:
        return int(nums[-1])
    raise ValueError(f"Could not extract concentration exponent from: {label}")


def _extract_conc_number(conc_label: str) -> str:
    nums = re.findall(r"\d+", str(conc_label))
    if not nums:
        raise ValueError(f"Could not extract concentration number from: {conc_label}")
    return nums[-1]


def _resolve_phi_columns(df: pd.DataFrame):
    candidates = [
        ("phi_start", "phi_end", "phi_medoid"),
        ("phi_ini", "phi_fin", "phi_medoid"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    raise ValueError(
        "Could not find phi columns. Expected one of: "
        "('phi_start','phi_end','phi_medoid') or "
        "('phi_ini','phi_fin','phi_medoid')."
    )


def _read_context_csv(path: Path) -> pd.DataFrame:
    try:
        df_ctx = pd.read_csv(path, sep=";", index_col=0)
        if df_ctx.shape[1] == 0:
            raise ValueError
    except Exception:
        df_ctx = pd.read_csv(path, sep=",", index_col=0)

    df_ctx = df_ctx.loc[:, ~df_ctx.columns.astype(str).str.contains(r"^Unnamed")]
    return df_ctx


def _count_ones_from_context(df_ctx: pd.DataFrame) -> int:
    return int(np.count_nonzero(df_ctx.to_numpy()))


def _load_context_activation_index(
    base_dir,
    selected_concentrations,
    shape_tag="23x11",
    semiring="minplus",
    contexts_dirname="contexts",
):
    """
    Build a lookup:
        lookup['$10^{-4}$'] -> DataFrame(phi, n1, activation_pct)
    """
    lookup = {}

    for conc in selected_concentrations:
        conc_label = _conc_label(conc)
        folder = Path(base_dir) / contexts_dirname / f"od_by_neuron{conc}_{shape_tag}_{semiring}"

        if not folder.exists():
            raise FileNotFoundError(f"Context folder not found: {folder}")

        rows = []
        for fp in sorted(folder.glob("*.csv")):
            m = re.search(r"_([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\.csv$", fp.name)
            if m is None:
                continue

            phi = float(m.group(1))
            df_ctx = _read_context_csv(fp)
            n1 = _count_ones_from_context(df_ctx)

            rows.append(
                {
                    "phi": phi,
                    "n1": n1,
                    "activation_pct": 100.0 * n1 / df_ctx.size,
                }
            )

        if not rows:
            raise ValueError(f"No valid contexts found in: {folder}")

        lookup[conc_label] = (
            pd.DataFrame(rows)
            .sort_values("phi")
            .drop_duplicates(subset=["phi"])
            .reset_index(drop=True)
        )

    return lookup


def _nearest_activation(phi_target: float, df_lookup: pd.DataFrame, atol: float = 1e-12) -> float:
    arr = df_lookup["phi"].to_numpy()
    idx = np.where(np.isclose(arr, phi_target, atol=atol, rtol=0))[0]
    idx = int(idx[0]) if len(idx) else int(np.argmin(np.abs(arr - phi_target)))
    return float(df_lookup.iloc[idx]["activation_pct"])