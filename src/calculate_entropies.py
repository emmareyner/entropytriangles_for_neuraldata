"""Utilities to compute SMET and CMET quantities.

This module only contains entropy-related calculations. Plotting utilities live
in ``visualize_triangles.py``.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fca import compute_concepts

ArrayLike = Union[np.ndarray, pd.DataFrame]


# Internal helpers


def _to_numpy_2d(X: ArrayLike) -> Tuple[np.ndarray, Optional[List[str]]]:
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
        arr = X.to_numpy()
        return (np.asarray(arr), cols)
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[:, None]
        return (arr, None)


def _entropy_from_counts(counts: np.ndarray, base: float=2.0) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p) / math.log(base)))


def _entropy_1d(values: np.ndarray, base: float=2.0) -> float:
    vals = np.asarray(values)
    unique, counts = np.unique(vals, return_counts=True)
    return _entropy_from_counts(counts, base=base)


def _entropy_multivariate(X: np.ndarray, base: float=2.0) -> float:
    X = np.asarray(X)
    if X.ndim == 1:
        return _entropy_1d(X, base=base)
    _, counts = np.unique(X, axis=0, return_counts=True)
    return _entropy_from_counts(counts, base=base)


def _cond_entropy(X: np.ndarray, Y: np.ndarray, base: float=2.0) -> float:
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y must have the same number of rows.')
    _, x_codes = np.unique(X, axis=0, return_inverse=True)
    _, y_codes = np.unique(Y, axis=0, return_inverse=True)
    xy = np.stack([x_codes, y_codes], axis=1)
    xy_unique, xy_counts = np.unique(xy, axis=0, return_counts=True)
    total = xy_counts.sum()
    y_vals, y_counts = np.unique(y_codes, return_counts=True)
    max_y = int(y_vals.max()) if len(y_vals) > 0 else -1
    p_y = np.zeros(max_y + 1)
    p_y[y_vals] = y_counts / total
    log_base = math.log(base)
    H = 0.0
    p_xy = xy_counts / total
    for (x_c, y_c), pxy in zip(xy_unique, p_xy):
        if pxy > 0 and p_y[y_c] > 0:
            H -= pxy * (math.log(pxy / p_y[y_c]) / log_base)
    return float(H)


def _log2_prod_n_distinct(X: np.ndarray) -> float:
    X = np.asarray(X)
    out = 0.0
    for j in range(X.shape[1]):
        nvals = len(np.unique(X[:, j]))
        if nvals > 0:
            out += math.log(nvals, 2)
    return out



# SMET

def sentropies(dat: ArrayLike, names: Optional[Iterable[str]]=None, type: str='total', base: float=2.0) -> pd.DataFrame:
    if type not in ('total', 'dual'):
        raise ValueError("type must be 'total' or 'dual'")
    X, colnames = _to_numpy_2d(dat)
    n_samples, n_vars = X.shape
    if n_samples == 0 or n_vars == 0:
        raise ValueError('sentropies: empty matrix')
    if names is not None:
        var_names = list(names)
    elif colnames is not None:
        var_names = colnames
    else:
        var_names = [f'x{i + 1}' for i in range(n_vars)]
    H_Uxi = np.array([math.log(max(2, len(np.unique(X[:, j]))), base) for j in range(n_vars)])
    H_Pxi = np.array([_entropy_1d(X[:, j], base=base) for j in range(n_vars)])
    if n_vars == 1:
        VI_Pxi = H_Pxi.copy()
    else:
        VI_Pxi = np.zeros(n_vars)
        for j in range(n_vars):
            VI_Pxi[j] = _cond_entropy(X[:, j:j + 1], np.delete(X, j, axis=1), base=base)
    DeltaH = H_Uxi - H_Pxi
    M = H_Pxi - VI_Pxi
    edf = pd.DataFrame({'name': var_names, 'H_Uxi': H_Uxi, 'H_Pxi': H_Pxi, 'DeltaH_Pxi': DeltaH, 'M_Pxi': M, 'VI_Pxi': VI_Pxi})
    sums = edf.iloc[:, 1:6].sum(axis=0)
    avg_row = pd.DataFrame([['@AVERAGE'] + sums.tolist()], columns=edf.columns)
    edf = pd.concat([edf, avg_row], ignore_index=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        edf['DeltaH_norm'] = edf['DeltaH_Pxi'] / edf['H_Uxi']
        edf['M_norm'] = edf['M_Pxi'] / edf['H_Uxi']
        edf['VI_norm'] = edf['VI_Pxi'] / edf['H_Uxi']
    edf[['DeltaH_norm', 'M_norm', 'VI_norm']] = edf[['DeltaH_norm', 'M_norm', 'VI_norm']].fillna(0)
    return edf


def build_smet_from_intents(contexts_bin, odor_names, neuron_names, phi):
    n_worms, _, _ = contexts_bin.shape
    SMET_list = []
    intents_all_list = []
    for w in range(n_worms):
        Cw = contexts_bin[w]
        concepts, extents_matrix, intents_matrix, obj_names, attr_names = compute_concepts(Cw, object_names=odor_names, attribute_names=neuron_names)
        intents_all_list.append(intents_matrix)
        SEDF_w = sentropies(dat=intents_matrix, names=neuron_names, type='total', base=2.0)
        SEDF_w['worm'] = w + 1
        SEDF_w['phi'] = phi
        SMET_list.append(SEDF_w)
    df_full = pd.concat(SMET_list, ignore_index=True)
    intents_all = np.vstack(intents_all_list)
    SEDF_global = sentropies(dat=intents_all, names=neuron_names, type='total', base=2.0)
    SEDF_global['worm'] = 'ALL'
    SEDF_global['phi'] = phi
    df_full = pd.concat([df_full, SEDF_global], ignore_index=True)
    return df_full



# CMET

def jentropies(X: ArrayLike, Y: ArrayLike, base: float=2.0) -> pd.DataFrame:
    X_arr, _ = _to_numpy_2d(X)
    Y_arr, _ = _to_numpy_2d(Y)
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError('X and Y must have the same number of rows.')
    H_U_X = _log2_prod_n_distinct(X_arr)
    H_U_Y = _log2_prod_n_distinct(Y_arr)
    H_P_X = _entropy_multivariate(X_arr, base=base)
    H_P_Y = _entropy_multivariate(Y_arr, base=base)
    H_X_given_Y = _cond_entropy(X_arr, Y_arr, base=base)
    H_Y_given_X = _cond_entropy(Y_arr, X_arr, base=base)
    I1 = H_P_X - H_X_given_Y
    I2 = H_P_Y - H_Y_given_X
    I = 0.5 * (I1 + I2)
    DeltaH_X = H_U_X - H_P_X
    DeltaH_Y = H_U_Y - H_P_Y
    H_U_XY = H_U_X + H_U_Y
    H_P_XY = H_P_X + H_P_Y
    DeltaH_XY = DeltaH_X + DeltaH_Y
    M_XY = 2 * I
    VI_XY = H_X_given_Y + H_Y_given_X
    df = pd.DataFrame({'type': ['X', 'Y', 'XY'], 'H_U': [H_U_X, H_U_Y, H_U_XY], 'H_P': [H_P_X, H_P_Y, H_P_XY], 'DeltaH_P': [DeltaH_X, DeltaH_Y, DeltaH_XY], 'M_P': [I, I, M_XY], 'VI_P': [H_X_given_Y, H_Y_given_X, VI_XY]})
    with np.errstate(divide='ignore', invalid='ignore'):
        df['DeltaH_norm'] = df['DeltaH_P'] / df['H_U']
        df['I_norm'] = df['M_P'] / df['H_U']
        df['Hcond_norm'] = df['VI_P'] / df['H_U']
    df[['DeltaH_norm', 'I_norm', 'Hcond_norm']] = df[['DeltaH_norm', 'I_norm', 'Hcond_norm']].fillna(0)
    return df


def build_smet_from_extents(contexts_bin, odor_names, neuron_names, phi):
    n_worms, _, _ = contexts_bin.shape
    SMET_list = []
    extents_all = []
    for w in range(n_worms):
        Cw = contexts_bin[w]
        _, ext_matrix, _, obj_names, attr_names = compute_concepts(Cw, object_names=odor_names, attribute_names=neuron_names)
        extents_all.append(ext_matrix)
        SEDF_w = sentropies(dat=ext_matrix, names=odor_names, type='total', base=2.0)
        SEDF_w['worm'] = w + 1
        SEDF_w['phi'] = phi
        SMET_list.append(SEDF_w)
    ext_all = np.vstack(extents_all)
    SEDF_global = sentropies(dat=ext_all, names=odor_names, type='total', base=2.0)
    SEDF_global['worm'] = 'ALL'
    SEDF_global['phi'] = phi
    SMET_list.append(SEDF_global)
    return pd.concat(SMET_list, ignore_index=True)
