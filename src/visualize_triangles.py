"""Plotting utilities for SMET, CMET, and generic ternary diagrams.

The functions in this file do not compute entropies. They only consume tables
already produced by ``calculate_entropies.py`` or by other project workflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

try:
    from ipywidgets import interact, widgets
except Exception:  # pragma: no cover - notebooks may not have widgets enabled
    interact = None

try:
    import mpltern  # noqa: F401
except Exception:  # pragma: no cover - importing mpltern registers the projection
    mpltern = None

_HAS_MPLTERN = mpltern is not None

_GENERIC_MARKERS = [
    'o', 's', '^', 'v', 'D', 'P', 'X', '*',
    '<', '>', 'h', 'H', 'p', '8', 'd', 'x', '+'
]

def _filter_by_phi(df, phi_val, accumulate=True):
    if accumulate:
        return df[df['phi'] <= phi_val]
    return df[df['phi'] == phi_val]


def _coords_smet(df):
    coords = df[['DeltaH_Pxi', 'M_Pxi', 'VI_Pxi']].to_numpy(float)
    return coords / coords.sum(axis=1, keepdims=True)


def _coords_cmet(df):
    coords = df[['I_norm', 'Hcond_norm', 'DeltaH_norm']].to_numpy(float)
    return coords / coords.sum(axis=1, keepdims=True)


def _colors_by_phi(vals):
    cmap = plt.get_cmap('turbo')
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    return (cmap(norm(vals)), cmap, norm)


def _colors_by_worm(df):
    cmap_base = plt.get_cmap('tab20')
    worms = df['worm'].astype(int).to_numpy()
    phis = df['phi'].astype(float).to_numpy()
    phi_min, phi_max = (phis.min(), phis.max())
    colors = []
    for w, phi_val in zip(worms, phis):
        base = np.array(cmap_base((w - 1) % 20))
        alpha = (phi_val - phi_min) / (phi_max - phi_min + 1e-12)
        final = base * (0.4 + 0.6 * alpha)
        colors.append(np.clip(final, 0, 1))
    return (np.array(colors), cmap_base)


def jitter_simple(coords, sigma=0.005, seed=0):
    rng = np.random.default_rng(seed)
    J = coords + rng.normal(0, sigma, size=coords.shape)
    J = np.clip(J, 0, None)
    J /= J.sum(axis=1, keepdims=True) + 1e-12
    return J


def _bary_to_xy(traj_coords):
    t = traj_coords[:, 0]
    l = traj_coords[:, 1]
    r = traj_coords[:, 2]
    x = r + 0.5 * t
    y = np.sqrt(3) / 2 * t
    return np.column_stack([x, y])


def _add_arrows_on_traj(ax, traj_coords, every=3, arrow_kwargs=None, min_len=0.03):
    if arrow_kwargs is None:
        arrow_kwargs = dict(scale_units='xy', scale=1, width=0.003, zorder=61)
    traj = np.asarray(traj_coords)
    if len(traj) < 2:
        return
    step = max(1, int(every))
    t0 = traj[:-1:step, 0]
    l0 = traj[:-1:step, 1]
    r0 = traj[:-1:step, 2]
    dt = traj[1::step, 0] - t0
    dl = traj[1::step, 1] - l0
    dr = traj[1::step, 2] - r0
    norm = np.sqrt(dt * dt + dl * dl + dr * dr)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    scale_fac = np.maximum(min_len / norm, 1.0)
    dt *= scale_fac
    dl *= scale_fac
    dr *= scale_fac
    outline = arrow_kwargs.pop('outline', None)
    if outline is not None:
        outline_kwargs = dict(arrow_kwargs)
        outline_kwargs.update(outline)
        ax.quiver(t0, l0, r0, dt, dl, dr, **outline_kwargs)
    ax.quiver(t0, l0, r0, dt, dl, dr, **arrow_kwargs)


def plot_smet_triangle(df, mode='global_avg', worm_ids=None, neuron=None, accumulate=True, phi_val=None, interactive=False, title=None, ax=None, fixed_phi_range=True, phi_min=None, phi_max=None, add_colorbar=True, s=50, marker_by=None, marker_legend=False, show_trajectory=False, trajectory_mode='centroid', trajectory_kwargs=None, trajectory_arrow_every=3, trajectory_arrow_kwargs=None, trajectory_arrow_min_len=0.03):
    if mode == 'global_avg':
        df_plot = df[(df['name'] == '@AVERAGE') & (df['worm'] == 'ALL')].copy()
    elif mode == 'per_worm_avg':
        df_plot = df[(df['name'] == '@AVERAGE') & (df['worm'] != 'ALL')].copy()
        if worm_ids:
            df_plot = df_plot[df_plot['worm'].astype(int).isin(worm_ids)]
    elif mode == 'one_worm_all':
        if not worm_ids:
            print('⚠️ Select one worm.')
            return
        if isinstance(worm_ids, int):
            worm_id = worm_ids
        else:
            if len(worm_ids) == 0:
                print('⚠️ Select at least one worm.')
                return
            worm_id = worm_ids[0]
        df_plot = df[df['name'] != '@AVERAGE'].copy()
        df_plot = df_plot[df_plot['worm'].astype(str) == str(worm_id)]
        if neuron is not None:
            if isinstance(neuron, (list, tuple)):
                if len(neuron) > 0:
                    df_plot = df_plot[df_plot['name'].isin(neuron)]
                else:
                    print('⚠️ Select at least one neuron.')
                    return
            else:
                df_plot = df_plot[df_plot['name'] == neuron]
    else:
        raise ValueError('Unknown SMET mode.')
    if phi_val is not None:
        if accumulate:
            df_plot = df_plot[df_plot['phi'] <= phi_val]
        else:
            df_plot = df_plot[np.isclose(df_plot['phi'], phi_val, atol=1e-12)]
    if df_plot.empty:
        print('No data to plot.')
        return
    coords = _coords_smet(df_plot)
    traj_coords = None
    if show_trajectory:
        phis_traj = df_plot['phi'].astype(float).to_numpy()
        if trajectory_mode == 'raw':
            order = np.argsort(phis_traj)
            traj_coords = coords[order, :]
        else:
            uniq_phis = np.array(sorted(df_plot['phi'].astype(float).unique()))
            traj_list = []
            for p in uniq_phis:
                m = np.isclose(phis_traj, p, atol=1e-12)
                traj_list.append(coords[m, :].mean(axis=0))
            traj_coords = np.vstack(traj_list)
    if mode == 'global_avg':
        phis = df_plot['phi'].astype(float).to_numpy()
        if fixed_phi_range:
            phi_min_val = phi_min if phi_min is not None else df['phi'].min()
            phi_max_val = phi_max if phi_max is not None else df['phi'].max()
            cmap = plt.get_cmap('turbo')
            norm = mpl.colors.Normalize(vmin=phi_min_val, vmax=phi_max_val)
            colors = cmap(norm(phis))
        else:
            colors, cmap, norm = _colors_by_phi(phis)
    elif mode == 'per_worm_avg':
        colors, cmap_base = _colors_by_worm(df_plot)
        phis = df_plot['phi'].astype(float).to_numpy()
        _, cmap_phi, norm_phi = _colors_by_phi(phis)
    elif mode == 'one_worm_all':
        if interactive:
            phis = df_plot['phi'].astype(float).to_numpy()
            if fixed_phi_range:
                phi_min_val = phi_min if phi_min is not None else df['phi'].min()
                phi_max_val = phi_max if phi_max is not None else df['phi'].max()
                cmap = plt.get_cmap('turbo')
                norm = mpl.colors.Normalize(vmin=phi_min_val, vmax=phi_max_val)
                colors = cmap(norm(phis))
            else:
                colors, cmap, norm = _colors_by_phi(phis)
        else:
            names = df_plot['name'].tolist()
            cmap = plt.get_cmap('tab20')
            unique = sorted(set(names))
            color_map = {n: cmap(i % 20) for i, n in enumerate(unique)}
            colors = [color_map[n] for n in names]
    if marker_by is not None:
        if marker_by not in df_plot.columns:
            raise ValueError(f"marker_by='{marker_by}' does not exist in df_plot. Columns: {list(df_plot.columns)}")
        _markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h', 'H', 'p', '8', 'd', 'x', '+']
        entities = df_plot[marker_by].astype(str).to_numpy()
        uniq_entities = sorted(set(entities))
        marker_map = {e: _markers[i % len(_markers)] for i, e in enumerate(uniq_entities)}
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='ternary', rotation=180)
    if marker_by is None:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=s, c=colors, alpha=0.25, edgecolors=None, linewidths=0.2)
    else:
        entities = df_plot[marker_by].astype(str).to_numpy()
        colors_arr = np.array(colors)
        for e in uniq_entities:
            m = entities == e
            ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2], s=s, c=colors_arr[m], marker=marker_map[e], alpha=0.25, edgecolors=None, linewidths=0.2, label=e if marker_legend else None)
        if marker_legend:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title=marker_by, fontsize=8)
    if show_trajectory and traj_coords is not None and (len(traj_coords) > 1):
        if trajectory_kwargs is None:
            trajectory_kwargs = dict(color='k', linewidth=2.0, zorder=60)
        ax.plot(traj_coords[:, 0], traj_coords[:, 1], traj_coords[:, 2], **trajectory_kwargs)
        if trajectory_arrow_kwargs is not None:
            _add_arrows_on_traj(ax, traj_coords, every=trajectory_arrow_every, arrow_kwargs=trajectory_arrow_kwargs, min_len=trajectory_arrow_min_len)
    ax.set_tlabel("$\\Delta H'_{P_{X_i}}$")
    ax.set_rlabel("$VI'_{P_{X_i}}$")
    ax.set_llabel("$M'_{P_{X_i}}$")
    ax.grid(True, linestyle='--', color='gray')
    if mode == 'global_avg':
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(phis)
        if add_colorbar:
            cbar = plt.colorbar(sm, ax=ax, pad=0.25)
            cbar.set_label('$\\varphi$')
    elif mode == 'per_worm_avg':
        worms = sorted(df_plot['worm'].astype(int).unique())
        handles = [mpl.lines.Line2D([], [], marker='o', linestyle='', color=cmap_base((w - 1) % 20), label=f'Worm {w}') for w in worms]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Worms')
    elif mode == 'one_worm_all' and interactive:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if add_colorbar:
            plt.colorbar(sm, ax=ax, pad=0.15)
    if title:
        ax.set_title(title)
    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_smet_interactive(df, phi_values, mode='global_avg', title='SMET', s=30, add_colorbar=True, marker_by=None, marker_legend=False, show_trajectory=False, trajectory_mode='centroid', trajectory_kwargs=None, trajectory_arrow_every=3, trajectory_arrow_kwargs=None, trajectory_arrow_min_len=0.03):

    def _update(k=0, accumulate=True, worms=None, neuron=None):
        phi_now = phi_values[k]
        plot_smet_triangle(df=df, mode=mode, worm_ids=worms, neuron=neuron, accumulate=accumulate, phi_val=phi_now, interactive=True, title=title, add_colorbar=add_colorbar, s=s, marker_by=marker_by, marker_legend=marker_legend, show_trajectory=show_trajectory, trajectory_mode=trajectory_mode, trajectory_kwargs=trajectory_kwargs, trajectory_arrow_every=trajectory_arrow_every, trajectory_arrow_kwargs=trajectory_arrow_kwargs, trajectory_arrow_min_len=trajectory_arrow_min_len)
    if mode == 'per_worm_avg':
        worm_widget = widgets.SelectMultiple(options=sorted(df[df['worm'] != 'ALL']['worm'].unique()), description='Worms')
    else:
        worm_widget = widgets.fixed(None)
    if mode == 'one_worm_all':
        worm_widget = widgets.Dropdown(options=sorted(df[df['worm'] != 'ALL']['worm'].unique()), description='Worm')
        neuron_widget = widgets.SelectMultiple(options=sorted(df[df['name'] != '@AVERAGE']['name'].unique()), description='Neurons', rows=6)
    else:
        neuron_widget = widgets.fixed(None)
    interact(_update, k=widgets.IntSlider(min=0, max=len(phi_values) - 1, step=1, description='φ index'), accumulate=widgets.Checkbox(value=True, description='Accumulate'), worms=worm_widget, neuron=neuron_widget)


def plot_cmet_triangle(df, typ='XY', accumulate=True, phi_val=None, interactive=False, title=None, fixed_phi_range=True, ax=None, phi_min=None, phi_max=None, s=30, add_colorbar=True, show_trajectory=False, trajectory_mode='centroid', trajectory_kwargs=None, trajectory_arrow_every=3, trajectory_arrow_kwargs=None, trajectory_arrow_min_len=0.03):
    df_t = df[df['type'] == typ].copy()
    if df_t.empty:
        print(f'No data for type {typ}.')
        return
    if phi_val is not None:
        if accumulate:
            df_t = df_t[df_t['phi'] <= phi_val]
        else:
            df_t = df_t[df_t['phi'] == phi_val]
    if df_t.empty:
        print('No CMET data after φ filtering.')
        return
    coords = _coords_cmet(df_t)
    traj_coords = None
    if show_trajectory:
        phis_traj = df_t['phi'].astype(float).to_numpy()
        if trajectory_mode == 'raw':
            order = np.argsort(phis_traj)
            traj_coords = coords[order, :]
        else:
            uniq_phis = np.array(sorted(df_t['phi'].astype(float).unique()))
            traj_list = []
            for p in uniq_phis:
                m = np.isclose(phis_traj, p, atol=1e-12)
                traj_list.append(coords[m, :].mean(axis=0))
            traj_coords = np.vstack(traj_list)
    phi_vals = df_t['phi'].astype(float).to_numpy()
    if fixed_phi_range:
        phi_min_val = phi_min if phi_min is not None else df['phi'].min()
        phi_max_val = phi_max if phi_max is not None else df['phi'].max()
        cmap = plt.get_cmap('turbo')
        norm = mpl.colors.Normalize(vmin=phi_min_val, vmax=phi_max_val)
        colors = cmap(norm(phi_vals))
    else:
        colors, cmap, norm = _colors_by_phi(phi_vals)
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='ternary')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=s, c=colors, alpha=0.25, edgecolors=None, linewidths=0.2)
    if show_trajectory and traj_coords is not None and (len(traj_coords) > 1):
        if trajectory_kwargs is None:
            trajectory_kwargs = dict(color='k', linewidth=2.0, zorder=60)
        ax.plot(traj_coords[:, 0], traj_coords[:, 1], traj_coords[:, 2], **trajectory_kwargs)
        if trajectory_arrow_kwargs is not None:
            _add_arrows_on_traj(ax, traj_coords, every=trajectory_arrow_every, arrow_kwargs=trajectory_arrow_kwargs, min_len=trajectory_arrow_min_len)
    ax.set_llabel("$VI'_{P_{\\bar X\\bar Y}}$")
    ax.set_tlabel("$2 I'_{P_{\\bar X\\bar Y}}$")
    ax.set_rlabel("$\\Delta H'_{P_{\\bar X|\\bar Y}}$")
    ax.grid(True, linestyle='--', color='gray')
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if add_colorbar:
        cbar = plt.colorbar(sm, ax=ax, pad=0.15)
        cbar.set_label('$\\varphi$')
    if title:
        ax.set_title(title)
    if ax is None:
        plt.tight_layout()
        plt.show()


def _plot_cmet_two_axes(df, phi_now, accumulate, fixed_phi_range=True, phi_min=None, phi_max=None, s=30, add_colorbar=True, show_trajectory=False, trajectory_mode='centroid', trajectory_kwargs=None, trajectory_arrow_every=3, trajectory_arrow_kwargs=None, trajectory_arrow_min_len=0.03):
    df_sub = _filter_by_phi(df, phi_now, accumulate)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='ternary')
    df_x = df_sub[df_sub['type'] == 'X']
    if df_x.empty:
        ax1.set_title('X: No data')
        norm_x = cmap_x = None
    else:
        coords_x = _coords_cmet(df_x)
        phi_vals_x = df_x['phi'].astype(float).to_numpy()
        if fixed_phi_range:
            phi_min_val = phi_min if phi_min is not None else df['phi'].min()
            phi_max_val = phi_max if phi_max is not None else df['phi'].max()
            cmap_x = plt.get_cmap('turbo')
            norm_x = mpl.colors.Normalize(vmin=phi_min_val, vmax=phi_max_val)
            colors_x = cmap_x(norm_x(phi_vals_x))
        else:
            colors_x, cmap_x, norm_x = _colors_by_phi(phi_vals_x)
        ax1.scatter(coords_x[:, 0], coords_x[:, 1], coords_x[:, 2], s=s, c=colors_x, alpha=0.25, edgecolors=None, linewidths=0.2)
        if show_trajectory:
            phis_traj = df_x['phi'].astype(float).to_numpy()
            if trajectory_mode == 'raw':
                order = np.argsort(phis_traj)
                traj_x = coords_x[order, :]
            else:
                uniq_phis = np.array(sorted(df_x['phi'].astype(float).unique()))
                traj_x = np.vstack([coords_x[np.isclose(phis_traj, p, atol=1e-12)].mean(axis=0) for p in uniq_phis])
            if trajectory_kwargs is None:
                tk = dict(color='k', linewidth=2.0, zorder=60)
            else:
                tk = trajectory_kwargs
            ax1.plot(traj_x[:, 0], traj_x[:, 1], traj_x[:, 2], **tk)
            if trajectory_arrow_kwargs is not None:
                _add_arrows_on_traj(ax1, traj_x, every=trajectory_arrow_every, arrow_kwargs=trajectory_arrow_kwargs, min_len=trajectory_arrow_min_len)
        ax1.set_title(f'CMET – TYPE X – φ={phi_now:.3f}')
        ax1.set_tlabel("$I'$")
        ax1.set_rlabel("$\\Delta H'$")
        ax1.set_llabel("$H_{\\mathrm{cond}}'$")
        ax1.grid(True, linestyle='--', color='gray')
    ax2 = fig.add_subplot(1, 2, 2, projection='ternary')
    df_y = df_sub[df_sub['type'] == 'Y']
    if df_y.empty:
        ax2.set_title('Y: No data')
        norm_y = cmap_y = None
    else:
        coords_y = _coords_cmet(df_y)
        phi_vals_y = df_y['phi'].astype(float).to_numpy()
        if fixed_phi_range:
            phi_min_val = phi_min if phi_min is not None else df['phi'].min()
            phi_max_val = phi_max if phi_max is not None else df['phi'].max()
            cmap_y = plt.get_cmap('turbo')
            norm_y = mpl.colors.Normalize(vmin=phi_min_val, vmax=phi_max_val)
            colors_y = cmap_y(norm_y(phi_vals_y))
        else:
            colors_y, cmap_y, norm_y = _colors_by_phi(phi_vals_y)
        ax2.scatter(coords_y[:, 0], coords_y[:, 1], coords_y[:, 2], s=30, c=colors_y, alpha=0.25, edgecolors=None, linewidths=0.2)
        if show_trajectory:
            phis_traj = df_y['phi'].astype(float).to_numpy()
            if trajectory_mode == 'raw':
                order = np.argsort(phis_traj)
                traj_y = coords_y[order, :]
            else:
                uniq_phis = np.array(sorted(df_y['phi'].astype(float).unique()))
                traj_y = np.vstack([coords_y[np.isclose(phis_traj, p, atol=1e-12)].mean(axis=0) for p in uniq_phis])
            if trajectory_kwargs is None:
                tk = dict(color='k', linewidth=2.0, zorder=60)
            else:
                tk = trajectory_kwargs
            ax2.plot(traj_y[:, 0], traj_y[:, 1], traj_y[:, 2], **tk)
            if trajectory_arrow_kwargs is not None:
                _add_arrows_on_traj(ax2, traj_y, every=trajectory_arrow_every, arrow_kwargs=trajectory_arrow_kwargs, min_len=trajectory_arrow_min_len)
        ax2.set_title(f'CMET – TYPE Y – φ={phi_now:.3f}')
        ax2.set_tlabel("$I'$")
        ax2.set_rlabel("$\\Delta H'$")
        ax2.set_llabel("$H_{\\mathrm{cond}}'$")
        ax2.grid(True, linestyle='--', color='gray')
    cmap_final = cmap_x or cmap_y
    norm_final = norm_x or norm_y
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = mpl.cm.ScalarMappable(cmap=cmap_final, norm=norm_final)
    sm.set_array([])
    if add_colorbar:
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('$\\varphi$')
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.1, top=0.9, wspace=0.25)
    plt.show()


def plot_cmet_interactive(df, phi_values, s=30, add_colorbar=True, show_trajectory=False, trajectory_mode='centroid', trajectory_kwargs=None, trajectory_arrow_every=3, trajectory_arrow_kwargs=None, trajectory_arrow_min_len=0.03):

    def _update(k=0, accumulate=True):
        phi_now = phi_values[k]
        _plot_cmet_two_axes(df, phi_now, accumulate, s=s, add_colorbar=add_colorbar, show_trajectory=show_trajectory, trajectory_mode=trajectory_mode, trajectory_kwargs=trajectory_kwargs, trajectory_arrow_every=trajectory_arrow_every, trajectory_arrow_kwargs=trajectory_arrow_kwargs, trajectory_arrow_min_len=trajectory_arrow_min_len)
    interact(_update, k=widgets.IntSlider(min=0, max=len(phi_values) - 1, step=1, description='φ index'), accumulate=widgets.Checkbox(value=True, description='Accumulate'))


def _normalize_rows_generic(coords: np.ndarray, eps: float=1e-12) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    row_sum = coords.sum(axis=1, keepdims=True)
    bad = np.abs(row_sum[:, 0]) <= eps
    if np.any(bad):
        n_bad = int(np.sum(bad))
        raise ValueError(f'There are {n_bad} rows with sum ~0; they cannot be normalized as barycentric coordinates.')
    return coords / row_sum


def _extract_coords_generic(df: pd.DataFrame, coord_cols, normalize: bool=True) -> np.ndarray:
    if len(coord_cols) != 3:
        raise ValueError('coord_cols must contain exactly 3 column names.')
    missing = [c for c in coord_cols if c not in df.columns]
    if missing:
        raise ValueError(f'These columns are not present in the DataFrame: {missing}')
    coords = df[list(coord_cols)].to_numpy(dtype=float)
    if normalize:
        coords = _normalize_rows_generic(coords)
    return coords


def _filter_by_value_generic(df: pd.DataFrame, value_col: str | None=None, value=None, accumulate: bool=True, atol: float=1e-12) -> pd.DataFrame:
    if value_col is None or value is None:
        return df.copy()
    if value_col not in df.columns:
        raise ValueError(f"value_col='{value_col}' does not exist in the DataFrame.")
    out = df.copy()
    series = out[value_col]
    if pd.api.types.is_numeric_dtype(series):
        vals = series.astype(float).to_numpy()
        target = float(value)
        if accumulate:
            return out[vals <= target]
        return out[np.isclose(vals, target, atol=atol)]
    if accumulate:
        raise ValueError('accumulate=True only makes sense for numeric value_col.')
    return out[out[value_col] == value]


def _is_numeric_series_generic(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _build_colors_generic(df: pd.DataFrame, color_by: str | None=None, color=None, cmap: str='turbo', color_palette: str='tab20', fixed_color_range: bool=True, color_min=None, color_max=None):
    n = len(df)
    if color is not None:
        colors = np.array([mpl.colors.to_rgba(color)] * n)
        return (colors, 'constant', None, None, None)
    if color_by is None:
        colors = np.array([mpl.colors.to_rgba('C0')] * n)
        return (colors, 'constant', None, None, None)
    if color_by not in df.columns:
        raise ValueError(f"color_by='{color_by}' does not exist in the DataFrame.")
    s = df[color_by]
    if _is_numeric_series_generic(s):
        vals = s.astype(float).to_numpy()
        cmap_obj = plt.get_cmap(cmap)
        if fixed_color_range:
            vmin = vals.min() if color_min is None else color_min
            vmax = vals.max() if color_max is None else color_max
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colors = cmap_obj(norm(vals))
        return (colors, 'numeric', cmap_obj, norm, None)
    cmap_obj = plt.get_cmap(color_palette)
    cats = df[color_by].astype(str).to_numpy()
    uniq = sorted(set(cats))
    color_map = {cat: cmap_obj(i % 20) for i, cat in enumerate(uniq)}
    colors = np.array([color_map[c] for c in cats])
    handles = [Line2D([], [], marker='o', linestyle='', color=color_map[c], label=str(c), markersize=8) for c in uniq]
    return (colors, 'categorical', None, None, handles)


def _build_marker_map_generic(df: pd.DataFrame, marker_by: str | None=None):
    if marker_by is None:
        return (None, None)
    if marker_by not in df.columns:
        raise ValueError(f"marker_by='{marker_by}' does not exist in the DataFrame.")
    entities = df[marker_by].astype(str).to_numpy()
    uniq = sorted(set(entities))
    marker_map = {e: _GENERIC_MARKERS[i % len(_GENERIC_MARKERS)] for i, e in enumerate(uniq)}
    return (entities, marker_map)


def _compute_trajectory_generic(df_plot: pd.DataFrame, coords: np.ndarray, mode: str='centroid', trajectory_groupby: str | None=None, trajectory_sort_by: str | None=None):
    if len(df_plot) < 2:
        return None
    if mode not in {'raw', 'centroid'}:
        raise ValueError("trajectory_mode debe ser 'raw' o 'centroid'.")
    if mode == 'raw':
        if trajectory_sort_by is None:
            raise ValueError("Con trajectory_mode='raw' debes indicar trajectory_sort_by.")
        if trajectory_sort_by not in df_plot.columns:
            raise ValueError(f"trajectory_sort_by='{trajectory_sort_by}' does not exist.")
        order = np.argsort(df_plot[trajectory_sort_by].to_numpy())
        traj = coords[order, :]
        return traj if len(traj) > 1 else None
    if trajectory_groupby is None:
        raise ValueError("Con trajectory_mode='centroid' debes indicar trajectory_groupby.")
    if trajectory_groupby not in df_plot.columns:
        raise ValueError(f"trajectory_groupby='{trajectory_groupby}' does not exist.")
    group_vals = df_plot[trajectory_groupby]
    uniq = np.array(sorted(group_vals.unique()))
    traj_list = []
    for g in uniq:
        if pd.api.types.is_numeric_dtype(group_vals):
            mask = np.isclose(group_vals.astype(float).to_numpy(), float(g), atol=1e-12)
        else:
            mask = group_vals.astype(str).to_numpy() == str(g)
        traj_list.append(coords[mask, :].mean(axis=0))
    traj = np.vstack(traj_list)
    return traj if len(traj) > 1 else None


def plot_custom_triangle(df: pd.DataFrame, coord_cols, rotation: float=0, *, labels=None, normalize: bool=False, value_col: str | None=None, value=None, accumulate: bool=True, color_by: str | None='phi', color=None, cmap: str='turbo', color_palette: str='tab20', fixed_color_range: bool=True, color_min=None, color_max=None, marker: str='o', marker_by: str | None=None, marker_legend: bool=False, color_legend: bool=False, show_colorbar: bool=True, s: float=30, alpha: float=0.35, edgecolor=None, linewidth: float=0.3, title: str | None=None, ax=None, grid: bool=True, grid_kwargs: dict | None=None, show_trajectory: bool=False, trajectory_mode: str='centroid', trajectory_groupby: str | None=None, trajectory_sort_by: str | None=None, trajectory_kwargs: dict | None=None, trajectory_arrow_every: int=3, trajectory_arrow_kwargs: dict | None=None, trajectory_arrow_min_len: float=0.03):
    if not _HAS_MPLTERN:
        raise ImportError('plot_custom_triangle requiere mpltern.')
    if labels is None:
        labels = tuple(coord_cols)
    if len(labels) != 3:
        raise ValueError('labels debe contener exactamente 3 elementos.')
    df_plot = _filter_by_value_generic(df, value_col=value_col, value=value, accumulate=accumulate)
    if df_plot.empty:
        raise ValueError('No data available after filtering.')
    coords = _extract_coords_generic(df_plot, coord_cols=coord_cols, normalize=normalize)
    colors, color_mode, cmap_obj, norm, color_handles = _build_colors_generic(df_plot, color_by=color_by, color=color, cmap=cmap, color_palette=color_palette, fixed_color_range=fixed_color_range, color_min=color_min, color_max=color_max)
    entities, marker_map = _build_marker_map_generic(df_plot, marker_by=marker_by)
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='ternary', rotation=rotation)
        created_fig = True
    else:
        fig = ax.figure
    colors_arr = np.array(colors)
    marker_handles = None
    if marker_by is None:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=s, c=colors_arr, marker=marker, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth)
    else:
        marker_handles = []
        for ent in sorted(marker_map.keys()):
            mask = entities == ent
            mk = marker_map[ent]
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], s=s, c=colors_arr[mask], marker=mk, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth)
            marker_handles.append(Line2D([], [], marker=mk, linestyle='', color='black', label=str(ent), markerfacecolor='lightgray', markeredgecolor='black', markersize=8))
    if show_trajectory:
        if trajectory_mode == 'centroid' and trajectory_groupby is None:
            trajectory_groupby = value_col
        if trajectory_mode == 'raw' and trajectory_sort_by is None:
            trajectory_sort_by = value_col
        traj_coords = _compute_trajectory_generic(df_plot, coords, mode=trajectory_mode, trajectory_groupby=trajectory_groupby, trajectory_sort_by=trajectory_sort_by)
        if traj_coords is not None and len(traj_coords) > 1:
            if trajectory_kwargs is None:
                trajectory_kwargs = dict(color='k', linewidth=2.0, zorder=60)
            ax.plot(traj_coords[:, 0], traj_coords[:, 1], traj_coords[:, 2], **trajectory_kwargs)
            if trajectory_arrow_kwargs is not None:
                _add_arrows_on_traj(ax, traj_coords, every=trajectory_arrow_every, arrow_kwargs=trajectory_arrow_kwargs, min_len=trajectory_arrow_min_len)
    ax.set_tlabel(labels[0])
    ax.set_llabel(labels[1])
    ax.set_rlabel(labels[2])
    if grid:
        if grid_kwargs is None:
            grid_kwargs = dict(linestyle='--', color='gray')
        ax.grid(True, **grid_kwargs)
    if title:
        ax.set_title(title)
    if show_colorbar and color_mode == 'numeric':
        sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.15)
        cbar.set_label(color_by)
    if marker_legend and marker_handles:
        leg1 = ax.legend(handles=marker_handles, bbox_to_anchor=(1.02, 1.0), loc='upper left', title=marker_by, fontsize=8)
        ax.add_artist(leg1)
    if color_legend and color_mode == 'categorical' and color_handles:
        ax.legend(handles=color_handles, bbox_to_anchor=(1.02, 0.0), loc='lower left', title=color_by, fontsize=8)
    if created_fig:
        plt.tight_layout()
        plt.show()
    return (fig, ax)


def plot_custom_triangle_interactive(df: pd.DataFrame, coord_cols, rotation: float=0, *, slider_col: str='phi', slider_values=None, labels=None, normalize: bool=True, accumulate_default: bool=True, color_by: str | None='phi', color=None, cmap: str='turbo', color_palette: str='tab20', fixed_color_range: bool=True, color_min=None, color_max=None, marker: str='o', marker_by: str | None=None, marker_legend: bool=False, color_legend: bool=False, show_colorbar: bool=True, s: float=30, alpha: float=0.35, edgecolor=None, linewidth: float=0.3, title: str | None=None, grid: bool=True, grid_kwargs: dict | None=None, show_trajectory: bool=False, trajectory_mode: str='centroid', trajectory_groupby: str | None=None, trajectory_sort_by: str | None=None, trajectory_kwargs: dict | None=None, trajectory_arrow_every: int=3, trajectory_arrow_kwargs: dict | None=None, trajectory_arrow_min_len: float=0.03):
    if slider_col not in df.columns:
        raise ValueError(f"slider_col='{slider_col}' does not exist in the DataFrame.")
    if slider_values is None:
        slider_values = sorted(df[slider_col].dropna().unique())
    slider_values = list(slider_values)

    def _update(k=0, accumulate=accumulate_default):
        current_value = slider_values[k]
        title_now = title if title is not None else f'{slider_col} = {current_value}'
        plot_custom_triangle(df=df, coord_cols=coord_cols, rotation=rotation, labels=labels, normalize=normalize, value_col=slider_col, value=current_value, accumulate=accumulate, color_by=color_by, color=color, cmap=cmap, color_palette=color_palette, fixed_color_range=fixed_color_range, color_min=color_min, color_max=color_max, marker=marker, marker_by=marker_by, marker_legend=marker_legend, color_legend=color_legend, show_colorbar=show_colorbar, s=s, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, title=title_now, grid=grid, grid_kwargs=grid_kwargs, show_trajectory=show_trajectory, trajectory_mode=trajectory_mode, trajectory_groupby=trajectory_groupby, trajectory_sort_by=trajectory_sort_by, trajectory_kwargs=trajectory_kwargs, trajectory_arrow_every=trajectory_arrow_every, trajectory_arrow_kwargs=trajectory_arrow_kwargs, trajectory_arrow_min_len=trajectory_arrow_min_len)
    interact(_update, k=widgets.IntSlider(min=0, max=len(slider_values) - 1, step=1, value=0, description=f'{slider_col} idx'), accumulate=widgets.Checkbox(value=accumulate_default, description='Accumulate'))


def _ctx_to_vec01(ctx) -> np.ndarray:
    if isinstance(ctx, pd.DataFrame):
        return ctx.values.astype(np.uint8).ravel()
    return np.asarray(ctx, dtype=np.uint8).ravel()


def jaccard_triangle_components(ctx1, ctx2):
    x = _ctx_to_vec01(ctx1)
    y = _ctx_to_vec01(ctx2)
    if x.size != y.size:
        raise ValueError('Both contexts must have the same size.')
    U = x.size
    inter = int(np.sum((x == 1) & (y == 1)))
    only1 = int(np.sum((x == 1) & (y == 0)))
    only2 = int(np.sum((x == 0) & (y == 1)))
    n1 = inter + only1
    n2 = inter + only2
    union = inter + only1 + only2
    sigma = inter / U
    delta = (only1 + only2) / (2.0 * U)
    Delta = (2.0 * U - n1 - n2) / (2.0 * U)
    j = 1.0 if union == 0 else float(inter / union)
    return {'sigma': sigma, 'delta': delta, 'Delta': Delta, 'jaccard': j, 'groundedness': 1.0 - Delta, 'intersection': inter, 'only_left': only1, 'only_right': only2, 'n_left': n1, 'n_right': n2, 'union': union}


def build_jaccard_triangle_df_from_contexts(contexts: dict, phi_mapper=None):
    phis = sorted(contexts.keys())
    rows = []
    for p1, p2 in zip(phis[:-1], phis[1:]):
        comp = jaccard_triangle_components(contexts[p1], contexts[p2])
        row = {'phi_left': float(p1), 'phi_right': float(p2), 'phi_mid': 0.5 * (float(p1) + float(p2)), **comp}
        if phi_mapper is not None:
            row['phi_mid_mapped'] = float(phi_mapper(row['phi_mid']))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_jaccard_triangle(df_jt: pd.DataFrame, ax=None, color_by: str | None='phi_mid_mapped', show_colorbar: bool=True, title: str | None='Jaccard triangle', s: float=36, alpha: float=0.8, marker: str='o', cmap: str='viridis', fixed_color_range: bool=True, color_min=None, color_max=None, show_trajectory: bool=True, trajectory_mode: str='raw', trajectory_groupby: str | None=None, trajectory_sort_by: str | None='phi_mid', trajectory_kwargs: dict | None=None, trajectory_arrow_every: int=3, trajectory_arrow_kwargs: dict | None=None, trajectory_arrow_min_len: float=0.03):
    return plot_custom_triangle(df=df_jt, coord_cols=('sigma', 'delta', 'Delta'), rotation=0, labels=('$\\sigma$', '$\\delta$', '$\\Delta$'), normalize=False, color_by=color_by, cmap=cmap, fixed_color_range=fixed_color_range, color_min=color_min, color_max=color_max, marker=marker, show_colorbar=show_colorbar, s=s, alpha=alpha, title=title, ax=ax, show_trajectory=show_trajectory, trajectory_mode=trajectory_mode, trajectory_groupby=trajectory_groupby, trajectory_sort_by=trajectory_sort_by, trajectory_kwargs=trajectory_kwargs, trajectory_arrow_every=trajectory_arrow_every, trajectory_arrow_kwargs=trajectory_arrow_kwargs, trajectory_arrow_min_len=trajectory_arrow_min_len)


def plot_jaccard_triangle_interactive(df_jt: pd.DataFrame, slider_col: str='phi_right', slider_values=None, **kwargs):
    return plot_custom_triangle_interactive(df=df_jt, coord_cols=('sigma', 'delta', 'Delta'), rotation=0, labels=('$\\sigma$', '$\\delta$', '$\\Delta$'), normalize=False, slider_col=slider_col, slider_values=slider_values, **kwargs)
