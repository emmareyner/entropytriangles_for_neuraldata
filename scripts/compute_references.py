"""compute_references.py: random references for CLasIC (Chapter 5 of the thesis).

Subcommands
-----------
bernoulli   Reachable geometry of single Bernoulli contexts.
            Sizes 10x10, 20x20, 30x30, 40x40, 10x40 and 23x11 by default; 20 densities
            (utils: randomization.density_levels); 250 contexts per density (50 for 40x40).
            -> {out}/bernoulli/{SMETint,SMEText,CMET}_{n}x{m}.csv

ensembles   Reachable geometry of pooled Bernoulli ensembles (N = 19 contexts of 23x11,
            redraw probability eps in {0, .1, .25, .5, .75, 1}, 100 ensembles per cell).
            -> {out}/ensembles/pooled_geometry_grid.csv

matched     Ensembles matched, at every phi, to the density p̂ and consistency ε̂ of the
            averaged + virtual worms (200 ensembles per phi; reference B of the thesis).
            -> {out}/nulls/B_population_{branch}_{100+c}.csv, worm_consistency_*.csv

margins     Margin-preserving randomizations of every worm at every phi (200 per phi):
            swap (both margins), column_preserving (neuron specificity) and
            row_preserving (odorant sparseness). Population triangles plus the SMET of
            every neuron and odorant of the averaged worm.
            -> {out}/nulls/{C,C_col,C_row}_{population,individual}_{branch}_{100+c}.csv(.gz)
            -> {out}/nulls/observed_{branch}_{100+c}.csv

Seeds follow the thesis analysis, so `matched` and `margins` reproduce the published
references exactly when the thresholds of the K-FCA contexts are available
(see scripts/compute_clasic.py).

Examples
--------
    python scripts/compute_references.py bernoulli --sizes 23x11 10x10
    python scripts/compute_references.py matched --conc 4 5 6
    python scripts/compute_references.py margins --conc 4 --processes 4
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fastclasic as fc  # noqa: E402
import randomization as rz  # noqa: E402
from utils import BRANCHES, binarize_branch, build_odor_neuron_structs, build_thesis_worms, \
    compute_peak_matrix, phi_grid  # noqa: E402

SEED, REPS = 42, 200
COLS = [f"{t}_{c}" for t in ("smetN", "smetO") for c in ("dH", "M", "VI")] + \
       [f"{t}_{c}" for t in ("cmetX", "cmetY") for c in ("dH", "I", "Hc")]
BRANCH_ID = {"minplus": 0, "maxplus": 1}
NULL_KEYS = {"C": (3, rz.curveball), "C_col": (4, rz.column_preserving), "C_row": (5, rz.row_preserving)}


def population_row(contexts):
    sN, sO, cX, cY, EI = fc.pooled(contexts)
    return np.concatenate([sN[-1], sO[-1], cX, cY]), EI


# ------------------------------------------------------------------ bernoulli

def _bernoulli_size(args):
    n, m, per_level, out = args
    rng = np.random.default_rng(SEED)
    si, se, cm = [], [], []
    for p in rz.density_levels():
        for k in range(per_level):
            C = rz.bernoulli_context(n, m, p, rng)
            a, b, X, Y = rz.single_context_entropies(C)
            for arr, store in ((a, si), (b, se)):
                for j, (dh, mm, vi) in enumerate(arr):
                    store.append((f"x{j + 1}", dh, mm, vi, p, k))
            cm += [("X", *X, p, k), ("Y", *Y, p, k)]
    cols = ["name", "DeltaH_norm", "M_norm", "VI_norm", "density", "ctx"]
    d = Path(out) / "bernoulli"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{n}x{m}"
    pd.DataFrame(si, columns=cols).to_csv(d / f"SMETint_{tag}.csv", index=False)
    pd.DataFrame(se, columns=cols).to_csv(d / f"SMEText_{tag}.csv", index=False)
    pd.DataFrame(cm, columns=["type", "DeltaH_norm", "I_norm", "Hcond_norm", "density", "ctx"]) \
        .to_csv(d / f"CMET_{tag}.csv", index=False)
    return tag


def cmd_bernoulli(a):
    sizes = [tuple(int(x) for x in s.split("x")) for s in a.sizes]
    tasks = [(n, m, 50 if n * m >= 1600 else 250, a.out) for n, m in sizes]
    with Pool(a.processes) as pool:
        for tag in pool.imap_unordered(_bernoulli_size, tasks):
            print(f"bernoulli {tag} done", flush=True)


# ------------------------------------------------------------------ ensembles

def _ensemble_cell(args):
    p, eps, n_ctx, reps, seed = args
    rng = np.random.default_rng(seed)
    return [[p, eps, r, *population_row(rz.bernoulli_ensemble(p, eps, n_ctx, (23, 11), rng))[0]]
            for r in range(reps)]


def cmd_ensembles(a):
    cells = [(p, e, a.n_ctx, 100, [SEED, 1, i, j]) for i, p in enumerate(rz.density_levels())
             for j, e in enumerate((0, .1, .25, .5, .75, 1))]
    with Pool(a.processes) as pool:
        rows = [r for res in pool.imap(_ensemble_cell, cells) for r in res]
    d = Path(a.out) / "ensembles"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["density", "eps", "rep", *COLS]).to_csv(d / "pooled_geometry_grid.csv", index=False)
    print(f"ensembles -> {d}", flush=True)


# ------------------------------------------------------------------ worms at every phi

def worms_and_grid(conc, branch, a):
    w = build_thesis_worms(Path(a.data_dir) / f"compiled_data_conc_{conc}.mat", seed=SEED)
    structs, n2i = build_odor_neuron_structs(w["data"]["compiled_LR_ordered"], w["data"]["neuron_names"])
    R = compute_peak_matrix(structs, n2i)
    grid = phi_grid(branch, R=R, context_dir=Path(a.contexts_dir) / f"od_by_neuron{conc}_23x11_{branch}")
    return w, grid


def _matched_task(args):
    conc, bi, i, phi, n_ctx, p, e = args
    seed = [SEED, 2, i, bi] if conc == 4 else [SEED, 2, conc, i, bi]
    rng = np.random.default_rng(seed)
    return [[phi, r, *population_row(rz.bernoulli_ensemble(p, e, n_ctx, (23, 11), rng))[0]] for r in range(REPS)]


def cmd_matched(a):
    d = Path(a.out) / "nulls"
    d.mkdir(parents=True, exist_ok=True)
    for conc in a.conc:
        for branch in a.branch:
            w, grid = worms_and_grid(conc, branch, a)
            tasks, cons = [], []
            for i, phi in enumerate(grid):
                B = binarize_branch(w["contexts"], phi, branch)
                p, D, e = rz.ensemble_consistency(B)
                cons.append([phi, p, D, e])
                tasks.append((conc, BRANCH_ID[branch], i, phi, len(B), p, e))
            with Pool(a.processes) as pool:
                rows = [r for res in pool.imap(_matched_task, tasks) for r in res]
            tag = f"{branch}_{100 + conc}"
            pd.DataFrame(rows, columns=["phi", "rep", *COLS]).to_csv(d / f"B_population_{tag}.csv", index=False)
            pd.DataFrame(cons, columns=["phi", "p_hat", "D", "eps_hat"]).to_csv(d / f"worm_consistency_{tag}.csv", index=False)
            print(f"matched {tag}: {len(grid)} thresholds", flush=True)


def _margin_task(args):
    key, bi, i, phi, B, odors, neurons = args
    sid, fn = NULL_KEYS[key]
    rng = np.random.default_rng([SEED, sid, i, bi])
    pop, ind = [], []
    for r in range(REPS):
        v, EI = population_row([fn(C, rng) for C in B])
        pop.append([phi, r, *v])
        E1, I1 = EI[0]                       # averaged worm
        for kind, M, names in (("neuron", I1, neurons), ("odorant", E1, odors)):
            for nm, (x, y, z) in zip(names, fc.smet(M)[:-1]):
                ind.append((phi, r, kind, nm, x, y, z))
    return key, pop, ind


def cmd_margins(a):
    d = Path(a.out) / "nulls"
    d.mkdir(parents=True, exist_ok=True)
    for conc in a.conc:
        for branch in a.branch:
            w, grid = worms_and_grid(conc, branch, a)
            odors, neurons = list(w["data"]["odor_names"]), list(w["data"]["neuron_names"])
            obs, tasks = [], []
            for i, phi in enumerate(grid):
                B = binarize_branch(w["contexts"], phi, branch).astype(np.uint8)
                obs.append([phi, *population_row(list(B))[0]])
                tasks += [(k, BRANCH_ID[branch], i, phi, B, odors, neurons) for k in NULL_KEYS]
            tag = f"{branch}_{100 + conc}"
            pd.DataFrame(obs, columns=["phi", *COLS]).to_csv(d / f"observed_{tag}.csv", index=False)
            res = {k: ([], []) for k in NULL_KEYS}
            with Pool(a.processes) as pool:
                for key, pop, ind in pool.imap(_margin_task, tasks):
                    res[key][0].extend(pop)
                    res[key][1].extend(ind)
            for key, (pop, ind) in res.items():
                pd.DataFrame(pop, columns=["phi", "rep", *COLS]).to_csv(d / f"{key}_population_{tag}.csv", index=False)
                pd.DataFrame(ind, columns=["phi", "rep", "kind", "name", "DeltaH_norm", "M_norm", "VI_norm"]) \
                    .to_csv(d / f"{key}_individual_{tag}.csv.gz", index=False)
            print(f"margins {tag}: {len(grid)} thresholds", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bernoulli", "ensembles", "matched", "margins"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(ROOT / "results"))
        p.add_argument("--processes", type=int, default=2)
        if name == "bernoulli":
            p.add_argument("--sizes", nargs="+", default=["10x10", "20x20", "30x30", "40x40", "10x40", "23x11"])
        if name == "ensembles":
            p.add_argument("--n-ctx", type=int, default=19)
        if name in ("matched", "margins"):
            p.add_argument("--data-dir", default=str(ROOT / "contexts"))
            p.add_argument("--contexts-dir", default=str(ROOT / "contexts"))
            p.add_argument("--conc", type=int, nargs="+", default=[4])
            p.add_argument("--branch", nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    a = ap.parse_args()
    print(f"fastclasic == fca.py + calculate_entropies (max |diff| {fc.check_against_reference(100):.1e})", flush=True)
    {"bernoulli": cmd_bernoulli, "ensembles": cmd_ensembles, "matched": cmd_matched, "margins": cmd_margins}[a.cmd](a)


if __name__ == "__main__":
    main()
