"""compute_clasic.py: population and individual CLasIC tables for the thesis analysis.

For each concentration and K-FCA branch:
  1. builds the averaged worm and the K virtual worms (src/utils.build_thesis_worms);
  2. sweeps the threshold of existence phi (R >= phi for 'minplus', R <= phi for 'maxplus');
  3. computes all formal concepts of every worm (src/fca.py) and the four entropy
     triangles (src/calculate_entropies.py):
       SEDF_neurons_*  SMET of the intents  (worm 1 = averaged worm, worm 'ALL' = pooled)
       SEDF_odorants_* SMET of the extents
       CMET_*          split CMET of the pooled extents/intents (types X and Y)

Usage (from the repository root):
    python scripts/compute_clasic.py --data-dir contexts --out results

`--data-dir` must contain compiled_data_conc_{4,5,6}.mat from Lin et al. (2023).
If `contexts/od_by_neuron{c}_23x11_{branch}/` exists (contexts exported by the K-FCA
live script of KFCA_for_neuraldata), its thresholds are used; otherwise every distinct
entry of R(c) is used (see utils.phi_grid).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calculate_entropies import build_smet_from_extents, build_smet_from_intents, jentropies  # noqa: E402
from fca import build_extents_intents_all_worms  # noqa: E402
from utils import (BRANCHES, binarize_branch, build_odor_neuron_structs, build_thesis_worms,  # noqa: E402
                   compute_peak_matrix, phi_grid)


def run(conc, branch, data_dir, contexts_dir, out, seed):
    worms = build_thesis_worms(Path(data_dir) / f"compiled_data_conc_{conc}.mat", seed=seed)
    data, cont = worms["data"], worms["contexts"]
    structs, n2i = build_odor_neuron_structs(data["compiled_LR_ordered"], data["neuron_names"])
    R = compute_peak_matrix(structs, n2i)
    grid = phi_grid(branch, R=R, context_dir=Path(contexts_dir) / f"od_by_neuron{conc}_23x11_{branch}")
    odors, neurons = data["odor_names"], data["neuron_names"]
    sn, so, cm, summ = [], [], [], []
    for phi in grid:
        B = binarize_branch(cont, phi, branch)
        sn.append(build_smet_from_intents(B, odors, neurons, phi=phi))
        so.append(build_smet_from_extents(B, odors, neurons, phi=phi))
        E, I = build_extents_intents_all_worms(B, odors, neurons)
        c = jentropies(E, I)
        c["phi"] = phi
        cm.append(c)
        summ.append(dict(conc=conc, branch=branch, phi=phi, n_worms=worms["K"] + 1,
                         density_avg_worm=float(B[0].mean()), n_concepts_pooled=len(E)))
    tag = f"{100 + conc}_{branch}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(sn, ignore_index=True).to_csv(out / f"SEDF_neurons_{tag}.csv", index=False)
    pd.concat(so, ignore_index=True).to_csv(out / f"SEDF_odorants_{tag}.csv", index=False)
    pd.concat(cm, ignore_index=True).to_csv(out / f"CMET_{tag}.csv", index=False)
    pd.DataFrame(summ).to_csv(out / f"summary_{tag}.csv", index=False)
    return tag, len(grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "contexts"))
    ap.add_argument("--contexts-dir", default=str(ROOT / "contexts"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--conc", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--branch", nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    for conc in a.conc:
        for branch in a.branch:
            tag, n = run(conc, branch, a.data_dir, a.contexts_dir, a.out, a.seed)
            print(f"{tag}: {n} thresholds -> {a.out}")


if __name__ == "__main__":
    main()
