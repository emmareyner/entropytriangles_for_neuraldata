# Entropy triangles and concept lattices for the C. elegans olfactory sensorium

This repository contains the code of **CLasIC** (*Concept Lattices as Information Channels*), used to study the *C. elegans* odorant–neuron system by combining:

- K - Formal Concept Analysis (KFCA) of the graded odorant–neuron responses, swept over the threshold of existence φ;
- multivariate entropy triangles: the Source Multivariate Entropy Triangle (SMET) of odorants and of neurons, and the split Channel Multivariate Entropy Triangle (CMET) from odorants to neurons;
- random references that separate what is imposed by the closure of FCA, what follows from the margins of each context (neuron specificity, odorant sparseness) and what is specific to the data.

Each concept lattice is read as an information channel: the extents (sets of odorants) are the source and the intents (sets of neurons) the destination. Following the four triangles along φ turns the sequence of lattices into trajectories through information space.

## Repository structure

```text
.
├── src/
│   ├── fca.py                  # Next Closure: all formal concepts of a binary context
│   ├── calculate_entropies.py  # SMET (sentropies) and CMET (jentropies)
│   ├── fastclasic.py           # vectorised CLasIC core and curveball sampler (random references)
│   ├── randomization.py        # Bernoulli contexts/ensembles and margin-preserving randomizations
│   ├── utils.py                # data loading, R(c), virtual worms, thresholding
│   └── visualize_triangles.py  # SMET, CMET and custom ternary plots
├── scripts/
│   ├── compute_clasic.py       # CLasIC tables for every concentration and branch
│   └── compute_references.py   # Bernoulli geometry, matched ensembles, margin-preserving nulls
├── notebooks/
│   ├── virtual_worms.ipynb
│   └── custom_triangle_playground.ipynb
├── contexts/                   # input data (not distributed, see below)
├── results/                    # generated tables
└── figures/                    # generated figures
```

## Data

No experimental data are distributed with this repository. The analysis uses the multineuronal calcium recordings of

- Lin A, Qin S, Casademunt H, Wu M, Hung W, Cain G, et al. *Functional imaging and quantification of multineuronal olfactory responses in C. elegans*. Science Advances 9(9):eade1249, 2023. DOI: 10.1126/sciadv.ade1249.

The data are available on Zenodo (DOI: 10.5281/zenodo.7563053, *CElegans Chemosensory Data / Compiled Neural Data*). Download `compiled_data_conc_4.mat`, `compiled_data_conc_5.mat` and `compiled_data_conc_6.mat` and place them in `contexts/`.

Optionally, the binary contexts exported by the K-FCA live script of [KFCA_for_neuraldata](https://github.com/emmareyner/KFCA_for_neuraldata) can be placed in `contexts/od_by_neuron{4,5,6}_23x11_{minplus,maxplus}/`. When these folders exist, their thresholds (read from the file names, e.g. `od_by_neuron4_23x11_minplus_structural_0.2500.csv`) are used, which reproduces the published results exactly. Otherwise every distinct entry of the graded matrix R(c) is used as a threshold; this is a superset of the structural thresholds and gives the same trajectories at the shared values of φ.

## Pipeline

1. **Graded matrices.** For every odorant–neuron pair, the signed mean response in a window of ±2 s around the largest absolute deflection of the trace. `utils.compute_peak_matrix` gives the matrix R(c) of the K-FCA analysis (from the mean trace of all presentations).
2. **Virtual worms.** The recordings do not identify the animal of origin, so several complete matrices are assembled from single presentations (`utils.build_thesis_worms`). For each odorant–neuron pair, K distinct traces are drawn **without replacement** (seed 42) and trace k goes to virtual worm k, so no trace is used in more than one worm; K is the smallest number of traces of any pair (18 at 10⁻⁴, 12 at 10⁻⁵ and 10⁻⁶). The averaged worm, the entry-wise mean of the K virtual worms, is stored as worm 1. Individual neuron and odorant trajectories use the averaged worm; population trajectories pool its concepts with those of the virtual worms.
3. **Thresholding.** Two K-FCA branches: `minplus` keeps an incidence when R(g, m) ≥ φ (activation, φ ≥ 0) and `maxplus` when R(g, m) ≤ φ (inhibition, φ < 0) (`utils.binarize_branch`).
4. **Concepts and triangles.** `fca.compute_concepts` enumerates all formal concepts; the extent and intent matrices are treated as equally likely observations, and `calculate_entropies` computes SMET and CMET.
5. **References** (`scripts/compute_references.py`):
   - single Bernoulli contexts of several sizes and densities (reachable geometry of one context);
   - Bernoulli ensembles of N contexts with a redraw probability ε, and ensembles matched at every φ to the density p̂ and consistency ε̂ of the worms (reachable geometry of the pooled description);
   - margin-preserving randomizations of every worm: swap randomization (curveball; both margins), column-preserving (neuron specificity) and row-preserving (odorant sparseness).

## Installation

Python 3.12 was used. With conda:

```bash
conda env create -f environment.yml
conda activate celegans-triangles
```

or with pip:

```bash
pip install numpy pandas matplotlib mpltern scipy ipywidgets ipython jupyter
pip install numba   # optional, speeds up the closed forms for large random contexts
```

## Usage

```bash
# CLasIC tables (SEDF_neurons_*, SEDF_odorants_*, CMET_*) for the three concentrations and both branches
python scripts/compute_clasic.py --data-dir contexts --out results

# random references
python scripts/compute_references.py bernoulli --sizes 10x10 20x20 30x30 40x40 10x40 23x11
python scripts/compute_references.py ensembles
python scripts/compute_references.py matched --conc 4 5 6
python scripts/compute_references.py margins --conc 4
```

Self-tests: `python src/fca.py` compares the concept enumeration with brute force, and `fastclasic.check_against_reference()` checks the fast core against `fca.py` + `calculate_entropies.py`.

The notebook `notebooks/virtual_worms.ipynb` runs the same pipeline interactively for one concentration and branch and draws the four triangles. `notebooks/custom_triangle_playground.ipynb` is a small playground for custom ternary plots.

## Changes in version 0.2.0 (November 2026)

- `fca.compute_concepts` now implements Next Closure correctly. Earlier versions built the candidate as A ∪ {i} instead of (A ∩ {0, …, i−1}) ∪ {i} and returned only a subset of the formal concepts; results computed with them should be recomputed.
- Virtual worms are sampled without replacement and the averaged worm is included as worm 1.
- Negative thresholds are analysed with the inhibitory rule R ≤ φ.
- The φ grid no longer requires the K-FCA contexts, and the random references used in the thesis are included.

## Citation

If you use this code, please cite:

- E. Reyner-Fuentes, C. Peláez-Moreno, F. J. Valverde-Albacete. *Qualitative and Quantitative Analyses of C. elegans' Perception with Concept Lattices as Information Channels*. CONCEPTS 2025, Springer LNCS, pp. 264–280. DOI: 10.1007/978-3-032-03364-2_17.
- E. Reyner-Fuentes, C. Peláez-Moreno, F. J. Valverde-Albacete. *On the Limits of Information Transmission in Concept Lattices*. CONCEPTS 2026, Springer LNCS, pp. 192–209. DOI: 10.1007/978-3-032-34742-8_13.
- E. Reyner-Fuentes. *Visualising Information Representations in Brains: the Case of C. elegans*. PhD thesis, Universidad Carlos III de Madrid, 2026.

The entropy triangles follow F. J. Valverde-Albacete and C. Peláez-Moreno, *The evaluation of data sources using multivariate entropy tools* (Expert Systems with Applications, 2017) and *Assessing information transmission in data transformations with the channel multivariate entropy triangle* (Entropy, 2018).

## Contact

Questions and bug reports: please open an issue in this repository.

## License

MIT. See `LICENSE`.
