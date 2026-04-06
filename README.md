# Triangles, FCA, and perception regimes in the *C. elegans* olfactory sensorium

This repository contains the code used to study the *C. elegans* odorant–neuron system through:

- Formal Concept Analysis (FCA)
- entropy triangles (SMET and CMET)
- similarity-based perception-regime detection across the activation threshold \(\varphi\)

The project is built around a simple idea: instead of looking only at which neuron responds to which odorant, we follow how the whole odorant–neuron relation reorganizes as the activation threshold changes. This makes it possible to compare informational structure, relational structure, and regime structure within a single workflow.

## Repository structure

```text
.
├── src/
│   ├── calculate_entropies.py
│   ├── fca.py
│   ├── similarities.py
│   ├── utils.py
│   └── visualize_triangles.py
├── notebooks/
│   ├── virtual_worms.ipynb
│   ├── perception_regimes.ipynb
│   └── custom_triangle_playground.ipynb
├── contexts/
├── results/
├── figures/
└── README.md
```

## Source modules

- `calculate_entropies.py`  
  Functions used to compute SMET and CMET magnitudes.

- `visualize_triangles.py`  
  Plotting functions for SMET, CMET, custom triangles, and interactive views.

- `fca.py`  
  FCA utilities, including concept computation and extent/intent construction.

- `similarities.py`  
  Similarity, boundary-score, and regime-detection functions. This module contains the Jaccard- and Dice-based comparison logic used for anchor-guided and anchor-free segmentation, as well as the helpers required to build perception regimes from thresholded contexts.

- `utils.py`  
  General project helpers for data loading, virtual worm generation, preprocessing from traces to graded contexts, thresholding into formal contexts, small mapping utilities, and summary plotting.

## Notebooks

- `virtual_worms.ipynb`  
  Builds virtual worms from the graded response data and visualizes the corresponding triangles.

- `perception_regimes.ipynb`  
  Detects perception regimes from thresholded formal contexts. Includes:
  - anchor-guided segmentation using Jaccard + \(\Delta\)
  - anchor-free recursive segmentation used as a robustness check
  - single-concentration and multi-concentration regime summaries

- `custom_triangle_playground.ipynb`  
  A lightweight playground to experiment with custom ternary plots using the project plotting functions.

## What the project does

For each concentration and each threshold value \(\varphi\), the graded odorant–neuron response matrix is binarized into a formal context. From there, the pipeline combines three complementary views:

1. **FCA**  
   The thresholded context is transformed into a concept lattice, which reveals the relational organization of odorants and neurons.

2. **Entropy triangles**  
   SMET and CMET provide compact information-theoretic summaries of source organization and source–destination transmission.

3. **Perception regimes**  
   Thresholded contexts are compared along the \(\varphi\)-axis to identify contiguous intervals of structural stability, each summarized by a medoid context.

The result is a framework for moving from graded neural-response data to interpretable statements about sensory organization.

## Data and repository policy

The `contexts/` folder is intentionally uploaded empty.

The thresholded formal contexts used in this project are derived from experimental data that do not belong to us, so they are not redistributed in this repository. To reproduce the full pipeline, users should obtain the original dataset from the source publication:

- Lin A, Qin S, Casademunt H, Wu M, Hung W, Cain G, et al. *Functional Imaging and Quantification of Multineuronal Olfactory Responses in C. elegans*. *Science Advances*. 2023;9(9):eade1249. DOI: 10.1126/sciadv.ade1249.

The preprocessing steps that transform the original graded recordings into the formal contexts is also included in this repository. You only need to obtain the "compiled_data_conc_X.mat" from the original repository.

In other words:

- this repository contains the preprocessing and analysis code
- the original biological recordings must be obtained from Lin et al.

## Expected data layout

Once the thresholded contexts have been generated, they are expected to be stored in `contexts/`.

Typical folders are:

```text
contexts/
├── od_by_neuron4_23x11_minplus/
├── od_by_neuron5_23x11_minplus/
├── od_by_neuron6_23x11_minplus/
├── od_by_neuron4_23x11_maxplus/
├── od_by_neuron5_23x11_maxplus/
└── od_by_neuron6_23x11_maxplus/
```

Each CSV is expected to contain one binary formal context for one threshold value \(\varphi\). The threshold value is read from the filename, which is expected to be similar to "od_by_neuron4_23x11_minplus_structural_-0.00033441824408787221.csv", with the \(\varphi\) value at the end.

The notebooks also assume the existence of:

- `results/` for intermediate tables
- `figures/` for exported figures

## Installation

A standard scientific Python (version 3.12 was used when developing this code) environment is enough. The project relies mainly on:

- `numpy`
- `pandas`
- `matplotlib`
- `mpltern`
- `scipy`
- `ipywidgets`
- `IPython`

You can install the dependencies with pip:

```bash
pip install numpy pandas matplotlib mpltern scipy ipywidgets ipython
```

If you work in Jupyter and want the interactive widgets to behave correctly, make sure that `ipywidgets` is enabled in your environment.

## How to use the repository

### 1. Virtual worms and triangle trajectories

Open:

```text
notebooks/virtual_worms.ipynb
```

This notebook:

- loads the graded response data
- builds virtual worms
- computes thresholded contexts across \(\varphi\)
- visualizes SMET and CMET trajectories

### 2. Perception regimes

Open:

```text
notebooks/perception_regimes.ipynb
```

This notebook:

- loads thresholded contexts from `contexts/`
- computes anchor-guided regime boundaries with Jaccard + \(\Delta\)
- computes an anchor-free recursive segmentation as a robustness check
- maps regimes to linear and logarithmic activity scales
- plots regime summaries for one or several concentrations

### 3. Custom ternary plots

Open:

```text
notebooks/custom_triangle_playground.ipynb
```

This notebook is a lightweight playground to experiment with custom ternary plots using the project plotting functions.

## Main methodological choices

### Anchor-guided regime detection

The main regime analysis is based on:

- Jaccard similarity between thresholded contexts
- a \(\Delta\)-based groundedness correction to reduce instability in sparse regions
- triangle-derived anchors to define informative search intervals
- medoid selection within each regime using ordinary Jaccard similarity

### Anchor-free segmentation

The fully automatic segmentation is used as a robustness check. It removes the anchors and applies a recursive split strategy. Its role is not to replace the anchor-guided analysis, but to test whether a nontrivial regime scaffold is already visible from context-to-context similarity alone.

## Outputs

The notebooks are written so that each relevant result can be saved explicitly.

Typical outputs include:

- CSV tables in `results/`
  - regime tables
  - mapped regime summaries
  - custom triangle data
  - direct/FCA-based SMET or CMET tables

- figures in `figures/`
  - entropy triangles
  - regime bar summaries
  - comparison figures
  - custom ternary plots

## Notes on interpretation

This repository includes both triangle-based analyses and regime-detection procedures, but they serve different roles.

- The entropy triangles identify informative structural transitions.
- FCA reveals the relational organization behind those transitions.
- The regime analysis formalizes intervals of relative structural stability along the threshold sweep.

The three views are meant to be read together.

## Citation

If you use this repository, please cite the associated paper and related methodological work.

You can replace this section with the exact references you want to expose publicly.

## License

Add the license you want for the public release, for example:

- MIT
- BSD-3-Clause
- GPL-3.0

If no license is added, others may not be able to reuse the code legally.

## Contact

For questions about the code or the analysis, please contact the repository maintainer.