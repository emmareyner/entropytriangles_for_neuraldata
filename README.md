# Entropy triangles and concept lattices for the C. elegans olfactory sensorium

This repository contains the code used to study the *C. elegans* odorant–neuron system through:

- Formal Concept Analysis (FCA)
- entropy triangles (SMET and CMET)

The project is built around a simple idea: instead of looking only at which neuron responds to which odorant, we follow how the whole odorant–neuron relation reorganizes as the activation threshold changes. This makes it possible to compare informational structure and relational structure within a single workflow.

## Repository structure

```text
.
├── src/
│   ├── calculate_entropies.py
│   ├── fca.py
│   ├── utils.py
│   └── visualize_triangles.py
├── notebooks/
│   ├── virtual_worms.ipynb
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

- `utils.py`  
  General project helpers for data loading, virtual worm generation, preprocessing from traces to graded contexts, thresholding into formal contexts, small mapping utilities, and summary plotting.

## Notebooks

- `virtual_worms.ipynb`  
  Builds virtual worms from the graded response data and visualizes the corresponding triangles.

- `custom_triangle_playground.ipynb`  
  A lightweight playground to experiment with custom ternary plots using the project plotting functions.

## What the project does

For each concentration and each threshold value \(\varphi\), the graded odorant–neuron response matrix is binarized into a formal context. From there, the pipeline combines three complementary views:

1. **FCA**  
   The thresholded context is transformed into a concept lattice, which reveals the relational organization of odorants and neurons.

2. **Entropy triangles**  
   SMET and CMET provide compact information-theoretic summaries of source organization and source–destination transmission.


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

### 2. Custom ternary plots

Open:

```text
notebooks/custom_triangle_playground.ipynb
```

This notebook is a lightweight playground to experiment with custom ternary plots using the project plotting functions.

## Outputs

The notebooks are written so that each relevant result can be saved explicitly.

Typical outputs include:

- CSV tables in `results/`
  - custom triangle data
  - direct/FCA-based SMET or CMET tables

- figures in `figures/`
  - entropy triangles
  - comparison figures
  - custom ternary plots

## Notes on interpretation

This repository includes FCA and triangle-based analyses, but they serve different roles.

- The entropy triangles identify informative structural transitions.
- FCA reveals the relational organization behind those transitions.

The two views are meant to be read together.

## Citation

If you use this repository, please cite the associated paper and related methodological work.

You can replace this section with the exact references you want to expose publicly.


## Contact

For questions about the code or the analysis, please contact the repository maintainer.