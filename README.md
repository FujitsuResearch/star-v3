# Repository for the paper "STAR-Magic Mutation: Even More Efficient Analog Rotation Gates for Early Fault-Tolerant Quantum Computer"

This repository contains the codes to reproduce the numerical results and figures in the paper ["STAR-Magic Mutation: Even More Efficient Analog Rotation Gates for Early Fault-Tolerant Quantum Computer"](https://arxiv.org/abs/2603.22891).

## Scope

The main entry points are the Jupyter notebooks in `paper/`. Each notebook can generate a specific subset of figures in the paper, and the surrounding files exist to support those figure-generation workflows.

## Environment Setup with `uv`

You can install the project environment with the Python package manager `uv`:

```bash
uv sync
```



## Guide for Reproducing Figures

The files in `paper/` are organized around reproducing individual figures from the manuscript:

- `paper/generate_fig_for_error_rate.ipynb`: generates the data and plots used for Fig. 6-8.
- `paper/generate_fig_for_tradeoff.ipynb`: generates the trade-off analysis shown in Fig. 9.
- `paper/generate_fig_for_bound.ipynb`: generates the bound analysis shown in Fig. 11.
- `paper/generate_fig_for_TEPAI.ipynb`: generates the results used for Fig. 12-13.
- `paper/modules_for_v3.py`: shared helper code used by the figure-generation workflow.
- `paper/2603.22891v1.pdf`: The PDF file of the paper ["STAR-Magic Mutation: Even More Efficient Analog Rotation Gates for Early Fault-Tolerant Quantum Computer"](https://arxiv.org/abs/2603.22891).

## Repository Layout

- `paper/`: The primary working directory for reproducing the paper's figures.
- `src/`: Source codes for analyzing error parameters and providing resource estimation routines.
- `output/`: Figures used for the paper.
- `pyproject.toml`: project metadata and dependency definitions for the `uv` environment.
- `uv.lock`: locked dependency set for reproducible installation.

## Source Codes in `src/`

The contents in `src/` are summarized as follows:

- `src/starsim/core/preprocess.py`: generate several error parameters required for the analyses of the STAR-magic mutation.
- `src/starsim/simulator/manager.py`: provides the resource-estimation entry point used in the analysis.
- `src/starsim/task/`: defines the task format and Hamiltonian representations needed by the notebooks.
- `src/starsim/util/plot_parameters.py`: centralizes plotting parameters used when generating publication figures.

