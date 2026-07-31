# Mollified Active Subspace

Official implementation of

> Mollified Active Subspace
> Romain Verdière, Clémentine Prieur, Olivier Zahm
> Preprint, 2025

[[Paper]](https://inria.hal.science/hal-05194926v2/file/article_MAS.pdf)

---

## Overview

This repository contains the official implementation of the method presented in our preprint.

The project provides

- dataset generation code
- training and evaluation code
- figures generation scripts

Mollified Active Subspace (MAS) is a linear dimension reduction method which improves upon  [Active Subspace](https://arxiv.org/pdf/1304.2070)(AS).
MAS uses well chosen additional gradient samples to prevent AS selection errors, see Section 2 of the paper for a comrehensive introtuction to the topic.

To approximate a model $u : \mathbb{R}^d \rightarrow \mathbb{R}$, the algorithm learns a linear feature map $U_m \in \mathbb{R}^{d \times m}$ for $m \ll d$ and then regresses the model output $u(X)$ against the learnt features $U_m^\top X$. 
The feature map is obtained by computing the first $m$ eigenvectoris of an Active Subspace matrix.

## Installation

The wave_model dataset generator relies on
[FEniCS](https://github.com/fenics). We recommend
installing the dependencies with Conda through the `conda-forge` channel.

Clone the repository:

```bash
git clone https://github.com/rverdiere/mollified_as.git
cd mollified_as
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate mollified_as
```

Verify the installation:

```bash
python -c "import fenics, numpy, matplotlib, torch; print('Installatcon successful')"
```
## How to use

The dataset generator can be run with: 
```bash 
python datasets/generate_wave_model.py 
```

The experiments from Sections 5.1 and 5.2 can be run with: 
```bash 
python sum_of_sin.py
python wave_model.py
```

After runing the experiements with the previous commands, Figures 1 to 8 can be generated using:
```bash 
python fig1.py
python fig2.py
python fig3.py
python fig4.py
python fig5.py
python fig6-7.py
python fig8.py
```

## Repository Structure

```text
├── datasets
│   ├── generate_wave_model.py  # Generate samples for the wave_model
│   └── wave_model              # Wave_model dataset
├── fig1.py     # Script to generate fig1.png
├── fig2.py     # Script to generate fig2.png
├── fig3.py     # Script to generate fig3.png
├── fig4.py     # Script to generate fig4.png
├── fig5.py     # Script to generate fig5.png from results stored in results/sum_of_sin
├── fig6-7.py   # Script to generate fig6.png and fig7.png from results stored in results/sum_of_sin
├── fig8.py     # Script to generate fig8.png from results stored in results/wave_model
├── figures
│   ├── sum_of_sin  # Sum_of_sin figures in png format
│   └── wave_model  # Wave_model figures in png format
├── LICENSE
├── README.md
├── results
│   ├── sum_of_sin  # Simulation results of the sum_of_sin model in csv format
│   └── wave_model  # Simulation results of the wave_model in csv format
├── src
│   ├── active_sub.py           # Active Subspace implementation
│   ├── benchmark_functions.py  # Sum of sine benchmark function
│   ├── mas.py                  # Mollified Active Subspace implementation 
│   ├── plots.py                # Function to plot error bounds and approximation errors
│   └── train_featuremap.py     # Feature map training function
├── sum_of_sin.py   #Compute error bounds and approximation errors for the sum_of_sin model
└── wave_model.py   #Compute error bounds and approximation errors for the wave_model
```

### Main Components

- **`datasets/`** contains utilities for generating the synthetic datasets used in the paper.
- **`src/`** contains the core implementation of the proposed algorithms.
- **`results/`** contains the simulation results in csv format.
- **`results/`** contains the figures in png format.
- **`sum_of_sin.py`** Script to run the expiriments from Section 5.1
- **`wave_model.py`** Script to run the expiriments from Sections 5.2

